// rope.cu — Rotary Position Embedding (in-place, SM 8.7)
//
// Applies RoPE to Q and K tensors before attention.
// Fused: processes both Q and K in one kernel launch.
//
// Normal RoPE uses adjacent pairs (2i, 2i+1). Qwen/Gemma/Phi use the
// GPT-NeoX layout from llama.cpp: pair i with i + head_dim/2.
//
// RoPE formula for pair (a, b):
//   q'[a] = q[a] * cos(θ) - q[b] * sin(θ)
//   q'[b] = q[a] * sin(θ) + q[b] * cos(θ)
//   where θ = position / (theta_base ^ (2i / head_dim))

#include "jllm_kernels.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace jllm {

// ── Precomputed RoPE cos/sin tables ─────────────────────────────────────────
// Each call to rope_precompute_table() builds and stores one (theta, half_dim)
// table on the device. The rope_inplace* wrappers look up the table by theta_base
// and dispatch to the _tbl kernel variants when a match is found. This eliminates
// expensive powf + cosf + sinf from every decode step (once per layer per token).

static constexpr int MAX_ROPE_TABLES = 8;

struct RopeTableEntry {
    float  theta_base;
    int    half_dim;
    float* d_cos;    // [max_seq × half_dim]
    float* d_sin;
    int    max_seq;
};

static RopeTableEntry g_rope_tbl[MAX_ROPE_TABLES];
static int            g_n_rope_tbl = 0;

static bool rope_table_enabled() {
    static const bool on = [] {
        const char* v = getenv("JLLM_ROPE_TABLE");
        return !v || strcmp(v, "0") != 0;
    }();
    return on;
}

// Find an existing table entry or return nullptr.
static const RopeTableEntry* find_rope_tbl(float theta_base, int half_dim) {
    for (int i = 0; i < g_n_rope_tbl; i++) {
        if (g_rope_tbl[i].theta_base == theta_base &&
            g_rope_tbl[i].half_dim  == half_dim) {
            return &g_rope_tbl[i];
        }
    }
    return nullptr;
}

// GPU kernel: build cos/sin table for position × freq pair.
// cos_tbl[p * half_dim + i] = cos(p / theta ^ (i / half_dim))
// Note: freq formula uses full head_dim = 2 * half_dim in denominator.
__global__ void rope_build_table_kernel(float* __restrict__ cos_tbl,
                                        float* __restrict__ sin_tbl,
                                        int half_dim, int max_seq,
                                        float theta_base)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_seq * half_dim) return;
    const int pos      = idx / half_dim;
    const int pair_idx = idx % half_dim;
    const int head_dim = half_dim * 2;
    // Mirror the existing rope kernel formula exactly so the build table and
    // runtime trig paths produce bit-identical values under --use_fast_math.
    const float freq  = 1.0f / powf(theta_base, (2.0f * pair_idx) / head_dim);
    const float angle = (float)pos * freq;
    cos_tbl[idx] = cosf(angle);
    sin_tbl[idx] = sinf(angle);
}

void rope_precompute_table(int max_seq, int head_dim, float theta_base) {
    if (!rope_table_enabled()) return;
    const int half_dim = head_dim / 2;
    if (find_rope_tbl(theta_base, half_dim)) return;   // already built
    if (g_n_rope_tbl >= MAX_ROPE_TABLES) {
        fprintf(stderr, "[rope] WARNING: MAX_ROPE_TABLES reached; table for "
                "theta=%g hd=%d skipped\n", theta_base, head_dim);
        return;
    }

    const int64_t n = (int64_t)max_seq * half_dim;
    float* d_cos = nullptr;
    float* d_sin = nullptr;
    if (cudaMalloc(&d_cos, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_sin, n * sizeof(float)) != cudaSuccess) {
        cudaFree(d_cos);
        fprintf(stderr, "[rope] cudaMalloc failed for table theta=%g hd=%d\n",
                theta_base, head_dim);
        return;
    }

    const int block = 256;
    const int grid  = ((int)n + block - 1) / block;
    rope_build_table_kernel<<<grid, block>>>(d_cos, d_sin, half_dim, max_seq, theta_base);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[rope] table build kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_cos);
        cudaFree(d_sin);
        return;
    }
    cudaDeviceSynchronize();

    RopeTableEntry& e = g_rope_tbl[g_n_rope_tbl++];
    e.theta_base = theta_base;
    e.half_dim   = half_dim;
    e.d_cos      = d_cos;
    e.d_sin      = d_sin;
    e.max_seq    = max_seq;

    fprintf(stderr, "[rope] precomputed table theta=%g half_dim=%d max_seq=%d "
            "(%.1f MB)\n", theta_base, half_dim, max_seq,
            2.0 * n * sizeof(float) / (1024.0 * 1024.0));
}

void rope_clear_tables() {
    for (int i = 0; i < g_n_rope_tbl; i++) {
        cudaFree(g_rope_tbl[i].d_cos);
        cudaFree(g_rope_tbl[i].d_sin);
    }
    g_n_rope_tbl = 0;
}

// ── Table-based kernel variants ───────────────────────────────────────────────
// cos_tbl / sin_tbl are [max_seq × half_dim]; indexed as [position * half_dim + pair_idx].
// Everything else is identical to the original kernels.

__global__ void rope_kernel_tbl(
    half* __restrict__ q, half* __restrict__ k,
    int n_heads, int n_kv_heads, int head_dim,
    const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
    int half_dim, int position, bool neox)
{
    const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    const int h2       = head_dim / 2;
    const int q_pairs  = n_heads    * h2;
    const int total    = (n_heads + n_kv_heads) * h2;
    if (tid >= total) return;

    const bool is_q = tid < q_pairs;
    int head, pair_idx; half* ptr;
    if (is_q) {
        head = tid / h2; pair_idx = tid % h2; ptr = q + head * head_dim;
    } else {
        int kt = tid - q_pairs; head = kt / h2; pair_idx = kt % h2;
        ptr = k + head * head_dim;
    }

    const float c  = cos_tbl[(int64_t)position * half_dim + pair_idx];
    const float s  = sin_tbl[(int64_t)position * half_dim + pair_idx];
    const int i0   = neox ? pair_idx          : pair_idx * 2;
    const int i1   = neox ? pair_idx + h2     : pair_idx * 2 + 1;
    const float v0 = __half2float(ptr[i0]);
    const float v1 = __half2float(ptr[i1]);
    ptr[i0] = __float2half(v0 * c - v1 * s);
    ptr[i1] = __float2half(v0 * s + v1 * c);
}

__global__ void rope_store_kv_fp16_kernel_tbl(
    half* __restrict__ q, half* __restrict__ k, const half* __restrict__ v,
    half* __restrict__ k_cache_dst, half* __restrict__ v_cache_dst,
    int n_heads, int n_kv_heads, int head_dim,
    const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
    int half_dim, int position, bool neox)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int h2      = head_dim / 2;
    const int q_pairs = n_heads * h2;
    const int total   = (n_heads + n_kv_heads) * h2;
    const int kv_dim  = n_kv_heads * head_dim;

    if (tid < total) {
        const bool is_q = tid < q_pairs;
        int head, pair_idx; half* ptr; half* dst = nullptr;
        if (is_q) {
            head = tid / h2; pair_idx = tid % h2; ptr = q + head * head_dim;
        } else {
            int kt = tid - q_pairs; head = kt / h2; pair_idx = kt % h2;
            ptr = k + head * head_dim; dst = k_cache_dst + head * head_dim;
        }
        const float c  = cos_tbl[(int64_t)position * half_dim + pair_idx];
        const float s  = sin_tbl[(int64_t)position * half_dim + pair_idx];
        const int i0   = neox ? pair_idx      : pair_idx * 2;
        const int i1   = neox ? pair_idx + h2 : pair_idx * 2 + 1;
        const float v0 = __half2float(ptr[i0]);
        const float v1 = __half2float(ptr[i1]);
        const half r0  = __float2half(v0 * c - v1 * s);
        const half r1  = __float2half(v0 * s + v1 * c);
        ptr[i0] = r0; ptr[i1] = r1;
        if (!is_q) { dst[i0] = r0; dst[i1] = r1; }
    }
    if (tid < kv_dim) v_cache_dst[tid] = v[tid];
}

__global__ void rope_batched_kernel_tbl(
    half* __restrict__ q, half* __restrict__ k,
    int n_heads, int n_kv_heads, int head_dim,
    const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
    int half_dim, int start_pos, int N, int q_stride, int k_stride, bool neox)
{
    const int i    = blockIdx.y; if (i >= N) return;
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int h2   = head_dim / 2;
    const int total = (n_heads + n_kv_heads) * h2;
    if (tid >= total) return;
    const int q_pairs = n_heads * h2;
    const int position = start_pos + i;
    const bool is_q = tid < q_pairs;
    int head, pair_idx; half* ptr;
    if (is_q) { head = tid / h2; pair_idx = tid % h2; ptr = q + (int64_t)i * q_stride + head * head_dim; }
    else      { int kt = tid - q_pairs; head = kt / h2; pair_idx = kt % h2; ptr = k + (int64_t)i * k_stride + head * head_dim; }
    const float c  = cos_tbl[(int64_t)position * half_dim + pair_idx];
    const float s  = sin_tbl[(int64_t)position * half_dim + pair_idx];
    const int i0   = neox ? pair_idx      : pair_idx * 2;
    const int i1   = neox ? pair_idx + h2 : pair_idx * 2 + 1;
    const float v0 = __half2float(ptr[i0]);
    const float v1 = __half2float(ptr[i1]);
    ptr[i0] = __float2half(v0 * c - v1 * s);
    ptr[i1] = __float2half(v0 * s + v1 * c);
}

__global__ void rope_store_kv_fp16_batched_kernel_tbl(
    half* __restrict__ q, half* __restrict__ k, const half* __restrict__ v,
    half* __restrict__ k_cache_base, half* __restrict__ v_cache_base, int kv_slot,
    int n_heads, int n_kv_heads, int head_dim,
    const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
    int half_dim, int start_pos, int N, int q_stride, int kv_stride, bool neox)
{
    const int i    = blockIdx.y; if (i >= N) return;
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int h2   = head_dim / 2;
    const int q_pairs = n_heads * h2;
    const int total   = (n_heads + n_kv_heads) * h2;
    const int kv_dim  = n_kv_heads * head_dim;
    const int position = start_pos + i;
    half* qi   = q + (int64_t)i * q_stride;
    half* ki   = k + (int64_t)i * kv_stride;
    const half* vi  = v + (int64_t)i * kv_stride;
    half* kdst = k_cache_base + (int64_t)i * kv_slot;
    half* vdst = v_cache_base + (int64_t)i * kv_slot;
    if (tid < total) {
        const bool is_q = tid < q_pairs;
        int head, pair_idx; half* ptr; half* dst = nullptr;
        if (is_q) { head = tid / h2; pair_idx = tid % h2; ptr = qi + head * head_dim; }
        else      { int kt = tid - q_pairs; head = kt / h2; pair_idx = kt % h2; ptr = ki + head * head_dim; dst = kdst + head * head_dim; }
        const float c  = cos_tbl[(int64_t)position * half_dim + pair_idx];
        const float s  = sin_tbl[(int64_t)position * half_dim + pair_idx];
        const int i0   = neox ? pair_idx      : pair_idx * 2;
        const int i1   = neox ? pair_idx + h2 : pair_idx * 2 + 1;
        const float v0 = __half2float(ptr[i0]);
        const float v1 = __half2float(ptr[i1]);
        const half r0  = __float2half(v0 * c - v1 * s);
        const half r1  = __float2half(v0 * s + v1 * c);
        ptr[i0] = r0; ptr[i1] = r1;
        if (!is_q) { dst[i0] = r0; dst[i1] = r1; }
    }
    if (tid < kv_dim) vdst[tid] = vi[tid];
}

__global__ void rope_store_kv_fp16_kernel_dyn_tbl(
    half*       __restrict__ q,
    half*       __restrict__ k,
    const half* __restrict__ v,
    half*       __restrict__ k_cache_layer_base,
    half*       __restrict__ v_cache_layer_base,
    int n_heads, int n_kv_heads, int head_dim, int kv_stride,
    const int*  __restrict__ d_pos,
    const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
    int half_dim, bool neox)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int h2      = head_dim / 2;
    const int q_pairs = n_heads * h2;
    const int total   = (n_heads + n_kv_heads) * h2;
    const int kv_dim  = n_kv_heads * head_dim;
    const int position = *d_pos;
    half* const k_dst = k_cache_layer_base + (int64_t)position * kv_stride;
    half* const v_dst = v_cache_layer_base + (int64_t)position * kv_stride;

    if (tid < total) {
        const bool is_q = tid < q_pairs;
        int head, pair_idx; half* ptr; half* dst = nullptr;
        if (is_q) { head = tid / h2; pair_idx = tid % h2; ptr = q + head * head_dim; }
        else      { int kt = tid - q_pairs; head = kt / h2; pair_idx = kt % h2; ptr = k + head * head_dim; dst = k_dst + head * head_dim; }
        const float c  = cos_tbl[(int64_t)position * half_dim + pair_idx];
        const float s  = sin_tbl[(int64_t)position * half_dim + pair_idx];
        const int i0   = neox ? pair_idx      : pair_idx * 2;
        const int i1   = neox ? pair_idx + h2 : pair_idx * 2 + 1;
        const float v0 = __half2float(ptr[i0]);
        const float v1 = __half2float(ptr[i1]);
        const half r0  = __float2half(v0 * c - v1 * s);
        const half r1  = __float2half(v0 * s + v1 * c);
        ptr[i0] = r0; ptr[i1] = r1;
        if (!is_q) { dst[i0] = r0; dst[i1] = r1; }
    }
    if (tid < kv_dim) v_dst[tid] = v[tid];
}

__global__ void rope_kernel(
    half* __restrict__ q,       // [n_heads × head_dim]
    half* __restrict__ k,       // [n_kv_heads × head_dim]
    int n_heads, int n_kv_heads, int head_dim,
    int position, float theta_base, bool neox)
{
    // One thread per dimension pair
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int total_pairs = (n_heads + n_kv_heads) * half_dim;

    if (tid >= total_pairs) return;

    // Determine if this is a Q or K pair
    const int q_pairs = n_heads * half_dim;
    bool is_q = (tid < q_pairs);

    int head, pair_idx;
    half* ptr;
    if (is_q) {
        head = tid / half_dim;
        pair_idx = tid % half_dim;
        ptr = q + head * head_dim;
    } else {
        int k_tid = tid - q_pairs;
        head = k_tid / half_dim;
        pair_idx = k_tid % half_dim;
        ptr = k + head * head_dim;
    }

    // Compute rotation angle
    float freq = 1.0f / powf(theta_base, (2.0f * pair_idx) / head_dim);
    float angle = position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Rotate the pair. llama.cpp uses GGML_ROPE_TYPE_NEOX for Qwen3:
    // pair_idx rotates with pair_idx + head_dim/2 instead of an adjacent dim.
    int idx0 = neox ? pair_idx : pair_idx * 2;
    int idx1 = neox ? pair_idx + half_dim : pair_idx * 2 + 1;
    float v0 = __half2float(ptr[idx0]);
    float v1 = __half2float(ptr[idx1]);

    ptr[idx0] = __float2half(v0 * cos_val - v1 * sin_val);
    ptr[idx1] = __float2half(v0 * sin_val + v1 * cos_val);
}

__global__ void rope_store_kv_fp16_kernel(
    half*       __restrict__ q,
    half*       __restrict__ k,
    const half* __restrict__ v,
    half*       __restrict__ k_cache_dst,
    half*       __restrict__ v_cache_dst,
    int n_heads, int n_kv_heads, int head_dim,
    int position, float theta_base, bool neox)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int q_pairs = n_heads * half_dim;
    const int total_pairs = (n_heads + n_kv_heads) * half_dim;
    const int kv_dim = n_kv_heads * head_dim;

    if (tid < total_pairs) {
        const bool is_q = tid < q_pairs;
        int head, pair_idx;
        half* ptr;
        half* dst = nullptr;

        if (is_q) {
            head = tid / half_dim;
            pair_idx = tid % half_dim;
            ptr = q + head * head_dim;
        } else {
            const int k_tid = tid - q_pairs;
            head = k_tid / half_dim;
            pair_idx = k_tid % half_dim;
            ptr = k + head * head_dim;
            dst = k_cache_dst + head * head_dim;
        }

        const float freq = 1.0f / powf(theta_base, (2.0f * pair_idx) / head_dim);
        const float angle = position * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);

        const int idx0 = neox ? pair_idx : pair_idx * 2;
        const int idx1 = neox ? pair_idx + half_dim : pair_idx * 2 + 1;
        const float v0 = __half2float(ptr[idx0]);
        const float v1 = __half2float(ptr[idx1]);
        const half r0 = __float2half(v0 * cos_val - v1 * sin_val);
        const half r1 = __float2half(v0 * sin_val + v1 * cos_val);

        ptr[idx0] = r0;
        ptr[idx1] = r1;
        if (!is_q) {
            dst[idx0] = r0;
            dst[idx1] = r1;
        }
    }

    if (tid < kv_dim) {
        v_cache_dst[tid] = v[tid];
    }
}

void rope_inplace(half* q, half* k, int n_heads, int n_kv_heads,
                  int head_dim, int position, float theta_base,
                  bool neox, cudaStream_t stream) {
    const int h2    = head_dim / 2;
    const int total = (n_heads + n_kv_heads) * h2;
    const int block = BLOCK_SIZE;
    const int grid  = (total + block - 1) / block;
    const RopeTableEntry* tbl = rope_table_enabled()
        ? find_rope_tbl(theta_base, h2) : nullptr;
    if (tbl) {
        rope_kernel_tbl<<<grid, block, 0, stream>>>(
            q, k, n_heads, n_kv_heads, head_dim,
            tbl->d_cos, tbl->d_sin, tbl->half_dim, position, neox);
    } else {
        rope_kernel<<<grid, block, 0, stream>>>(
            q, k, n_heads, n_kv_heads, head_dim, position, theta_base, neox);
    }
}

void rope_inplace_store_kv_fp16(
    half* q, half* k, const half* v,
    half* k_cache_dst, half* v_cache_dst,
    int n_heads, int n_kv_heads, int head_dim,
    int position, float theta_base, bool neox, cudaStream_t stream) {
    const int h2     = head_dim / 2;
    const int t_rope = (n_heads + n_kv_heads) * h2;
    const int kv_dim = n_kv_heads * head_dim;
    const int total  = t_rope > kv_dim ? t_rope : kv_dim;
    const int block  = BLOCK_SIZE;
    const int grid   = (total + block - 1) / block;
    const RopeTableEntry* tbl = rope_table_enabled()
        ? find_rope_tbl(theta_base, h2) : nullptr;
    if (tbl) {
        rope_store_kv_fp16_kernel_tbl<<<grid, block, 0, stream>>>(
            q, k, v, k_cache_dst, v_cache_dst,
            n_heads, n_kv_heads, head_dim,
            tbl->d_cos, tbl->d_sin, tbl->half_dim, position, neox);
    } else {
        rope_store_kv_fp16_kernel<<<grid, block, 0, stream>>>(
            q, k, v, k_cache_dst, v_cache_dst,
            n_heads, n_kv_heads, head_dim, position, theta_base, neox);
    }
}

// ── Batched RoPE (one launch for all N prefill tokens) ───────────────────────
// genie prefill RoPE'd per token (N launches/layer = ~44k launches at 1261 tok,
// ~8% of prefill in pure launch overhead). blockIdx.y = token; per-token offsets
// q + i*q_stride etc., position = start_pos + i. Same math/layout as the per-token
// kernels, so results are identical.
__global__ void rope_batched_kernel(
    half* __restrict__ q, half* __restrict__ k,
    int n_heads, int n_kv_heads, int head_dim,
    int start_pos, int N, int q_stride, int k_stride, float theta_base, bool neox) {
    const int i = blockIdx.y; if (i >= N) return;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int total_pairs = (n_heads + n_kv_heads) * half_dim;
    if (tid >= total_pairs) return;
    const int q_pairs = n_heads * half_dim;
    const int position = start_pos + i;
    const bool is_q = tid < q_pairs;
    int head, pair_idx; half* ptr;
    if (is_q) { head = tid / half_dim; pair_idx = tid % half_dim; ptr = q + (int64_t)i*q_stride + head*head_dim; }
    else      { int kt = tid - q_pairs; head = kt / half_dim; pair_idx = kt % half_dim; ptr = k + (int64_t)i*k_stride + head*head_dim; }
    const float freq = 1.0f / powf(theta_base, (2.0f*pair_idx)/head_dim);
    const float angle = position*freq, c = cosf(angle), s = sinf(angle);
    const int i0 = neox ? pair_idx : pair_idx*2, i1 = neox ? pair_idx+half_dim : pair_idx*2+1;
    const float v0 = __half2float(ptr[i0]), v1 = __half2float(ptr[i1]);
    ptr[i0] = __float2half(v0*c - v1*s); ptr[i1] = __float2half(v0*s + v1*c);
}

__global__ void rope_store_kv_fp16_batched_kernel(
    half* __restrict__ q, half* __restrict__ k, const half* __restrict__ v,
    half* __restrict__ k_cache_base, half* __restrict__ v_cache_base, int kv_slot,
    int n_heads, int n_kv_heads, int head_dim, int start_pos, int N,
    int q_stride, int kv_stride, float theta_base, bool neox) {
    const int i = blockIdx.y; if (i >= N) return;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int q_pairs = n_heads * half_dim;
    const int total_pairs = (n_heads + n_kv_heads) * half_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int position = start_pos + i;
    half* qi = q + (int64_t)i*q_stride;
    half* ki = k + (int64_t)i*kv_stride;
    const half* vi = v + (int64_t)i*kv_stride;
    half* kdst = k_cache_base + (int64_t)i*kv_slot;
    half* vdst = v_cache_base + (int64_t)i*kv_slot;
    if (tid < total_pairs) {
        const bool is_q = tid < q_pairs;
        int head, pair_idx; half* ptr; half* dst = nullptr;
        if (is_q) { head = tid / half_dim; pair_idx = tid % half_dim; ptr = qi + head*head_dim; }
        else      { int kt = tid - q_pairs; head = kt / half_dim; pair_idx = kt % half_dim; ptr = ki + head*head_dim; dst = kdst + head*head_dim; }
        const float freq = 1.0f / powf(theta_base, (2.0f*pair_idx)/head_dim);
        const float angle = position*freq, c = cosf(angle), s = sinf(angle);
        const int i0 = neox ? pair_idx : pair_idx*2, i1 = neox ? pair_idx+half_dim : pair_idx*2+1;
        const float v0 = __half2float(ptr[i0]), v1 = __half2float(ptr[i1]);
        const half r0 = __float2half(v0*c - v1*s), r1 = __float2half(v0*s + v1*c);
        ptr[i0] = r0; ptr[i1] = r1;
        if (!is_q) { dst[i0] = r0; dst[i1] = r1; }
    }
    if (tid < kv_dim) vdst[tid] = vi[tid];
}

void rope_inplace_batched(half* q, half* k, int n_heads, int n_kv_heads, int head_dim,
                          int start_pos, int N, int q_stride, int k_stride,
                          float theta_base, bool neox, cudaStream_t stream) {
    const int h2    = head_dim / 2;
    const int total = (n_heads + n_kv_heads) * h2;
    const int block = BLOCK_SIZE;
    const dim3 grid((total + block - 1) / block, N);
    const RopeTableEntry* tbl = rope_table_enabled()
        ? find_rope_tbl(theta_base, h2) : nullptr;
    if (tbl) {
        rope_batched_kernel_tbl<<<grid, block, 0, stream>>>(
            q, k, n_heads, n_kv_heads, head_dim,
            tbl->d_cos, tbl->d_sin, tbl->half_dim,
            start_pos, N, q_stride, k_stride, neox);
    } else {
        rope_batched_kernel<<<grid, block, 0, stream>>>(
            q, k, n_heads, n_kv_heads, head_dim, start_pos, N, q_stride, k_stride, theta_base, neox);
    }
}

void rope_inplace_store_kv_fp16_batched(
    half* q, half* k, const half* v, half* k_cache_base, half* v_cache_base, int kv_slot,
    int n_heads, int n_kv_heads, int head_dim, int start_pos, int N,
    int q_stride, int kv_stride, float theta_base, bool neox, cudaStream_t stream) {
    const int h2     = head_dim / 2;
    const int t_rope = (n_heads + n_kv_heads) * h2;
    const int kv_dim = n_kv_heads * head_dim;
    const int total  = t_rope > kv_dim ? t_rope : kv_dim;
    const int block  = BLOCK_SIZE;
    const dim3 grid((total + block - 1) / block, N);
    const RopeTableEntry* tbl = rope_table_enabled()
        ? find_rope_tbl(theta_base, h2) : nullptr;
    if (tbl) {
        rope_store_kv_fp16_batched_kernel_tbl<<<grid, block, 0, stream>>>(
            q, k, v, k_cache_base, v_cache_base, kv_slot,
            n_heads, n_kv_heads, head_dim,
            tbl->d_cos, tbl->d_sin, tbl->half_dim,
            start_pos, N, q_stride, kv_stride, neox);
    } else {
        rope_store_kv_fp16_batched_kernel<<<grid, block, 0, stream>>>(
            q, k, v, k_cache_base, v_cache_base, kv_slot,
            n_heads, n_kv_heads, head_dim, start_pos, N, q_stride, kv_stride, theta_base, neox);
    }
}

// CUDA-graph-friendly variant: position is read from a device int* at
// kernel-execution time, so a single captured graph can be replayed for
// every decode step. The K/V cache destinations are likewise computed
// inside the kernel as `cache_base + pos * kv_stride`, so only the
// per-layer base pointers (which are fixed across the model's lifetime)
// need to be captured.
__global__ void rope_store_kv_fp16_kernel_dyn(
    half*       __restrict__ q,
    half*       __restrict__ k,
    const half* __restrict__ v,
    half*       __restrict__ k_cache_layer_base,
    half*       __restrict__ v_cache_layer_base,
    int n_heads, int n_kv_heads, int head_dim, int kv_stride,
    const int*  __restrict__ d_pos,
    float theta_base, bool neox)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int q_pairs = n_heads * half_dim;
    const int total_pairs = (n_heads + n_kv_heads) * half_dim;
    const int kv_dim = n_kv_heads * head_dim;

    const int position = *d_pos;
    half* const k_cache_dst = k_cache_layer_base + (int64_t)position * kv_stride;
    half* const v_cache_dst = v_cache_layer_base + (int64_t)position * kv_stride;

    if (tid < total_pairs) {
        const bool is_q = tid < q_pairs;
        int head, pair_idx;
        half* ptr;
        half* dst = nullptr;

        if (is_q) {
            head = tid / half_dim;
            pair_idx = tid % half_dim;
            ptr = q + head * head_dim;
        } else {
            const int k_tid = tid - q_pairs;
            head = k_tid / half_dim;
            pair_idx = k_tid % half_dim;
            ptr = k + head * head_dim;
            dst = k_cache_dst + head * head_dim;
        }

        const float freq = 1.0f / powf(theta_base, (2.0f * pair_idx) / head_dim);
        const float angle = position * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);

        const int idx0 = neox ? pair_idx : pair_idx * 2;
        const int idx1 = neox ? pair_idx + half_dim : pair_idx * 2 + 1;
        const float v0 = __half2float(ptr[idx0]);
        const float v1 = __half2float(ptr[idx1]);
        const half r0 = __float2half(v0 * cos_val - v1 * sin_val);
        const half r1 = __float2half(v0 * sin_val + v1 * cos_val);

        ptr[idx0] = r0;
        ptr[idx1] = r1;
        if (!is_q) {
            dst[idx0] = r0;
            dst[idx1] = r1;
        }
    }

    if (tid < kv_dim) {
        v_cache_dst[tid] = v[tid];
    }
}

void rope_inplace_store_kv_fp16_dyn(
    half* q, half* k, const half* v,
    half* k_cache_layer_base, half* v_cache_layer_base,
    int n_heads, int n_kv_heads, int head_dim, int kv_stride,
    const int* d_pos, float theta_base, bool neox, cudaStream_t stream)
{
    const int h2     = head_dim / 2;
    const int t_rope = (n_heads + n_kv_heads) * h2;
    const int kv_dim = n_kv_heads * head_dim;
    const int total  = t_rope > kv_dim ? t_rope : kv_dim;
    const int block  = BLOCK_SIZE;
    const int grid   = (total + block - 1) / block;
    const RopeTableEntry* tbl = rope_table_enabled()
        ? find_rope_tbl(theta_base, h2) : nullptr;
    if (tbl) {
        rope_store_kv_fp16_kernel_dyn_tbl<<<grid, block, 0, stream>>>(
            q, k, v, k_cache_layer_base, v_cache_layer_base,
            n_heads, n_kv_heads, head_dim, kv_stride,
            d_pos, tbl->d_cos, tbl->d_sin, tbl->half_dim, neox);
    } else {
        rope_store_kv_fp16_kernel_dyn<<<grid, block, 0, stream>>>(
            q, k, v, k_cache_layer_base, v_cache_layer_base,
            n_heads, n_kv_heads, head_dim, kv_stride,
            d_pos, theta_base, neox);
    }
}

}  // namespace jllm
