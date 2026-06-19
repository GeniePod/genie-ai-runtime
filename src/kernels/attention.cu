// attention.cu — Flash Attention decode (single query) for Orin SM 8.7
//
// BUG #6 FIX: proper per-dimension accumulator using shared memory output buffer
// instead of broken acc[d % 4]. Each thread handles head_dim / blockDim.x dimensions.
//
// One block per query head. Tiles KV in chunks of 64.
// Online softmax: never materializes full seq×seq attention matrix.
// Supports GQA and INT8 KV cache.

#include "jllm_kernels.h"
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace jllm {

static constexpr int ATTN_TILE_KV = 64;
static constexpr int ATTN_BLOCK   = 128;

static bool fast_attention_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_FAST_ATTN");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

__device__ __forceinline__ float warp_max_f(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum_f(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// Block reduce over up to 4 warps (ATTN_BLOCK=128). red[] is shared scratch.
__device__ __forceinline__ float block_max_f(float v, float* red, int warp_id, int lane) {
    v = warp_max_f(v);
    if (lane == 0) red[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        float x = (lane < 4) ? red[lane] : -FLT_MAX;
        x = warp_max_f(x);
        if (lane == 0) red[0] = x;
    }
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_sum_f(float v, float* red, int warp_id, int lane) {
    v = warp_sum_f(v);
    if (lane == 0) red[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        float x = (lane < 4) ? red[lane] : 0.0f;
        x = warp_sum_f(x);
        if (lane == 0) red[0] = x;
    }
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}

// ===================================================================
// F32 query-tiled flash-attention prefill (opt-in via JLLM_FLASH_TC).
// Roofline on Orin: attention is MEMORY-BANDWIDTH-bound (AI 16-32 FLOP/byte
// << ridge ~200), so the win is K/V-byte reduction, NOT tensor-core compute.
// 16 queries/block (one warp each) share each cp.async-loaded K/V tile ->
// ~16x less K/V DRAM than the scalar 1-query/block kernel, and the speedup
// GROWS with context (scalar prefill 63 tok/s @8k -> 101, ~flat). Math is
// F32 (warp all-reduce dot + online softmax), numerically == the scalar
// kernel, avoiding the f16 tensor-core precision loss that Gemma's large V
// outliers amplify. FP16 KV only (Gemma forces FP16 KV); int8 -> scalar path.
// ===================================================================
static bool flash_tc_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_FLASH_TC");
        return v && strcmp(v, "0") != 0;   // opt-in: default OFF
    }();
    return enabled;
}

static constexpr int TC_KT = 16;   // keys per K/V tile
// Finite-math-safe sentinels: the build uses --use_fast_math (finite-math-only,
// no Inf/NaN), so masking must NOT use -INFINITY (a fully-masked row would do
// exp(-INF - -INF) = exp(NaN), undefined under fast-math). Use a large negative
// value + a guard so the kernel never produces Inf/NaN.
#define TC_NEG   (-1e30f)
#define TC_NEG_T (-1e29f)   // threshold: anything <= this is "masked"

// async-load one K/V tile (abs positions kt..kt+15) into shared f16 buffers.
// cache stride = kv_dim (= n_kv_heads*cache_head_dim); only HEAD_DIM cols read
// from the cache_head_dim-wide slot. Out-of-range rows zero-filled.
template<int D>
__device__ __forceinline__ void tc_load_tile(half* Kb, half* Vb,
        const half* k_cache, const half* v_cache, int kt, int block_seq,
        int kv_dim, int kv_hoff, int lane, int bdim) {
    const int nchunk = (TC_KT*D)/8;
    for (int chunk = lane; chunk < nchunk; chunk += bdim) {
        const int hidx = chunk*8, r = hidx/D, c = hidx%D, kp = kt + r;
        if (kp < block_seq) {
            __pipeline_memcpy_async(&Kb[hidx], &k_cache[(int64_t)kp*kv_dim + kv_hoff + c], 16);
            __pipeline_memcpy_async(&Vb[hidx], &v_cache[(int64_t)kp*kv_dim + kv_hoff + c], 16);
        } else {
            #pragma unroll
            for (int z = 0; z < 8; z++) { Kb[hidx+z]=__float2half(0.f); Vb[hidx+z]=__float2half(0.f); }
        }
    }
}

// ===================================================================
// F32 query-tiled flash-attention prefill (the roofline-correct kernel).
// Attention is MEMORY-BOUND on Orin, so the win is K/V-byte reduction, NOT
// tensor-core compute. MQ=16 queries/block (one warp per query) share each
// cp.async-loaded K/V tile -> 16x less K/V DRAM than the scalar MQ=1 kernel.
// All math in F32 (warp all-reduce dot + online softmax) -> bit-for-bit the
// scalar kernel's precision, avoiding the f16-P MMA error that Gemma's large
// V outliers amplify. No MMA, no f16-P, no 2-warp split. FP16 KV only.
// ===================================================================
template<int HEAD_DIM, int MQ>
__global__ void flash_attn_prefill_f32t_kernel(
        half* __restrict__ output, const half* __restrict__ q,
        const half* __restrict__ k_cache, const half* __restrict__ v_cache,
        int n_heads, int n_kv_heads, int cache_head_dim,
        int N, int start_pos, float scale, int window) {
    const int head = blockIdx.x, qt = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q_idx = qt * MQ + warp;          // query index within the chunk
    const int q_abs = start_pos + q_idx;       // absolute position (for masking)
    const int q_dim = n_heads * HEAD_DIM;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_dim  = n_kv_heads * cache_head_dim;
    const int kv_hoff = kv_head * cache_head_dim;
    constexpr int CPL = HEAD_DIM / 32;         // head-dim components per lane

    extern __shared__ half f32t_smem[];
    half* Kbuf = f32t_smem;                    // 2 * TC_KT * HEAD_DIM (double-buffered)
    half* Vbuf = Kbuf + 2*TC_KT*HEAD_DIM;
    half* Kb[2] = { Kbuf, Kbuf + TC_KT*HEAD_DIM };
    half* Vb[2] = { Vbuf, Vbuf + TC_KT*HEAD_DIM };

    const bool valid_q = (q_idx < N);
    float qreg[CPL], oreg[CPL];
    #pragma unroll
    for (int c = 0; c < CPL; c++) {
        const int d = lane + c*32;
        qreg[c] = valid_q ? __half2float(q[(int64_t)q_idx*q_dim + (int64_t)head*HEAD_DIM + d]) : 0.f;
        oreg[c] = 0.f;
    }
    float m = -1e30f, l = 0.f;

    const int block_seq = start_pos + min(qt*MQ + MQ - 1, N - 1) + 1;   // uniform per block
    int kt_start = 0;
    if (window > 0) { kt_start = start_pos + qt*MQ + 1 - window; if (kt_start < 0) kt_start = 0; kt_start = (kt_start/TC_KT)*TC_KT; }

    int buf = 0;
    tc_load_tile<HEAD_DIM>(Kb[0], Vb[0], k_cache, v_cache, kt_start, block_seq, kv_dim, kv_hoff, threadIdx.x, blockDim.x);
    __pipeline_commit();

    for (int kt = kt_start; kt < block_seq; kt += TC_KT) {
        const int ktn = kt + TC_KT;
        if (ktn < block_seq) {
            tc_load_tile<HEAD_DIM>(Kb[buf^1], Vb[buf^1], k_cache, v_cache, ktn, block_seq, kv_dim, kv_hoff, threadIdx.x, blockDim.x);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        const half* Ksh = Kb[buf];
        const half* Vsh = Vb[buf];

        #pragma unroll 1
        for (int j = 0; j < TC_KT; j++) {
            const int kp = kt + j;
            const bool masked = !valid_q || (kp > q_abs) || (window > 0 && kp < q_abs + 1 - window);
            float dot = 0.f;
            #pragma unroll
            for (int c = 0; c < CPL; c++) { const int d = lane + c*32; dot += qreg[c] * __half2float(Ksh[j*HEAD_DIM + d]); }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);   // all-reduce -> every lane has full dot
            const float s = masked ? -1e30f : dot * scale;
            const float nm = fmaxf(m, s);
            const float corr = (m <= -1e29f) ? 0.f : __expf(m - nm);
            const float p    = (s <= -1e29f) ? 0.f : __expf(s - nm);
            m = nm; l = l*corr + p;
            #pragma unroll
            for (int c = 0; c < CPL; c++) { const int d = lane + c*32; oreg[c] = oreg[c]*corr + p * __half2float(Vsh[j*HEAD_DIM + d]); }
        }
        __syncthreads();
        buf ^= 1;
    }

    if (valid_q) {
        const float inv = (l > 0.f) ? 1.f/l : 0.f;
        #pragma unroll
        for (int c = 0; c < CPL; c++) { const int d = lane + c*32; output[(int64_t)q_idx*q_dim + (int64_t)head*HEAD_DIM + d] = __float2half(oreg[c]*inv); }
    }
}

// launch wrapper: picks HEAD_DIM=256/512 template + opt-in shared. Returns
// false on any unsupported shape (caller falls back to the scalar path).
static bool launch_flash_attn_prefill_tc(
        half* output, const half* q, const void* k_cache, const void* v_cache,
        int n_heads, int n_kv_heads, int head_dim, int cache_head_dim,
        int N, int start_pos, float scale, int window, cudaStream_t stream) {
    if (head_dim != 256 && head_dim != 512) return false;
    constexpr int MQ = 16;                                  // queries per block (1 warp each)
    size_t smbytes = (size_t)(4*TC_KT*head_dim)*sizeof(half);  // double-buffered K+V tiles
    const dim3 grid(n_heads, (N + MQ - 1)/MQ, 1);
    const int block = MQ*32;
    const half* kh = (const half*)k_cache;
    const half* vh = (const half*)v_cache;
    cudaError_t sa;
    if (head_dim == 256) {
        sa = cudaFuncSetAttribute(flash_attn_prefill_f32t_kernel<256,MQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smbytes);
        if (sa != cudaSuccess) { fprintf(stderr, "[attention] F32T d256 smem %zuB setattr FAIL: %s\n", smbytes, cudaGetErrorString(sa)); return false; }
        flash_attn_prefill_f32t_kernel<256,MQ><<<grid, block, smbytes, stream>>>(
            output, q, kh, vh, n_heads, n_kv_heads, cache_head_dim, N, start_pos, scale, window);
    } else {
        sa = cudaFuncSetAttribute(flash_attn_prefill_f32t_kernel<512,MQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smbytes);
        if (sa != cudaSuccess) { fprintf(stderr, "[attention] F32T d512 smem %zuB setattr FAIL: %s\n", smbytes, cudaGetErrorString(sa)); return false; }
        flash_attn_prefill_f32t_kernel<512,MQ><<<grid, block, smbytes, stream>>>(
            output, q, kh, vh, n_heads, n_kv_heads, cache_head_dim, N, start_pos, scale, window);
    }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { fprintf(stderr, "[attention] F32T d%d launch FAIL (smem %zuB): %s\n", head_dim, smbytes, cudaGetErrorString(le)); return false; }
    return true;
}

// Each thread handles ceil(head_dim / blockDim.x) output dimensions.
// Accumulators stored in shared memory (visible to all threads in block).

__global__ void flash_attention_decode_kernel(
    half*        __restrict__ output,
    const half*  __restrict__ q,
    const void*  __restrict__ k_cache,
    const void*  __restrict__ v_cache,
    int n_heads, int n_kv_heads, int head_dim, int cache_head_dim, int seq_len,
    float scale, bool kv_int8,
    const float* __restrict__ k_scales,   // Path I3: [seq_len, n_kv_heads]
    const float* __restrict__ v_scales,
    int window)   // Gemma sliding layers: attend only to last `window` keys (0 = full)
{
    const int head    = blockIdx.x;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    // Per-position stride in the KV cache uses the cache slot head dim, which
    // for Gemma sliding layers (active head_dim 256) is the larger global slot
    // (512). For all other models cache_head_dim == head_dim (no-op).
    const int kv_dim  = n_kv_heads * cache_head_dim;
    const int kv_hoff = kv_head * cache_head_dim;

    // Shared memory layout:
    //   s_scores[ATTN_TILE_KV]  — attention scores for current KV tile
    //   s_out[head_dim]         — output accumulator
    //   s_q[head_dim]           — query cached once, reused across every KV tile
    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_out    = smem + ATTN_TILE_KV;
    float* s_q      = smem + ATTN_TILE_KV + head_dim;

    __shared__ float s_running_max;
    __shared__ float s_running_sum;
    __shared__ float s_correction;
    __shared__ float s_red[4];

    // Load Q once into shared memory; initialize output accumulator.
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_out[d] = 0.0f;
        s_q[d]   = __half2float(q[head * head_dim + d]);
    }
    if (tid == 0) {
        s_running_max = -FLT_MAX;
        s_running_sum = 0.0f;
        s_correction  = 1.0f;
    }
    __syncthreads();

    for (int kv_start = 0; kv_start < seq_len; kv_start += ATTN_TILE_KV) {
        int tile_len = min(ATTN_TILE_KV, seq_len - kv_start);

        // ── Step 1: Q × K^T — one warp per KV slot, lane-strided inner loop
        //   then warp-shuffle reduce. Eliminates the serial head_dim loop and
        //   the repeated Q global reads that the old single-thread path had.
        for (int t = warp_id; t < tile_len; t += (blockDim.x >> 5)) {
            int kv_pos = kv_start + t;
            // Sliding-window mask (Gemma local layers only).
            if (window > 0 && kv_pos < seq_len - window) {
                if (lane == 0) s_scores[t] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            if (kv_int8) {
                const int8_t* krow = (const int8_t*)k_cache
                    + (int64_t)kv_pos * kv_dim + kv_hoff;
                float ks = k_scales ? k_scales[(int64_t)kv_pos * n_kv_heads + kv_head] : 1.0f;
                for (int d = lane; d < head_dim; d += 32)
                    dot += s_q[d] * (krow[d] * ks);
            } else {
                const half* krow = (const half*)k_cache
                    + (int64_t)kv_pos * kv_dim + kv_hoff;
                for (int d = lane; d < head_dim; d += 32)
                    dot += s_q[d] * __half2float(krow[d]);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) s_scores[t] = dot * scale;
        }
        __syncthreads();

        // ── Step 2: Block-parallel online softmax ──────────────────
        float lmax = -FLT_MAX;
        for (int t = tid; t < tile_len; t += blockDim.x)
            lmax = fmaxf(lmax, s_scores[t]);
        float tile_max = block_max_f(lmax, s_red, warp_id, lane);
        if (tid == 0) {
            const float old_max = s_running_max;
            s_running_max = fmaxf(s_running_max, tile_max);
            s_correction  = expf(old_max - s_running_max);
            s_running_sum *= s_correction;
        }
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x)
            s_out[d] *= s_correction;

        float lsum = 0.0f;
        for (int t = tid; t < tile_len; t += blockDim.x) {
            float p = expf(s_scores[t] - s_running_max);
            s_scores[t] = p;
            lsum += p;
        }
        float tile_sum = block_sum_f(lsum, s_red, warp_id, lane);
        if (tid == 0) s_running_sum += tile_sum;
        __syncthreads();

        // ── Step 3: Accumulate P × V ────────────────────────────────
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                int kv_pos = kv_start + t;
                float v_val;
                if (kv_int8) {
                    const int8_t* vi = (const int8_t*)v_cache;
                    float vs = v_scales ? v_scales[(int64_t)kv_pos * n_kv_heads + kv_head] : 1.0f;
                    v_val = vi[(int64_t)kv_pos * kv_dim + kv_hoff + d] * vs;
                } else {
                    const half* vf = (const half*)v_cache;
                    v_val = __half2float(vf[(int64_t)kv_pos * kv_dim + kv_hoff + d]);
                }
                val += s_scores[t] * v_val;
            }
            s_out[d] += val;
        }
        __syncthreads();
    }

    float inv_sum = (s_running_sum > 0.0f) ? 1.0f / s_running_sum : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x)
        output[head * head_dim + d] = __float2half(s_out[d] * inv_sum);
}

// Prefill counterpart of flash_attention_decode_kernel. Each block
// computes attention for one (query_head, query_token) pair. Grid is
// (n_heads, N). Causal mask collapses to seq_len = start_pos + token + 1
// per query, so the existing online-softmax loop only needs to widen
// its q/output addressing by one dimension.
__global__ void flash_attention_prefill_batched_kernel(
    half*        __restrict__ output,        // [N × n_heads × head_dim]
    const half*  __restrict__ q,              // [N × n_heads × head_dim]
    const void*  __restrict__ k_cache,
    const void*  __restrict__ v_cache,
    int n_heads, int n_kv_heads, int head_dim, int cache_head_dim,
    int start_pos,
    float scale, bool kv_int8,
    const float* __restrict__ k_scales,   // Path I3
    const float* __restrict__ v_scales,
    int window)   // sliding-window size (0 = full attention)
{
    const int head    = blockIdx.x;
    const int token   = blockIdx.y;
    const int seq_len = start_pos + token + 1;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int tid     = threadIdx.x;
    const int kv_dim  = n_kv_heads * cache_head_dim;  // cache slot stride
    const int kv_hoff = kv_head * cache_head_dim;
    const int q_dim   = n_heads    * head_dim;

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_out    = smem + ATTN_TILE_KV;
    float* s_q      = smem + ATTN_TILE_KV + head_dim;   // cached query row

    __shared__ float s_running_max;
    __shared__ float s_running_sum;
    __shared__ float s_correction;
    __shared__ float s_red[4];

    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const half* q_local   = q      + (int64_t)token * q_dim + (int64_t)head * head_dim;
    half*       out_local = output + (int64_t)token * q_dim + (int64_t)head * head_dim;

    // Cache the query row once (reused across every KV position).
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_out[d] = 0.0f;
        s_q[d]   = __half2float(q_local[d]);
    }
    if (tid == 0) {
        s_running_max = -FLT_MAX;
        s_running_sum = 0.0f;
        s_correction  = 1.0f;
    }
    __syncthreads();

    for (int kv_start = 0; kv_start < seq_len; kv_start += ATTN_TILE_KV) {
        int tile_len = min(ATTN_TILE_KV, seq_len - kv_start);

        // Q × K^T for tile: one warp per KV position, 32 lanes split + reduce
        // the head_dim dot (vs the old one-thread-per-position serial dot).
        for (int t = warp_id; t < tile_len; t += (blockDim.x >> 5)) {
            int kv_pos = kv_start + t;
            if (window > 0 && kv_pos < seq_len - window) {
                if (lane == 0) s_scores[t] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            if (kv_int8) {
                const int8_t* krow = (const int8_t*)k_cache
                    + (int64_t)kv_pos * kv_dim + kv_hoff;
                float ks = k_scales ? k_scales[(int64_t)kv_pos * n_kv_heads + kv_head] : 1.0f;
                for (int d = lane; d < head_dim; d += 32)
                    dot += s_q[d] * (krow[d] * ks);
            } else {
                const half* krow = (const half*)k_cache
                    + (int64_t)kv_pos * kv_dim + kv_hoff;
                for (int d = lane; d < head_dim; d += 32)
                    dot += s_q[d] * __half2float(krow[d]);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) s_scores[t] = dot * scale;
        }
        __syncthreads();

        // Block-parallel online-softmax max (replaces the old tid==0 loop).
        float lmax = -FLT_MAX;
        for (int t = tid; t < tile_len; t += blockDim.x)
            lmax = fmaxf(lmax, s_scores[t]);
        float tile_max = block_max_f(lmax, s_red, warp_id, lane);
        if (tid == 0) {
            const float old_max = s_running_max;
            s_running_max = fmaxf(s_running_max, tile_max);
            s_correction  = expf(old_max - s_running_max);
            s_running_sum *= s_correction;
        }
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x)
            s_out[d] *= s_correction;

        // Block-parallel exp + sum.
        float lsum = 0.0f;
        for (int t = tid; t < tile_len; t += blockDim.x) {
            float p = expf(s_scores[t] - s_running_max);
            s_scores[t] = p;
            lsum += p;
        }
        float tile_sum = block_sum_f(lsum, s_red, warp_id, lane);
        if (tid == 0) s_running_sum += tile_sum;
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                int kv_pos = kv_start + t;
                float v_val;
                if (kv_int8) {
                    const int8_t* vi = (const int8_t*)v_cache;
                    float vs = v_scales ? v_scales[(int64_t)kv_pos * n_kv_heads + kv_head] : 1.0f;
                    v_val = vi[(int64_t)kv_pos * kv_dim + kv_hoff + d] * vs;
                } else {
                    const half* vf = (const half*)v_cache;
                    v_val = __half2float(
                        vf[(int64_t)kv_pos * kv_dim + kv_hoff + d]);
                }
                val += s_scores[t] * v_val;
            }
            s_out[d] += val;
        }
        __syncthreads();
    }

    float inv_sum = (s_running_sum > 0.0f) ? 1.0f / s_running_sum : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_local[d] = __float2half(s_out[d] * inv_sum);
    }
}

// ── Tensor-core unfused prefill attention via cuBLAS (JLLM_ATTN_CUBLAS) ───────
// QK^T (fp16 TC gemm) → masked f32 softmax → P·V (fp16 TC gemm). ~2-10x the F32
// flash kernel (genie's attn was 0.22% tensor-core util / ALU-bound; llama uses
// TC). Numerically safe: fp16 inputs + f32 accumulate, validated rel-RMS 3e-4
// even on Gemma's 10-20x V-outliers (the prior TC-attn failure was f16 *accumulate*,
// not f16 inputs). FP16 KV + MQA (n_kv_heads==1) only; else returns false to fall
// back to the F32 path.
static bool attn_cublas_enabled() {
    static int e = -1;
    if (e < 0) { const char* v = getenv("JLLM_ATTN_CUBLAS"); e = (v && v[0] == '0') ? 0 : 1; }
    return e;   // default on; JLLM_ATTN_CUBLAS=0 reverts to the F32 flash kernel
}
static cublasHandle_t g_attn_cublas = nullptr;
static float* g_attn_scores = nullptr; static size_t g_attn_scores_cap = 0;
static half*  g_attn_p      = nullptr; static size_t g_attn_p_cap = 0;

// per (head, query i): scores S already scaled (gemm alpha = scale). Mask causal
// (key j ≤ start_pos+i) + sliding window (j ≥ start_pos+i+1-window), then softmax.
// S[(head*N + i)*seq + j] → P fp16 (normalized).
__global__ void attn_mask_softmax_kernel(const float* __restrict__ S, half* __restrict__ P,
                                         int N, int seq, int start_pos, int window) {
    const int i = blockIdx.x, head = blockIdx.y;
    const int64_t row = ((int64_t)head * N + i) * seq;
    const int qpos = start_pos + i, hi = qpos;
    int lo = (window > 0) ? (qpos + 1 - window) : 0; if (lo < 0) lo = 0;
    extern __shared__ float sh[];
    float mx = -1e30f;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) {
        float v = (j >= lo && j <= hi) ? S[row + j] : -1e30f; sh[j] = v; mx = fmaxf(mx, v);
    }
    __syncthreads();
    __shared__ float r[32]; int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    if (l == 0) r[w] = mx; __syncthreads();
    if (threadIdx.x == 0) { float m = -1e30f; for (int k = 0; k < (blockDim.x >> 5); k++) m = fmaxf(m, r[k]); r[0] = m; }
    __syncthreads(); mx = r[0];
    float sum = 0;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) { float p = sh[j] > -1e29f ? __expf(sh[j] - mx) : 0.f; sh[j] = p; sum += p; }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    __shared__ float rs[32]; if (l == 0) rs[w] = sum; __syncthreads();
    if (threadIdx.x == 0) { float s = 0; for (int k = 0; k < (blockDim.x >> 5); k++) s += rs[k]; rs[0] = s; }
    __syncthreads();
    float inv = rs[0] > 0 ? 1.f / rs[0] : 0.f;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) P[row + j] = __float2half(sh[j] * inv);
}

static bool launch_attn_cublas_prefill(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim, int cache_head_dim,
    int N, int start_pos, float scale, int window, cudaStream_t stream) {
    if (n_kv_heads != 1) return false;             // MQA only for now (Gemma E2B)
    const int seq    = start_pos + N;
    const int q_dim  = n_heads * head_dim;
    const int kv_dim = n_kv_heads * cache_head_dim;
    if (!g_attn_cublas) { if (cublasCreate(&g_attn_cublas) != CUBLAS_STATUS_SUCCESS) return false; }
    cublasSetStream(g_attn_cublas, stream);
    const size_t sc = (size_t)n_heads * N * seq;
    if (sc > g_attn_scores_cap) { if (g_attn_scores) cudaFree(g_attn_scores);
        if (cudaMalloc(&g_attn_scores, sc * sizeof(float)) != cudaSuccess) { g_attn_scores_cap = 0; return false; } g_attn_scores_cap = sc; }
    if (sc > g_attn_p_cap) { if (g_attn_p) cudaFree(g_attn_p);
        if (cudaMalloc(&g_attn_p, sc * sizeof(half)) != cudaSuccess) { g_attn_p_cap = 0; return false; } g_attn_p_cap = sc; }
    float alpha = scale, zero = 0.f;
    // S[head][N×seq] row-major = scale·Q·K^T. Col-major C^T(seq×N)=K·Q^T → OP_T(K), OP_N(Q).
    // K shared across heads (MQA → strideK 0); Q per head (stride head_dim within q_dim).
    if (cublasGemmStridedBatchedEx(g_attn_cublas, CUBLAS_OP_T, CUBLAS_OP_N, seq, N, head_dim,
            &alpha, k_cache, CUDA_R_16F, kv_dim, 0,
                    q,       CUDA_R_16F, q_dim,  head_dim,
            &zero,  g_attn_scores, CUDA_R_32F, seq, (long long)N * seq,
            n_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
    attn_mask_softmax_kernel<<<dim3(N, n_heads), 256, (size_t)seq * sizeof(float), stream>>>(
        g_attn_scores, g_attn_p, N, seq, start_pos, window);
    // O[head][N×head_dim] row-major = P·V. Col-major C^T(head_dim×N) → OP_N(V), OP_N(P).
    float one = 1.f;
    if (cublasGemmStridedBatchedEx(g_attn_cublas, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, N, seq,
            &one, v_cache, CUDA_R_16F, kv_dim, 0,
                  g_attn_p, CUDA_R_16F, seq, (long long)N * seq,
            &zero, output, CUDA_R_16F, q_dim, head_dim,
            n_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return false;
    return true;
}

void flash_attention_prefill_batched(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim, int cache_head_dim,
    int N, int start_pos,
    float scale, bool kv_int8,
    const float* k_scales, const float* v_scales, int window, cudaStream_t stream)
{
    if (N <= 0) return;
    // Tensor-core cuBLAS prefill attention (FP16 KV + MQA); falls back if unsupported.
    if (attn_cublas_enabled() && !kv_int8 &&
        launch_attn_cublas_prefill(output, q, k_cache, v_cache, n_heads, n_kv_heads,
            head_dim, cache_head_dim, N, start_pos, scale, window, stream)) {
        return;
    }
    // Path I3 (#62): removed the `kv_int8 → N-sequential-decode` fallback
    // that lived here. The batched kernel now handles INT8 via per-position
    // k_scales / v_scales lookups; no reason to walk per-token anymore.
    if (!fast_attention_enabled()) {
        const int q_dim = n_heads * head_dim;
        for (int t = 0; t < N; t++) {
            flash_attention_decode(
                output + (int64_t)t * q_dim,
                q      + (int64_t)t * q_dim,
                k_cache, v_cache,
                n_heads, n_kv_heads, head_dim, cache_head_dim,
                start_pos + t + 1, scale, kv_int8, k_scales, v_scales, window, stream);
        }
        return;
    }

    // Opt-in F32 query-tiled path (FP16 KV only — Gemma forces FP16 KV).
    // Memory-bound on Orin, so the win is K/V-byte reduction via 16-query
    // tiles (16x less K/V DRAM than the scalar MQ=1 kernel); f32 math.
    if (flash_tc_enabled() && !kv_int8 &&
        launch_flash_attn_prefill_tc(output, q, k_cache, v_cache, n_heads,
            n_kv_heads, head_dim, cache_head_dim, N, start_pos, scale, window, stream)) {
        static bool logged_tc = false;
        if (!logged_tc) {
            fprintf(stderr, "[attention] Using F32 query-tiled prefill attention path (JLLM_FLASH_TC)\n");
            logged_tc = true;
        }
        return;
    }

    static bool logged = false;
    if (!logged) {
        fprintf(stderr, "[attention] Using CUDA chunked-prefill attention path\n");
        logged = true;
    }

    const dim3 grid(n_heads, N, 1);
    const int smem = (ATTN_TILE_KV + 2 * head_dim) * (int)sizeof(float);
    flash_attention_prefill_batched_kernel<<<grid, ATTN_BLOCK, smem, stream>>>(
        output, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, cache_head_dim,
        start_pos, scale, kv_int8, k_scales, v_scales, window);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[attention] chunked-prefill CUDA launch failed: %s; "
                "falling back to N decode launches\n", cudaGetErrorString(err));
        const int q_dim = n_heads * head_dim;
        for (int t = 0; t < N; t++) {
            flash_attention_decode(
                output + (int64_t)t * q_dim,
                q      + (int64_t)t * q_dim,
                k_cache, v_cache,
                n_heads, n_kv_heads, head_dim, cache_head_dim,
                start_pos + t + 1, scale, kv_int8, k_scales, v_scales, window, stream);
        }
    }
}

// CUDA-graph-friendly variant of flash_attention_decode_kernel: seq_len is
// read from a device int* at kernel-execution time as (*d_pos) + 1. Same
// warp-parallel Q×K^T + block-parallel softmax body as the non-graph path.
__global__ void flash_attention_decode_kernel_dyn(
    half*        __restrict__ output,
    const half*  __restrict__ q,
    const void*  __restrict__ k_cache,
    const void*  __restrict__ v_cache,
    int n_heads, int n_kv_heads, int head_dim,
    const int*   __restrict__ d_pos,
    float scale, bool kv_int8, const float* kv_scales)
{
    const int head    = blockIdx.x;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int kv_dim  = n_kv_heads * head_dim;
    const int seq_len = (*d_pos) + 1;

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_out    = smem + ATTN_TILE_KV;
    float* s_q      = smem + ATTN_TILE_KV + head_dim;

    __shared__ float s_running_max;
    __shared__ float s_running_sum;
    __shared__ float s_correction;
    __shared__ float s_red[4];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_out[d] = 0.0f;
        s_q[d]   = __half2float(q[head * head_dim + d]);
    }
    if (tid == 0) {
        s_running_max = -FLT_MAX;
        s_running_sum = 0.0f;
        s_correction  = 1.0f;
    }
    __syncthreads();

    for (int kv_start = 0; kv_start < seq_len; kv_start += ATTN_TILE_KV) {
        int tile_len = min(ATTN_TILE_KV, seq_len - kv_start);

        for (int t = warp_id; t < tile_len; t += (blockDim.x >> 5)) {
            int kv_pos = kv_start + t;
            float dot = 0.0f;
            if (kv_int8) {
                const int8_t* krow = (const int8_t*)k_cache
                    + (int64_t)kv_pos * kv_dim + kv_head * head_dim;
                float ks = kv_scales ? kv_scales[kv_head] : 1.0f;
                for (int d = lane; d < head_dim; d += 32)
                    dot += s_q[d] * (krow[d] * ks);
            } else {
                const half* krow = (const half*)k_cache
                    + (int64_t)kv_pos * kv_dim + kv_head * head_dim;
                for (int d = lane; d < head_dim; d += 32)
                    dot += s_q[d] * __half2float(krow[d]);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) s_scores[t] = dot * scale;
        }
        __syncthreads();

        float lmax = -FLT_MAX;
        for (int t = tid; t < tile_len; t += blockDim.x)
            lmax = fmaxf(lmax, s_scores[t]);
        float tile_max = block_max_f(lmax, s_red, warp_id, lane);
        if (tid == 0) {
            const float old_max = s_running_max;
            s_running_max = fmaxf(s_running_max, tile_max);
            s_correction  = expf(old_max - s_running_max);
            s_running_sum *= s_correction;
        }
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x)
            s_out[d] *= s_correction;

        float lsum = 0.0f;
        for (int t = tid; t < tile_len; t += blockDim.x) {
            float p = expf(s_scores[t] - s_running_max);
            s_scores[t] = p;
            lsum += p;
        }
        float tile_sum = block_sum_f(lsum, s_red, warp_id, lane);
        if (tid == 0) s_running_sum += tile_sum;
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                int kv_pos = kv_start + t;
                float v_val;
                if (kv_int8) {
                    const int8_t* vi = (const int8_t*)v_cache;
                    float vs = kv_scales ? kv_scales[n_kv_heads + kv_head] : 1.0f;
                    v_val = vi[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d] * vs;
                } else {
                    const half* vf = (const half*)v_cache;
                    v_val = __half2float(vf[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d]);
                }
                val += s_scores[t] * v_val;
            }
            s_out[d] += val;
        }
        __syncthreads();
    }

    float inv_sum = (s_running_sum > 0.0f) ? 1.0f / s_running_sum : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x)
        output[head * head_dim + d] = __float2half(s_out[d] * inv_sum);
}

void flash_attention_decode_dyn(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim,
    const int* d_pos,
    float scale, bool kv_int8, const float* kv_scales, cudaStream_t stream)
{
    const int smem = (ATTN_TILE_KV + 2 * head_dim) * (int)sizeof(float);
    flash_attention_decode_kernel_dyn<<<n_heads, ATTN_BLOCK, smem, stream>>>(
        output, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim,
        d_pos, scale, kv_int8, kv_scales);
}

void flash_attention_decode(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim, int cache_head_dim, int seq_len,
    float scale, bool kv_int8,
    const float* k_scales, const float* v_scales, int window, cudaStream_t stream)
{
    if (fast_attention_enabled()) {
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[attention] Using CUDA decode attention path\n");
            logged = true;
        }

        const int smem = (ATTN_TILE_KV + 2 * head_dim) * (int)sizeof(float);
        flash_attention_decode_kernel<<<n_heads, ATTN_BLOCK, smem, stream>>>(
            output, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, cache_head_dim,
            seq_len, scale, kv_int8, k_scales, v_scales, window);

        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            return;
        }
        fprintf(stderr, "[attention] CUDA attention launch failed: %s; falling back to CPU reference\n",
                cudaGetErrorString(err));
    }

    cudaStreamSynchronize(stream);

    if (kv_int8) {
        fprintf(stderr, "[attention] FATAL: reference attention does not support INT8 KV yet\n");
        cudaMemset(output, 0, (size_t)n_heads * head_dim * sizeof(half));
        return;
    }

    const int kv_dim = n_kv_heads * cache_head_dim;  // cache slot stride
    const int q_dim = n_heads * head_dim;
    const int gqa = n_heads / n_kv_heads;

    std::vector<half> h_q(q_dim);
    std::vector<half> h_k((size_t)seq_len * kv_dim);
    std::vector<half> h_v((size_t)seq_len * kv_dim);
    std::vector<half> h_out(q_dim);
    std::vector<float> scores(seq_len);

    cudaMemcpy(h_q.data(), q, (size_t)q_dim * sizeof(half), cudaMemcpyDefault);
    cudaMemcpy(h_k.data(), k_cache, (size_t)seq_len * kv_dim * sizeof(half), cudaMemcpyDefault);
    cudaMemcpy(h_v.data(), v_cache, (size_t)seq_len * kv_dim * sizeof(half), cudaMemcpyDefault);

    for (int head = 0; head < n_heads; ++head) {
        const int kv_head = head / gqa;
        const int q_base = head * head_dim;

        float max_score = -FLT_MAX;
        for (int pos = 0; pos < seq_len; ++pos) {
            if (window > 0 && pos < seq_len - window) {
                scores[pos] = -INFINITY;
                continue;
            }
            const int kv_base = pos * kv_dim + kv_head * cache_head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += __half2float(h_q[q_base + d]) * __half2float(h_k[kv_base + d]);
            }
            const float score = dot * scale;
            scores[pos] = score;
            if (score > max_score) max_score = score;
        }

        float sum = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            const float p = expf(scores[pos] - max_score);
            scores[pos] = p;
            sum += p;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int pos = 0; pos < seq_len; ++pos) {
                const int kv_base = pos * kv_dim + kv_head * cache_head_dim;
                acc += scores[pos] * inv_sum * __half2float(h_v[kv_base + d]);
            }
            h_out[q_base + d] = __float2half(acc);
        }
    }

    cudaMemcpy(output, h_out.data(), (size_t)q_dim * sizeof(half), cudaMemcpyDefault);
}

}  // namespace jllm
