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

// Tiled flash-attention prefill: one warp per query, K/V tiles reused from
// shared across the WARPS queries in the block (vs the one-query-per-block
// kernel that re-reads all K/V per query -> O(N^2) memory). Default on for
// INT8 KV + head_dim in {256,512}; JLLM_FLASH_TILED=0 reverts to the old path.
static bool flash_tiled_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_FLASH_TILED");
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
    const int head = blockIdx.x;
    const int kv_head = head / (n_heads / n_kv_heads);  // GQA
    const int tid = threadIdx.x;
    // Per-position stride in the KV cache uses the cache slot head dim, which
    // for Gemma sliding layers (active head_dim 256) is the larger global slot
    // (512). For all other models cache_head_dim == head_dim (no-op).
    const int kv_dim = n_kv_heads * cache_head_dim;
    const int kv_hoff = kv_head * cache_head_dim;

    // Shared memory layout:
    //   s_scores[ATTN_TILE_KV]     — attention scores for current KV tile
    //   s_out[head_dim]            — output accumulator (BUG #6 FIX)
    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_out    = smem + ATTN_TILE_KV;

    __shared__ float s_running_max;
    __shared__ float s_running_sum;
    __shared__ float s_correction;

    // Initialize output accumulator to zero
    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    if (tid == 0) {
        s_running_max = -FLT_MAX;
        s_running_sum = 0.0f;
        s_correction = 1.0f;
    }
    __syncthreads();

    // Tile over KV sequence
    for (int kv_start = 0; kv_start < seq_len; kv_start += ATTN_TILE_KV) {
        int tile_len = min(ATTN_TILE_KV, seq_len - kv_start);

        // ── Step 1: Q × K^T for tile ────────────────────────────
        for (int t = tid; t < tile_len; t += blockDim.x) {
            int kv_pos = kv_start + t;
            // Sliding-window mask: the query (at position seq_len-1) attends
            // only to keys in [seq_len-window, seq_len-1]. Older keys are -inf.
            if (window > 0 && kv_pos < seq_len - window) {
                s_scores[t] = -INFINITY;
                continue;
            }
            float dot = 0.0f;

            for (int d = 0; d < head_dim; d++) {
                float q_val = __half2float(q[head * head_dim + d]);
                float k_val;
                if (kv_int8) {
                    const int8_t* ki = (const int8_t*)k_cache;
                    // Path I3: per-position scale lookup.
                    float ks = k_scales ? k_scales[(int64_t)kv_pos * n_kv_heads + kv_head] : 1.0f;
                    k_val = ki[(int64_t)kv_pos * kv_dim + kv_hoff + d] * ks;
                } else {
                    const half* kf = (const half*)k_cache;
                    k_val = __half2float(
                        kf[(int64_t)kv_pos * kv_dim + kv_hoff + d]);
                }
                dot += q_val * k_val;
            }
            s_scores[t] = dot * scale;
        }
        __syncthreads();

        // ── Step 2: Online softmax ──────────────────────────────
        if (tid == 0) {
            float tile_max = -FLT_MAX;
            for (int t = 0; t < tile_len; ++t) {
                tile_max = fmaxf(tile_max, s_scores[t]);
            }

            const float old_max = s_running_max;
            s_running_max = fmaxf(s_running_max, tile_max);
            s_correction = expf(old_max - s_running_max);
            s_running_sum *= s_correction;
        }
        __syncthreads();

        // Correct existing accumulators
        for (int d = tid; d < head_dim; d += blockDim.x)
            s_out[d] *= s_correction;
        __syncthreads();

        // Exponentiate scores
        if (tid == 0) {
            float tile_sum = 0.0f;
            for (int t = 0; t < tile_len; ++t) {
                float p = expf(s_scores[t] - s_running_max);
                s_scores[t] = p;
                tile_sum += p;
            }
            s_running_sum += tile_sum;
        }
        __syncthreads();

        // ── Step 3: Accumulate P × V (BUG #6 FIX — per-dimension) ──
        // Each thread handles its slice of head_dim
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                int kv_pos = kv_start + t;
                float v_val;
                if (kv_int8) {
                    const int8_t* vi = (const int8_t*)v_cache;
                    // Path I3: per-position scale lookup.
                    float vs = v_scales ? v_scales[(int64_t)kv_pos * n_kv_heads + kv_head] : 1.0f;
                    v_val = vi[(int64_t)kv_pos * kv_dim + kv_hoff + d] * vs;
                } else {
                    const half* vf = (const half*)v_cache;
                    v_val = __half2float(
                        vf[(int64_t)kv_pos * kv_dim + kv_hoff + d]);
                }
                val += s_scores[t] * v_val;
            }
            s_out[d] += val;  // BUG #6 FIX: accumulate into correct dimension
        }
        __syncthreads();
    }

    // ── Finalize: normalize and write output ─────────────────────
    float inv_sum = (s_running_sum > 0.0f) ? 1.0f / s_running_sum : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        output[head * head_dim + d] = __float2half(s_out[d] * inv_sum);
    }
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

// Tiled flash-attention prefill kernel. Templated on HEAD_DIM so the per-query
// Q/O fragments live in registers. INT8 KV only (the genie default). One warp
// per query; the WARPS queries in a block share each K/V tile loaded into
// shared. Validated in tests/test_flash_tiled.cu (the f16 reference).
template<int HEAD_DIM, int WARPS, int TK>
__global__ void flash_attn_prefill_tiled_kernel(
    half* __restrict__ output, const half* __restrict__ q,
    const void* __restrict__ k_cache, const void* __restrict__ v_cache,
    int n_heads, int n_kv_heads, int cache_head_dim, int start_pos,
    float scale, const float* __restrict__ k_scales, const float* __restrict__ v_scales,
    int window, int N)
{
    const int head  = blockIdx.x;
    const int qtile = blockIdx.y;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_dim  = n_kv_heads * cache_head_dim;
    const int kv_hoff = kv_head * cache_head_dim;
    const int q_dim   = n_heads * HEAD_DIM;
    constexpr int DPL = HEAD_DIM / 32;

    const int q_token = qtile * WARPS + warp;     // chunk-local token
    const int qpos    = start_pos + q_token;       // absolute KV position of this query
    const int seq_q   = qpos + 1;                  // causal end

    extern __shared__ char fsmem[];
    int8_t* Ksh = (int8_t*)fsmem;                  // [TK][HEAD_DIM]
    int8_t* Vsh = Ksh + TK * HEAD_DIM;             // [TK][HEAD_DIM]
    float*  ksc = (float*)(Vsh + TK * HEAD_DIM);   // [TK]
    float*  vsc = ksc + TK;                         // [TK]
    float*  Ssh = vsc + TK;                         // [WARPS][TK]

    float qreg[DPL], oreg[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; i++) {
        qreg[i] = (q_token < N)
            ? __half2float(q[(int64_t)q_token * q_dim + head * HEAD_DIM + lane + i*32]) : 0.0f;
        oreg[i] = 0.0f;
    }
    float run_max = -INFINITY, run_sum = 0.0f;

    const int block_qtoken_max = min(qtile * WARPS + WARPS - 1, N - 1);
    const int block_seq = start_pos + block_qtoken_max + 1;  // uniform tile range

    for (int kt = 0; kt < block_seq; kt += TK) {
        const int tklen = min(TK, block_seq - kt);
        for (int idx = threadIdx.x; idx < TK * HEAD_DIM; idx += blockDim.x) {
            const int kk = idx / HEAD_DIM, dd = idx % HEAD_DIM;
            if (kk < tklen) {
                const int64_t off = (int64_t)(kt + kk) * kv_dim + kv_hoff + dd;
                Ksh[idx] = ((const int8_t*)k_cache)[off];
                Vsh[idx] = ((const int8_t*)v_cache)[off];
            } else { Ksh[idx] = 0; Vsh[idx] = 0; }
        }
        for (int kk = threadIdx.x; kk < tklen; kk += blockDim.x) {
            const int kvpos = kt + kk;
            ksc[kk] = k_scales ? k_scales[(int64_t)kvpos * n_kv_heads + kv_head] : 1.0f;
            vsc[kk] = v_scales ? v_scales[(int64_t)kvpos * n_kv_heads + kv_head] : 1.0f;
        }
        __syncthreads();

        if (q_token < N) {
            for (int kk = 0; kk < tklen; kk++) {
                const int kvpos = kt + kk;
                float s;
                if (kvpos > qpos || (window > 0 && kvpos < seq_q - window)) {
                    s = -INFINITY;
                } else {
                    float p = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < DPL; i++)
                        p += qreg[i] * (float)Ksh[kk * HEAD_DIM + lane + i*32];
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
                    s = p * ksc[kk] * scale;
                }
                if (lane == 0) Ssh[warp * TK + kk] = s;
            }
            __syncwarp();
            float lmax = -INFINITY;
            for (int kk = lane; kk < tklen; kk += 32) lmax = fmaxf(lmax, Ssh[warp * TK + kk]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
            if (lmax > -INFINITY) {
                const float new_max = fmaxf(run_max, lmax);
                const float corr = (run_max == -INFINITY) ? 0.0f : expf(run_max - new_max);
                run_max = new_max;
                run_sum *= corr;
                #pragma unroll
                for (int i = 0; i < DPL; i++) oreg[i] *= corr;
                float lsum = 0.0f;
                for (int kk = lane; kk < tklen; kk += 32) {
                    float e = expf(Ssh[warp * TK + kk] - run_max);
                    Ssh[warp * TK + kk] = e; lsum += e;
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
                run_sum += lsum;
                __syncwarp();
                #pragma unroll
                for (int i = 0; i < DPL; i++) {
                    const int d = lane + i*32;
                    float acc = 0.0f;
                    for (int kk = 0; kk < tklen; kk++)
                        acc += Ssh[warp * TK + kk] * (vsc[kk] * (float)Vsh[kk * HEAD_DIM + d]);
                    oreg[i] += acc;
                }
            }
        }
        __syncthreads();
    }

    if (q_token < N) {
        const float inv = (run_sum > 0.0f) ? 1.0f / run_sum : 0.0f;
        #pragma unroll
        for (int i = 0; i < DPL; i++)
            output[(int64_t)q_token * q_dim + head * HEAD_DIM + lane + i*32] = __float2half(oreg[i] * inv);
    }
}

void flash_attention_prefill_batched(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim, int cache_head_dim,
    int N, int start_pos,
    float scale, bool kv_int8,
    const float* k_scales, const float* v_scales, int window, cudaStream_t stream)
{
    if (N <= 0) return;
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

    static bool logged = false;
    if (!logged) {
        fprintf(stderr, "[attention] Using CUDA chunked-prefill attention path\n");
        logged = true;
    }

    // Tiled flash-attention (one warp per query, shared K/V reuse) for INT8 KV
    // + the Gemma head dims. Cuts the O(N^2) K/V re-reads of the old kernel,
    // which dominates long-context prefill.
    if (kv_int8 && flash_tiled_enabled() && (head_dim == 256 || head_dim == 512)) {
        constexpr int WARPS = 8, TK = 32;
        static bool tlog = false;
        if (!tlog) { tlog = true; fprintf(stderr,
            "[attention] tiled flash-attention prefill active (set JLLM_FLASH_TILED=0 to disable)\n"); }
        const dim3 tgrid(n_heads, (N + WARPS - 1) / WARPS, 1);
        const size_t tsmem = (size_t)2 * TK * head_dim + (size_t)2 * TK * sizeof(float)
                           + (size_t)WARPS * TK * sizeof(float);
        if (head_dim == 256)
            flash_attn_prefill_tiled_kernel<256, WARPS, TK><<<tgrid, WARPS*32, tsmem, stream>>>(
                output, q, k_cache, v_cache, n_heads, n_kv_heads, cache_head_dim,
                start_pos, scale, k_scales, v_scales, window, N);
        else
            flash_attn_prefill_tiled_kernel<512, WARPS, TK><<<tgrid, WARPS*32, tsmem, stream>>>(
                output, q, k_cache, v_cache, n_heads, n_kv_heads, cache_head_dim,
                start_pos, scale, k_scales, v_scales, window, N);
        cudaError_t terr = cudaGetLastError();
        if (terr == cudaSuccess) return;
        fprintf(stderr, "[attention] tiled flash-attention launch failed (%s); using old path\n",
                cudaGetErrorString(terr));
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

        const int smem = (ATTN_TILE_KV + head_dim) * (int)sizeof(float);
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
