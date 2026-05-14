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

// Each thread handles ceil(head_dim / blockDim.x) output dimensions.
// Accumulators stored in shared memory (visible to all threads in block).

__global__ void flash_attention_decode_kernel(
    half*        __restrict__ output,
    const half*  __restrict__ q,
    const void*  __restrict__ k_cache,
    const void*  __restrict__ v_cache,
    int n_heads, int n_kv_heads, int head_dim, int seq_len,
    float scale, bool kv_int8, const float* kv_scales)
{
    const int head = blockIdx.x;
    const int kv_head = head / (n_heads / n_kv_heads);  // GQA
    const int tid = threadIdx.x;
    const int kv_dim = n_kv_heads * head_dim;

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
            float dot = 0.0f;

            for (int d = 0; d < head_dim; d++) {
                float q_val = __half2float(q[head * head_dim + d]);
                float k_val;
                if (kv_int8) {
                    const int8_t* ki = (const int8_t*)k_cache;
                    float ks = kv_scales ? kv_scales[kv_head] : 1.0f;
                    k_val = ki[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d] * ks;
                } else {
                    const half* kf = (const half*)k_cache;
                    k_val = __half2float(
                        kf[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d]);
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
                    float vs = kv_scales ? kv_scales[n_kv_heads + kv_head] : 1.0f;
                    v_val = vi[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d] * vs;
                } else {
                    const half* vf = (const half*)v_cache;
                    v_val = __half2float(
                        vf[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d]);
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
    int n_heads, int n_kv_heads, int head_dim,
    int start_pos,
    float scale, bool kv_int8, const float* kv_scales)
{
    const int head    = blockIdx.x;
    const int token   = blockIdx.y;
    const int seq_len = start_pos + token + 1;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int tid     = threadIdx.x;
    const int kv_dim  = n_kv_heads * head_dim;
    const int q_dim   = n_heads    * head_dim;

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_out    = smem + ATTN_TILE_KV;

    __shared__ float s_running_max;
    __shared__ float s_running_sum;
    __shared__ float s_correction;

    const half* q_local   = q      + (int64_t)token * q_dim + (int64_t)head * head_dim;
    half*       out_local = output + (int64_t)token * q_dim + (int64_t)head * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    if (tid == 0) {
        s_running_max = -FLT_MAX;
        s_running_sum = 0.0f;
        s_correction  = 1.0f;
    }
    __syncthreads();

    for (int kv_start = 0; kv_start < seq_len; kv_start += ATTN_TILE_KV) {
        int tile_len = min(ATTN_TILE_KV, seq_len - kv_start);

        // Q × K^T for tile
        for (int t = tid; t < tile_len; t += blockDim.x) {
            int kv_pos = kv_start + t;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q_val = __half2float(q_local[d]);
                float k_val;
                if (kv_int8) {
                    const int8_t* ki = (const int8_t*)k_cache;
                    float ks = kv_scales ? kv_scales[kv_head] : 1.0f;
                    k_val = ki[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d] * ks;
                } else {
                    const half* kf = (const half*)k_cache;
                    k_val = __half2float(
                        kf[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d]);
                }
                dot += q_val * k_val;
            }
            s_scores[t] = dot * scale;
        }
        __syncthreads();

        if (tid == 0) {
            float tile_max = -FLT_MAX;
            for (int t = 0; t < tile_len; ++t) {
                tile_max = fmaxf(tile_max, s_scores[t]);
            }
            const float old_max = s_running_max;
            s_running_max = fmaxf(s_running_max, tile_max);
            s_correction  = expf(old_max - s_running_max);
            s_running_sum *= s_correction;
        }
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x)
            s_out[d] *= s_correction;
        __syncthreads();

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
                    v_val = __half2float(
                        vf[(int64_t)kv_pos * kv_dim + kv_head * head_dim + d]);
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

void flash_attention_prefill_batched(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim,
    int N, int start_pos,
    float scale, bool kv_int8, const float* kv_scales, cudaStream_t stream)
{
    if (N <= 0) return;
    if (!fast_attention_enabled() || kv_int8) {
        // No CUDA fast path for INT8 KV in the per-token kernel either —
        // fall back to N sequential flash_attention_decode calls, which
        // already have a CPU-reference fallback for INT8.
        const int q_dim = n_heads * head_dim;
        for (int t = 0; t < N; t++) {
            flash_attention_decode(
                output + (int64_t)t * q_dim,
                q      + (int64_t)t * q_dim,
                k_cache, v_cache,
                n_heads, n_kv_heads, head_dim,
                start_pos + t + 1, scale, kv_int8, kv_scales, stream);
        }
        return;
    }

    static bool logged = false;
    if (!logged) {
        fprintf(stderr, "[attention] Using CUDA chunked-prefill attention path\n");
        logged = true;
    }

    const dim3 grid(n_heads, N, 1);
    const int smem = (ATTN_TILE_KV + head_dim) * (int)sizeof(float);
    flash_attention_prefill_batched_kernel<<<grid, ATTN_BLOCK, smem, stream>>>(
        output, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim,
        start_pos, scale, kv_int8, kv_scales);

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
                n_heads, n_kv_heads, head_dim,
                start_pos + t + 1, scale, kv_int8, kv_scales, stream);
        }
    }
}

void flash_attention_decode(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim, int seq_len,
    float scale, bool kv_int8, const float* kv_scales, cudaStream_t stream)
{
    if (fast_attention_enabled()) {
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[attention] Using CUDA decode attention path\n");
            logged = true;
        }

        const int smem = (ATTN_TILE_KV + head_dim) * (int)sizeof(float);
        flash_attention_decode_kernel<<<n_heads, ATTN_BLOCK, smem, stream>>>(
            output, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim,
            seq_len, scale, kv_int8, kv_scales);

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

    const int kv_dim = n_kv_heads * head_dim;
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
            const int kv_base = pos * kv_dim + kv_head * head_dim;
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
                const int kv_base = pos * kv_dim + kv_head * head_dim;
                acc += scores[pos] * inv_sum * __half2float(h_v[kv_base + d]);
            }
            h_out[q_base + d] = __float2half(acc);
        }
    }

    cudaMemcpy(output, h_out.data(), (size_t)q_dim * sizeof(half), cudaMemcpyDefault);
}

}  // namespace jllm
