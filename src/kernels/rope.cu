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

namespace jllm {

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
    int total_pairs = (n_heads + n_kv_heads) * (head_dim / 2);
    int block = BLOCK_SIZE;  // 128
    int grid = (total_pairs + block - 1) / block;

    rope_kernel<<<grid, block, 0, stream>>>(
        q, k, n_heads, n_kv_heads, head_dim, position, theta_base, neox);
}

void rope_inplace_store_kv_fp16(
    half* q, half* k, const half* v,
    half* k_cache_dst, half* v_cache_dst,
    int n_heads, int n_kv_heads, int head_dim,
    int position, float theta_base, bool neox, cudaStream_t stream) {
    const int total_pairs = (n_heads + n_kv_heads) * (head_dim / 2);
    const int kv_dim = n_kv_heads * head_dim;
    const int total = total_pairs > kv_dim ? total_pairs : kv_dim;
    const int block = BLOCK_SIZE;
    const int grid = (total + block - 1) / block;

    rope_store_kv_fp16_kernel<<<grid, block, 0, stream>>>(
        q, k, v, k_cache_dst, v_cache_dst,
        n_heads, n_kv_heads, head_dim, position, theta_base, neox);
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
    const int total_pairs = (n_heads + n_kv_heads) * (head_dim / 2);
    const int kv_dim = n_kv_heads * head_dim;
    const int total = total_pairs > kv_dim ? total_pairs : kv_dim;
    const int block = BLOCK_SIZE;
    const int grid = (total + block - 1) / block;

    rope_store_kv_fp16_kernel_dyn<<<grid, block, 0, stream>>>(
        q, k, v, k_cache_layer_base, v_cache_layer_base,
        n_heads, n_kv_heads, head_dim, kv_stride,
        d_pos, theta_base, neox);
}

}  // namespace jllm
