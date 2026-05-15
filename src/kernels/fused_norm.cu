// fused_norm.cu — Fused RMSNorm + Residual Add (SM 8.7)
// Handles both FP32 and FP16 norm weights (GGUF stores them as F32 typically).

#include "jllm_kernels.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace jllm {

static bool fast_norm_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_FAST_NORM");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

// Single-pass RMSNorm: avoid the previous two-pass sdata cache entirely.
//
// On Qwen3-4B (hidden_dim 2560, block 128) the previous implementation
// produced output with every-other element zeroed — the signature of a
// shared-memory layout bug where `extern __shared__ float sdata[]`
// (sized hidden_dim×4) and the statically-allocated
// `__shared__ float warp_sums[4]` could end up at overlapping or
// otherwise miscomputed offsets, plus a missing __syncthreads between
// Pass 1 (write sdata) and Pass 2 (read sdata) which technically only
// each thread reads what it wrote but the compiler is free to reorder
// the writes around the inter-warp reduction.
//
// This rewrite eliminates the cache entirely. Each thread reads x once
// for the sum-of-squares pass, then re-reads x for the scale-and-write
// pass. Two reads of a 5 KB hidden-state vector cost ~100 ns of LPDDR5
// bandwidth — invisible next to the 100s of µs the kernel actually runs
// for — and the correctness win is unambiguous.
__global__ void fused_rmsnorm_residual_kernel(
    half*       __restrict__ output,
    const half* __restrict__ x,
    const half* __restrict__ residual,
    const void* __restrict__ weight,
    int rows, int dim, float eps, bool weight_fp32)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int offset = row * dim;

    // ── Pass 1: compute sum of squares (read x[+residual] once) ─────────
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        float val = __half2float(x[offset + i]);
        if (residual)
            val += __half2float(residual[offset + i]);
        sum_sq += val * val;
    }

    // Warp-level reduction first (intra-warp, no shared memory).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    // Block-level reduction via shared memory. Sized for up to 32 warps
    // (block size ≤ 1024). Statically allocated; doesn't touch the
    // dynamic shared memory pool, so launch-time smem can be 0.
    __shared__ float block_partial[32];
    const int warp_id   = tid >> 5;     // tid / 32
    const int warp_lane = tid & 31;
    if (warp_lane == 0) block_partial[warp_id] = sum_sq;
    __syncthreads();

    // Thread 0 reduces across warps and writes rrms back into slot 0.
    if (tid == 0) {
        const int n_warps = (stride + 31) >> 5;
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += block_partial[w];
        block_partial[0] = rsqrtf(total / dim + eps);
    }
    __syncthreads();

    const float rrms = block_partial[0];

    // ── Pass 2: read x again, normalize, scale by weight, write ─────────
    for (int i = tid; i < dim; i += stride) {
        float val = __half2float(x[offset + i]);
        if (residual)
            val += __half2float(residual[offset + i]);
        float w = weight_fp32
            ? ((const float*)weight)[i]
            : __half2float(((const half*)weight)[i]);
        output[offset + i] = __float2half(val * rrms * w);
    }
}

__device__ __forceinline__ half add_residual_half(const half a, const half* residual,
                                                  int idx) {
    if (!residual) return a;
    return __float2half(__half2float(a) + __half2float(residual[idx]));
}

__global__ void fused_rmsnorm_residual_store_kernel(
    half*       __restrict__ residual_out,
    half*       __restrict__ norm_out,
    const half* __restrict__ x,
    const half* __restrict__ residual,
    const void* __restrict__ weight,
    int rows, int dim, float eps, bool weight_fp32)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int offset = row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        const int idx = offset + i;
        const half sum_h = add_residual_half(x[idx], residual, idx);
        const float val = __half2float(sum_h);
        sum_sq += val * val;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    __shared__ float block_partial[32];
    const int warp_id   = tid >> 5;
    const int warp_lane = tid & 31;
    if (warp_lane == 0) block_partial[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        const int n_warps = (stride + 31) >> 5;
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += block_partial[w];
        block_partial[0] = rsqrtf(total / dim + eps);
    }
    __syncthreads();

    const float rrms = block_partial[0];
    for (int i = tid; i < dim; i += stride) {
        const int idx = offset + i;
        const half sum_h = add_residual_half(x[idx], residual, idx);
        residual_out[idx] = sum_h;

        const float val = __half2float(sum_h);
        const float w = weight_fp32
            ? ((const float*)weight)[i]
            : __half2float(((const half*)weight)[i]);
        norm_out[idx] = __float2half(val * rrms * w);
    }
}

void fused_rmsnorm_residual(half* output, const half* x, const half* residual,
                            const void* weight, int rows, int hidden_dim,
                            float eps, bool weight_fp32, cudaStream_t stream) {
    if (fast_norm_enabled()) {
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[norm] Using CUDA RMSNorm path\n");
            logged = true;
        }

        const int block = 128;
        fused_rmsnorm_residual_kernel<<<rows, block, 0, stream>>>(
            output, x, residual, weight, rows, hidden_dim, eps, weight_fp32);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            return;
        }

        fprintf(stderr, "[norm] CUDA RMSNorm launch failed: %s; falling back to CPU reference\n",
                cudaGetErrorString(err));
    }

    const int total = rows * hidden_dim;
    std::vector<half> h_x(total);
    std::vector<half> h_out(total);
    std::vector<float> h_w(hidden_dim);

    cudaStreamSynchronize(stream);
    cudaMemcpy(h_x.data(), x, total * sizeof(half), cudaMemcpyDeviceToHost);

    std::vector<half> h_res;
    if (residual) {
        h_res.resize(total);
        cudaMemcpy(h_res.data(), residual, total * sizeof(half), cudaMemcpyDeviceToHost);
    }

    if (weight_fp32) {
        cudaMemcpy(h_w.data(), weight, hidden_dim * sizeof(float), cudaMemcpyDefault);
    } else {
        std::vector<half> h_wh(hidden_dim);
        cudaMemcpy(h_wh.data(), weight, hidden_dim * sizeof(half), cudaMemcpyDefault);
        for (int i = 0; i < hidden_dim; i++) h_w[i] = __half2float(h_wh[i]);
    }

    for (int row = 0; row < rows; row++) {
        const int off = row * hidden_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float val = __half2float(h_x[off + i]);
            if (residual) val += __half2float(h_res[off + i]);
            sum_sq += val * val;
        }

        const float rrms = 1.0f / sqrtf(sum_sq / hidden_dim + eps);
        for (int i = 0; i < hidden_dim; i++) {
            float val = __half2float(h_x[off + i]);
            if (residual) val += __half2float(h_res[off + i]);
            h_out[off + i] = __float2half(val * rrms * h_w[i]);
        }
    }

    cudaMemcpy(output, h_out.data(), total * sizeof(half), cudaMemcpyHostToDevice);
}

void fused_rmsnorm_residual_store(half* residual_out, half* norm_out,
                                  const half* x, const half* residual,
                                  const void* weight, int rows,
                                  int hidden_dim, float eps,
                                  bool weight_fp32, cudaStream_t stream) {
    if (fast_norm_enabled()) {
        const int block = 128;
        fused_rmsnorm_residual_store_kernel<<<rows, block, 0, stream>>>(
            residual_out, norm_out, x, residual, weight, rows, hidden_dim,
            eps, weight_fp32);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            return;
        }

        fprintf(stderr,
                "[norm] CUDA residual+RMSNorm launch failed: %s; falling back to CPU reference\n",
                cudaGetErrorString(err));
    }

    const int total = rows * hidden_dim;
    std::vector<half> h_x(total);
    std::vector<half> h_sum(total);
    std::vector<half> h_norm(total);
    std::vector<float> h_w(hidden_dim);

    cudaStreamSynchronize(stream);
    cudaMemcpy(h_x.data(), x, total * sizeof(half), cudaMemcpyDeviceToHost);

    std::vector<half> h_res;
    if (residual) {
        h_res.resize(total);
        cudaMemcpy(h_res.data(), residual, total * sizeof(half), cudaMemcpyDeviceToHost);
    }

    if (weight_fp32) {
        cudaMemcpy(h_w.data(), weight, hidden_dim * sizeof(float), cudaMemcpyDefault);
    } else {
        std::vector<half> h_wh(hidden_dim);
        cudaMemcpy(h_wh.data(), weight, hidden_dim * sizeof(half), cudaMemcpyDefault);
        for (int i = 0; i < hidden_dim; i++) h_w[i] = __half2float(h_wh[i]);
    }

    for (int row = 0; row < rows; row++) {
        const int off = row * hidden_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            const int idx = off + i;
            const float b = residual ? __half2float(h_res[idx]) : 0.0f;
            h_sum[idx] = __float2half(__half2float(h_x[idx]) + b);
            const float val = __half2float(h_sum[idx]);
            sum_sq += val * val;
        }

        const float rrms = 1.0f / sqrtf(sum_sq / hidden_dim + eps);
        for (int i = 0; i < hidden_dim; i++) {
            const int idx = off + i;
            h_norm[idx] = __float2half(__half2float(h_sum[idx]) * rrms * h_w[i]);
        }
    }

    cudaMemcpy(residual_out, h_sum.data(), total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(norm_out, h_norm.data(), total * sizeof(half), cudaMemcpyHostToDevice);
}

}  // namespace jllm
