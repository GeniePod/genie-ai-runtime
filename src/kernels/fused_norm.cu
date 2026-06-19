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

// RMSNorm kernel: half2-vectorized two-pass implementation.
//
// Each thread processes 2 elements per iteration (half2 load), halving
// the number of memory instructions vs the scalar path.  dim is always
// even (power-of-2 hidden dims: 2560, 3840, 4096 …) so no tail handling
// is needed.
//
// Pass 1: accumulate sum-of-squares; warp+block reduce → rrms.
// Pass 2: re-read x, normalize, scale by weight, store.
// Two passes avoid a shared-memory cache for the full hidden vector while
// keeping code correct across any block size.
__global__ void fused_rmsnorm_residual_kernel(
    half*       __restrict__ output,
    const half* __restrict__ x,
    const half* __restrict__ residual,
    const void* __restrict__ weight,
    int rows, int dim, float eps, bool weight_fp32)
{
    const int row    = blockIdx.x;
    if (row >= rows) return;

    const int tid    = threadIdx.x;
    const int stride = blockDim.x;
    const int offset = row * dim;

    // ── Pass 1: sum of squares via half2 loads ───────────────────────────
    float sum_sq = 0.0f;
    for (int i = tid * 2; i < dim; i += stride * 2) {
        half2 xh = *reinterpret_cast<const half2*>(&x[offset + i]);
        float2 v = __half22float2(xh);
        if (residual) {
            half2 rh = *reinterpret_cast<const half2*>(&residual[offset + i]);
            float2 r = __half22float2(rh);
            v.x += r.x;  v.y += r.y;
        }
        sum_sq += v.x * v.x + v.y * v.y;
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

    // ── Pass 2: normalize and write via half2 ─────────────────────────────
    for (int i = tid * 2; i < dim; i += stride * 2) {
        half2 xh = *reinterpret_cast<const half2*>(&x[offset + i]);
        float2 v = __half22float2(xh);
        if (residual) {
            half2 rh = *reinterpret_cast<const half2*>(&residual[offset + i]);
            float2 r = __half22float2(rh);
            v.x += r.x;  v.y += r.y;
        }
        float w0, w1;
        if (weight_fp32) {
            w0 = ((const float*)weight)[i];
            w1 = ((const float*)weight)[i + 1];
        } else {
            half2 wh = *reinterpret_cast<const half2*>(&((const half*)weight)[i]);
            float2 wf = __half22float2(wh);
            w0 = wf.x;  w1 = wf.y;
        }
        half2 out2 = __floats2half2_rn(v.x * rrms * w0, v.y * rrms * w1);
        *reinterpret_cast<half2*>(&output[offset + i]) = out2;
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

}  // namespace jllm
