// convert.cu — FP16 ↔ INT8 conversion for KV cache quantization (SM 8.7)
//
// KV cache quantization: store K/V in INT8 instead of FP16 → 2× less memory.
// Per-head absmax quantization: scale = max(|vals|) / 127

#include "jllm_kernels.h"
#include <cuda_fp16.h>
#include <cmath>

namespace jllm {

// FP16 → INT8 with per-row absmax scaling
//
// Path I phase I2 (#62) corrections:
//   - row_max __shared__ was used by atomicMax without initialization.
//     First atomicMax was comparing against whatever garbage was in
//     shared memory. Now zeroed before the warp-reductions land.
//   - scale_out[row] was unconditionally written. Earlier callers
//     passed nullptr (alpha.2..alpha.11 INT8 KV stub); the write would
//     silently corrupt or crash depending on GPU state. Now null-checked.
//
// The 4-byte alias trick (atomicMax on (int*)&row_max with the bit
// pattern of the float) is the standard NVIDIA-recommended technique
// for reducing positive floats — float-bit-pattern monotonicity matches
// int monotonicity for non-negative values, which `fabsf` guarantees.
__global__ void fp16_to_int8_kernel(
    int8_t*      __restrict__ dst,
    float*       __restrict__ scale_out,
    const half*  __restrict__ src,
    int rows, int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int offset = row * cols;

    __shared__ float row_max;
    if (tid == 0) row_max = 0.0f;
    __syncthreads();

    // Pass 1: find absmax for this row
    float local_max = 0.0f;
    for (int c = tid; c < cols; c += stride) {
        float val = fabsf(__half2float(src[offset + c]));
        local_max = fmaxf(local_max, val);
    }

    // Warp reduce max
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));

    if (tid % 32 == 0) atomicMax((int*)&row_max, __float_as_int(local_max));
    __syncthreads();

    float scale = row_max / 127.0f;
    float inv_scale = (scale > 0.0f) ? 127.0f / row_max : 0.0f;

    if (tid == 0 && scale_out != nullptr) scale_out[row] = scale;

    // Pass 2: quantize
    for (int c = tid; c < cols; c += stride) {
        float val = __half2float(src[offset + c]);
        int q = __float2int_rn(val * inv_scale);  // round to nearest
        q = max(-127, min(127, q));                // clamp to INT8 range
        dst[offset + c] = (int8_t)q;
    }
}

void fp16_to_int8(int8_t* dst, float* scale_out, const half* src,
                  int rows, int cols, cudaStream_t stream) {
    fp16_to_int8_kernel<<<rows, BLOCK_SIZE, 0, stream>>>(
        dst, scale_out, src, rows, cols);
}

// ── Fused SwiGLU (bonus: belongs here as a simple elementwise) ──────────
//
// output = silu(gate) * up
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// half2 vectorized: 2 elements per thread.  intermediate_dim is always
// even (power-of-2 FFN dims), so the single-element tail rarely fires.

__global__ void swiglu_kernel(
    half*       __restrict__ output,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int rows, int dim)
{
    const int idx2  = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int total = rows * dim;
    if (idx2 >= total) return;

    half2 g2 = *reinterpret_cast<const half2*>(&gate[idx2]);
    half2 u2 = *reinterpret_cast<const half2*>(&up[idx2]);
    float g0 = __half2float(g2.x), g1 = __half2float(g2.y);
    float u0 = __half2float(u2.x), u1 = __half2float(u2.y);
    half2 out2;
    out2.x = __float2half(g0 / (1.0f + expf(-g0)) * u0);
    out2.y = (idx2 + 1 < total) ? __float2half(g1 / (1.0f + expf(-g1)) * u1)
                                 : __float2half(0.0f);
    *reinterpret_cast<half2*>(&output[idx2]) = out2;
}

void fused_swiglu(half* output, const half* gate, const half* up,
                  int rows, int intermediate_dim, cudaStream_t stream) {
    const int total2 = (rows * intermediate_dim + 1) / 2;
    const int grid   = (total2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    swiglu_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        output, gate, up, rows, intermediate_dim);
}

// ── Fused GeGLU (Gemma) ─────────────────────────────────────────────────
//
// output = gelu_tanh(gate) * up
// gelu_tanh(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
// This is the "gelu_pytorch_tanh" approximation Gemma's FFN uses.
// half2 vectorized: 2 elements per thread.

__global__ void geglu_kernel(
    half*       __restrict__ output,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int rows, int dim)
{
    const int idx2  = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int total = rows * dim;
    if (idx2 >= total) return;

    half2 g2 = *reinterpret_cast<const half2*>(&gate[idx2]);
    half2 u2 = *reinterpret_cast<const half2*>(&up[idx2]);
    float g0 = __half2float(g2.x), g1 = __half2float(g2.y);
    float u0 = __half2float(u2.x), u1 = __half2float(u2.y);
    const float kSqrt2OverPi = 0.7978845608028654f;
    auto gelu = [&](float g) {
        return 0.5f * g * (1.0f + tanhf(kSqrt2OverPi * (g + 0.044715f * g * g * g)));
    };
    half2 out2;
    out2.x = __float2half(gelu(g0) * u0);
    out2.y = (idx2 + 1 < total) ? __float2half(gelu(g1) * u1) : __float2half(0.0f);
    *reinterpret_cast<half2*>(&output[idx2]) = out2;
}

void fused_geglu(half* output, const half* gate, const half* up,
                 int rows, int intermediate_dim, cudaStream_t stream) {
    const int total2 = (rows * intermediate_dim + 1) / 2;
    const int grid   = (total2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    geglu_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        output, gate, up, rows, intermediate_dim);
}

// ── Scalar multiply (Gemma layer scale: x *= layer_scale_val) ────────────
// half2 vectorized: 2 elements per thread using __hmul2.
__global__ void vec_scale_kernel(half* x, int n, float s) {
    const int i2 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i2 + 1 < n) {
        half2 sh = __float2half2_rn(s);
        *reinterpret_cast<half2*>(&x[i2]) =
            __hmul2(*reinterpret_cast<const half2*>(&x[i2]), sh);
    } else if (i2 < n) {
        x[i2] = __float2half(__half2float(x[i2]) * s);
    }
}

void vec_scale(half* x, int n, float s, cudaStream_t stream) {
    const int n2   = (n + 1) >> 1;
    const int grid = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vec_scale_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, n, s);
}

// ── Final-logit soft-cap (Gemma: x = c * tanh(x / c)) ─────────────────────
__global__ void logit_softcap_kernel(float* x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

void logit_softcap(float* x, int n, float cap, cudaStream_t stream) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    logit_softcap_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, n, cap);
}

}  // namespace jllm
