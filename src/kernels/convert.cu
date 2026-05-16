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

__global__ void swiglu_kernel(
    half*       __restrict__ output,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int rows, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * dim;
    if (idx >= total) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu = g / (1.0f + expf(-g));
    output[idx] = __float2half(silu * u);
}

void fused_swiglu(half* output, const half* gate, const half* up,
                  int rows, int intermediate_dim, cudaStream_t stream) {
    int total = rows * intermediate_dim;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    swiglu_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        output, gate, up, rows, intermediate_dim);
}

}  // namespace jllm
