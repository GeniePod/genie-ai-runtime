// dense_tc.cu — tiled fp16 tensor-core dense GEMM (nvcuda::wmma) for genie's
// dense (F32/F16/BF16) prefill GEMMs — Gemma 4's per-layer inp_gate/proj
// projections. Replaces the naive warp-per-output-element gemm_dense_batched
// (~0.6% of fp16 peak) with a 16x16x16 wmma kernel (~5x faster on those shapes).
//   out[N x M] = x[N x K] (half, row-major) . W^T,  W[M x K] (wtype), row-major.
//   out[n*M + m] = sum_k W[m][k] * x[n][k].
// Standalone-validated (rel-RMS 3e-4 vs CPU ref). W must already be a device
// pointer (caller resolves host-mapped weights via resolve_weight_device_ptr).
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
using namespace nvcuda;

namespace jllm {

// wtype: 0=F32, 1=F16, 30=BF16 (matches gemv_q4.cu's dense_weight).
static __device__ __forceinline__ float dense_w(const void* W, int wtype, int64_t i) {
    if (wtype == 0)      return ((const float*)W)[i];
    else if (wtype == 1) return __half2float(((const half*)W)[i]);
    uint32_t bits = (uint32_t)((const uint16_t*)W)[i] << 16;  // BF16
    return __int_as_float(bits);
}

template <int WM, int WN>   // warps along M and N; block tile = 16*WM x 16*WN
__launch_bounds__(WM*WN*32, 2)
__global__ void dense_gemm_wmma_kernel(half* __restrict__ out, const void* __restrict__ W,
                                       int wtype, const half* __restrict__ x,
                                       int M, int N, int K) {
    constexpr int BM = WM*16, BN = WN*16;
    const int wid = threadIdx.x >> 5, wm = wid / WN, wn = wid % WN;
    const int row0 = blockIdx.x * BM, col0 = blockIdx.y * BN;

    __shared__ __align__(16) half Ash[BM][16];
    __shared__ __align__(16) half Bsh[BN][16];
    __shared__ float Csh[BM][BN];

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b;  // B[k][n]=x[n][k]
    wmma::fragment<wmma::accumulator, 16,16,16, float> c;
    wmma::fill_fragment(c, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int idx = threadIdx.x; idx < BM*16; idx += blockDim.x) {
            int r = idx >> 4, cc = idx & 15, g = row0 + r;
            Ash[r][cc] = (g < M) ? __float2half(dense_w(W, wtype, (int64_t)g*K + k0 + cc)) : __float2half(0.f);
        }
        for (int idx = threadIdx.x; idx < BN*16; idx += blockDim.x) {
            int r = idx >> 4, cc = idx & 15, g = col0 + r;
            Bsh[r][cc] = (g < N) ? x[(int64_t)g*K + k0 + cc] : __float2half(0.f);
        }
        __syncthreads();
        wmma::load_matrix_sync(a, &Ash[wm*16][0], 16);
        wmma::load_matrix_sync(b, &Bsh[wn*16][0], 16);
        wmma::mma_sync(c, a, b, c);
        __syncthreads();
    }
    wmma::store_matrix_sync(&Csh[wm*16][wn*16], c, BN, wmma::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM*BN; idx += blockDim.x) {
        int m = idx / BN, n = idx % BN, gm = row0 + m, gn = col0 + n;
        if (gm < M && gn < N) out[(int64_t)gn*M + gm] = __float2half(Csh[m][n]);
    }
}

// Default on; JLLM_DENSE_TC=0 reverts to the naive gemm_dense_batched_kernel.
bool dense_tc_enabled() {
    static int e = -1;
    if (e < 0) { const char* v = getenv("JLLM_DENSE_TC"); e = (v && v[0] == '0') ? 0 : 1; }
    return e;
}

// W must already be device-resident (caller resolves). Returns false if the
// shape isn't handled (K not a multiple of 16) so the caller can fall back.
bool gemm_dense_tc(half* out, const void* Wdev, int wtype, const half* x,
                   int M, int N, int K, cudaStream_t stream) {
    if (K % 16 != 0) return false;
    constexpr int WM = 4, WN = 4, BM = WM*16, BN = WN*16;
    dim3 grid((M + BM - 1)/BM, (N + BN - 1)/BN), block(WM*WN*32);
    dense_gemm_wmma_kernel<WM,WN><<<grid, block, 0, stream>>>(out, Wdev, wtype, x, M, N, K);
    return true;
}

// ── cuBLAS dense GEMM (F16/BF16 weights) ─────────────────────────────────────
// The wmma kernel above accumulates through shared memory and stalls (ncu:
// ~5.5% TC util, 44% SM throughput at large N), so cuBLAS's tuned GEMM is much
// faster for the big batched projections (Gemma's PLE model_proj is BF16).
// Default on; JLLM_DENSE_CUBLAS=0 reverts to the wmma kernel. F32 weights stay
// on wmma (returns false). out[N×M] row-major = x[N×K]·W[M×K]^T, computed as
// the column-major [M×N] = opT(W[K×M]) · opN(x[K×N]).
bool dense_cublas_enabled() {
    static int e = -1;
    if (e < 0) { const char* v = getenv("JLLM_DENSE_CUBLAS"); e = (v && v[0] == '0') ? 0 : 1; }
    return e;
}

__global__ void f16_to_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                   const half* __restrict__ in, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__half2float(in[i]));
}

bool gemm_dense_cublas(half* out, const void* Wdev, int wtype, const half* x,
                       int M, int N, int K, cudaStream_t stream) {
    if (wtype != 1 && wtype != 30) return false;   // F16 / BF16 only
    static cublasHandle_t h = nullptr;
    if (!h) { if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { h = nullptr; return false; } }
    cublasSetStream(h, stream);

    cudaDataType_t wt, xt;
    const void* xptr = x;
    if (wtype == 30) {                  // BF16 weight: convert activations to BF16
        static __nv_bfloat16* xbf = nullptr; static size_t cap = 0;
        size_t n = (size_t)N * K;
        if (n > cap) {
            if (xbf) cudaFree(xbf);
            if (cudaMalloc(&xbf, n * sizeof(__nv_bfloat16)) != cudaSuccess) { xbf = nullptr; cap = 0; return false; }
            cap = n;
        }
        int t = 256; int64_t b = ((int64_t)n + t - 1) / t;
        f16_to_bf16_kernel<<<b, t, 0, stream>>>(xbf, x, n);
        wt = CUDA_R_16BF; xt = CUDA_R_16BF; xptr = xbf;
    } else {                            // F16 weight
        wt = CUDA_R_16F; xt = CUDA_R_16F;
    }
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        &alpha, Wdev, wt, K, xptr, xt, K,
        &beta,  out, CUDA_R_16F, M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return s == CUBLAS_STATUS_SUCCESS;
}

}  // namespace jllm
