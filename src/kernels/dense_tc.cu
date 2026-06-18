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
#include <cstdint>
#include <cstdlib>
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

}  // namespace jllm
