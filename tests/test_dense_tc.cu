// Standalone reference/validation for the fp16 tensor-core dense GEMM.
// Build: nvcc -arch=sm_87 -O3 --use_fast_math tests/test_dense_tc.cu -o t && ./t [M N K]
// Validates wmma 16x16x16 out=x.W^T vs CPU ref (rel-RMS ~3e-4) + reports GFLOP/s.
// Not in CMake.
// test_dense_tc.cu — tiled fp16 tensor-core dense GEMM (nvcuda::wmma) to replace
// genie's naive gemm_dense_batched_kernel (Gemma 4 F32 inp_gate/proj, ~0.6% of peak).
//   out[N x M] = x[N x K] (half, row-major) . W^T,  W[M x K] (F32, row-major).
//   out[n*M + m] = sum_k W[m][k] * x[n][k].
// Each warp computes one 16x16 output tile via wmma 16x16x16 f16->f32.
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace nvcuda;

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} }while(0)

template <int WM, int WN>   // warps along M and N (block tile = 16*WM x 16*WN)
__launch_bounds__(WM*WN*32, 2)
__global__ void dense_gemm_wmma(half* __restrict__ out, const float* __restrict__ W,
                                const half* __restrict__ x, int M, int N, int K) {
    constexpr int BM = WM*16, BN = WN*16;
    const int wid = threadIdx.x >> 5, wm = wid / WN, wn = wid % WN;
    const int row0 = blockIdx.x * BM, col0 = blockIdx.y * BN;
    const int wrow = row0 + wm*16, wcol = col0 + wn*16;

    __shared__ __align__(16) half Ash[BM][16];   // W tile (M x K-step), row-major
    __shared__ __align__(16) half Bsh[BN][16];   // x tile (N x K-step), row-major
    __shared__ float Csh[BM][BN];

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b;  // B[k][n]=x[n][k]
    wmma::fragment<wmma::accumulator, 16,16,16, float> c;
    wmma::fill_fragment(c, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int idx = threadIdx.x; idx < BM*16; idx += blockDim.x) {
            int r = idx >> 4, cc = idx & 15, g = row0 + r;
            Ash[r][cc] = (g < M) ? __float2half(W[(int64_t)g*K + k0 + cc]) : __float2half(0.f);
        }
        for (int idx = threadIdx.x; idx < BN*16; idx += blockDim.x) {
            int r = idx >> 4, cc = idx & 15, g = col0 + r;
            Bsh[r][cc] = (g < N) ? x[(int64_t)g*K + k0 + cc] : __float2half(0.f);
        }
        __syncthreads();
        wmma::load_matrix_sync(a, &Ash[wm*16][0], 16);  // A[i][k]=Ash[wm*16+i][k]=W[wrow+i][k0+k]
        wmma::load_matrix_sync(b, &Bsh[wn*16][0], 16);  // B[k][n]=Bsh[wn*16+n][k]=x[wcol+n][k0+k]
        wmma::mma_sync(c, a, b, c);
        __syncthreads();
    }
    wmma::store_matrix_sync(&Csh[wm*16][wn*16], c, BN, wmma::mem_row_major); // Csh[m][n]
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM*BN; idx += blockDim.x) {
        int m = idx / BN, n = idx % BN, gm = row0 + m, gn = col0 + n;
        if (gm < M && gn < N) out[(int64_t)gn*M + gm] = __float2half(Csh[m][n]);
    }
}

int main(int argc, char** argv) {
    int M = argc>1?atoi(argv[1]):256, N = argc>2?atoi(argv[2]):1261, K = argc>3?atoi(argv[3]):1536;
#ifndef CWM
#define CWM 2
#endif
#ifndef CWN
#define CWN 4
#endif
    constexpr int WM=CWM, WN=CWN, BM=WM*16, BN=WN*16;
    printf("dense WMMA GEMM: M=%d N=%d K=%d (tile %dx%d, %d warps)\n", M,N,K,BM,BN,WM*WN);
    srand(11);
    std::vector<float> W((size_t)M*K); for(auto&v:W) v=(rand()/(float)RAND_MAX)*2-1;
    std::vector<float> Xf((size_t)N*K); for(auto&v:Xf) v=(rand()/(float)RAND_MAX)*2-1;
    std::vector<half> X((size_t)N*K); for(size_t i=0;i<X.size();i++) X[i]=__float2half(Xf[i]);
    std::vector<float> ref((size_t)N*M);
    for(int n=0;n<N;n++) for(int m=0;m<M;m++){ double a=0;
        for(int k=0;k<K;k++) a+=(double)W[(size_t)m*K+k]*__half2float(X[(size_t)n*K+k]);
        ref[(size_t)n*M+m]=(float)a; }
    float *dW; half *dX,*dout;
    CK(cudaMalloc(&dW,W.size()*4)); CK(cudaMalloc(&dX,X.size()*2)); CK(cudaMalloc(&dout,(size_t)N*M*2));
    CK(cudaMemcpy(dW,W.data(),W.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX,X.data(),X.size()*2,cudaMemcpyHostToDevice));
    dim3 grid((M+BM-1)/BM,(N+BN-1)/BN), block(WM*WN*32);
    dense_gemm_wmma<WM,WN><<<grid,block>>>(dout,dW,dX,M,N,K);
    CK(cudaDeviceSynchronize());
    std::vector<half> oh((size_t)N*M); CK(cudaMemcpy(oh.data(),dout,(size_t)N*M*2,cudaMemcpyDeviceToHost));
    double sq=0,rq=0,maxrel=0; int nbad=0;
    for(size_t i=0;i<(size_t)N*M;i++) rq+=(double)ref[i]*ref[i];
    double sig=sqrt(rq/((double)N*M));
    for(size_t i=0;i<(size_t)N*M;i++){ double g=__half2float(oh[i]),r=ref[i],ad=fabs(g-r),rel=ad/(fabs(r)+1e-6);
        if(rel>maxrel)maxrel=rel; if(rel>0.05&&ad>0.02*sig)nbad++; sq+=ad*ad; }
    printf("rel-RMS=%.5f sig=%.3f maxrel=%.3f nbad=%d/%d -> %s\n", sqrt(sq/(rq+1e-9)),sig,maxrel,nbad,N*M,
           (sqrt(sq/(rq+1e-9))<0.02&&nbad==0)?"PASS":"FAIL");
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for(int w=0;w<5;w++) dense_gemm_wmma<WM,WN><<<grid,block>>>(dout,dW,dX,M,N,K);
    CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
    for(int i=0;i<50;i++) dense_gemm_wmma<WM,WN><<<grid,block>>>(dout,dW,dX,M,N,K);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=50;
    printf("%.4f ms/iter  %.1f GFLOP/s  (%.1f%% of 16.2 fp16-TC TFLOP/s; naive baseline ~99)\n",
           ms, 2.0*M*N*K/1e9/(ms/1e3), 2.0*M*N*K/1e9/(ms/1e3)/16200.0*100.0);
    return 0;
}
