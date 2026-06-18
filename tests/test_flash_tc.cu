// test_flash_tc.cu — tensor-core flash attention prefill for genie.
//
// The long-context prefill bottleneck is the attention's O(N^2) COMPUTE (Q.K^T
// and P.V dot products), which scalar code can't do fast on the 8-SM Orin.
// llama.cpp solves it with tensor cores. This is genie's version using the
// wmma API (16x16x16 f16 MMA):
//   S = Q @ K^T   (K loaded col-major = K^T) -> f32 accumulator
//   softmax(S) in shared (online, per query row, causal + sliding)
//   O += P @ V    (P cast to f16) -> f32 accumulator
// One warp per 16-query tile. f16 K/V here (algorithm validation); int8-KV +
// occupancy tuning + the engine port come after this is correct.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace nvcuda;
#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} }while(0)

constexpr int D  = 256;   // head_dim
constexpr int MQ = 16;    // queries per tile (wmma M)
constexpr int TK = 16;    // keys per tile (wmma N/K)

// naive reference (one block per query)
__global__ void attn_naive(half* __restrict__ out, const half* __restrict__ q,
                           const half* __restrict__ k, const half* __restrict__ v,
                           int N, int H, float scale, int window) {
    const int head = blockIdx.x, token = blockIdx.y, tid = threadIdx.x;
    const int seq = token + 1;
    extern __shared__ float sm[];
    float* sc = sm; float* so = sm + 4096;
    __shared__ float rmax, rsum;
    for (int d = tid; d < D; d += blockDim.x) so[d] = 0.0f;
    if (tid == 0) { rmax = -INFINITY; rsum = 0.0f; }
    __syncthreads();
    const half* ql = q + ((int64_t)token*H+head)*D;
    for (int p = tid; p < seq; p += blockDim.x) {
        float s;
        if (window>0 && p < seq-window) s = -INFINITY;
        else { float dt=0; for(int d=0;d<D;d++) dt += __half2float(ql[d])*__half2float(k[((int64_t)p*H+head)*D+d]); s=dt*scale; }
        sc[p]=s;
    }
    __syncthreads();
    if (tid==0){ float m=-INFINITY; for(int p=0;p<seq;p++) m=fmaxf(m,sc[p]); rmax=m; float su=0; for(int p=0;p<seq;p++){float e=expf(sc[p]-m); sc[p]=e; su+=e;} rsum=su; }
    __syncthreads();
    for (int d=tid; d<D; d+=blockDim.x){ float a=0; for(int p=0;p<seq;p++) a+=sc[p]*__half2float(v[((int64_t)p*H+head)*D+d]); out[((int64_t)token*H+head)*D+d]=__float2half(a/rsum); }
}

// tensor-core flash attention: one warp per 16-query tile.
__global__ void attn_tc(half* __restrict__ out, const half* __restrict__ q,
                        const half* __restrict__ k, const half* __restrict__ v,
                        int N, int H, float scale, int window) {
    const int head = blockIdx.x, qt = blockIdx.y, tid = threadIdx.x;  // 1 warp
    const int q0 = qt * MQ;

    __shared__ half  Qsh[MQ][D], Ksh[TK][D], Vsh[TK][D], Psh[MQ][TK];
    __shared__ float Ssh[MQ][TK], Osh[MQ][D], Tmp[MQ][TK], mrun[MQ], lrun[MQ];

    for (int idx = tid; idx < MQ*D; idx += 32) {
        const int i = idx / D, d = idx % D; const int qq = q0 + i;
        Qsh[i][d] = (qq < N) ? q[((int64_t)qq*H+head)*D + d] : __float2half(0.0f);
        Osh[i][d] = 0.0f;
    }
    for (int i = tid; i < MQ; i += 32) { mrun[i] = -INFINITY; lrun[i] = 0.0f; }
    __syncwarp();

    const int block_qmax = min(q0 + MQ - 1, N - 1);
    const int block_seq  = block_qmax + 1;

    for (int kt = 0; kt < block_seq; kt += TK) {
        for (int idx = tid; idx < TK*D; idx += 32) {
            const int j = idx / D, d = idx % D; const int kp = kt + j;
            Ksh[j][d] = (kp < block_seq) ? k[((int64_t)kp*H+head)*D + d] : __float2half(0.0f);
            Vsh[j][d] = (kp < block_seq) ? v[((int64_t)kp*H+head)*D + d] : __float2half(0.0f);
        }
        __syncwarp();

        // S = Q @ K^T  (K loaded col-major = K^T), accumulate over D chunks.
        wmma::fragment<wmma::accumulator, 16,16,16, float> cS;
        wmma::fill_fragment(cS, 0.0f);
        for (int dc = 0; dc < D; dc += 16) {
            wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> aQ;
            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> bK;
            wmma::load_matrix_sync(aQ, &Qsh[0][dc], D);
            wmma::load_matrix_sync(bK, &Ksh[0][dc], D);
            wmma::mma_sync(cS, aQ, bK, cS);
        }
        wmma::store_matrix_sync(&Ssh[0][0], cS, TK, wmma::mem_row_major);
        __syncwarp();

        // online softmax per query row (16 rows, one thread each).
        if (tid < MQ) {
            const int i = tid, qq = q0 + i;
            if (qq < N) {
                const int seq_q = qq + 1;
                float tmax = -INFINITY;
                for (int j = 0; j < TK; j++) {
                    const int kp = kt + j;
                    if (kp > qq || (window > 0 && kp < seq_q - window)) Ssh[i][j] = -INFINITY;
                    else Ssh[i][j] *= scale;
                    tmax = fmaxf(tmax, Ssh[i][j]);
                }
                if (tmax > -INFINITY) {
                    const float nm = fmaxf(mrun[i], tmax);
                    const float corr = (mrun[i] == -INFINITY) ? 0.0f : expf(mrun[i] - nm);
                    mrun[i] = nm; lrun[i] *= corr;
                    for (int d = 0; d < D; d++) Osh[i][d] *= corr;
                    float s = 0.0f;
                    for (int j = 0; j < TK; j++) { float e = expf(Ssh[i][j] - nm); Psh[i][j] = __float2half(e); s += e; }
                    lrun[i] += s;
                } else {
                    for (int j = 0; j < TK; j++) Psh[i][j] = __float2half(0.0f);
                }
            } else {
                for (int j = 0; j < TK; j++) Psh[i][j] = __float2half(0.0f);
            }
        }
        __syncwarp();

        // O += P @ V  (per D-chunk; small Tmp reused, added into Osh).
        for (int dc = 0; dc < D; dc += 16) {
            wmma::fragment<wmma::accumulator, 16,16,16, float> cO;
            wmma::fill_fragment(cO, 0.0f);
            wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> aP;
            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> bV;
            wmma::load_matrix_sync(aP, &Psh[0][0], TK);
            wmma::load_matrix_sync(bV, &Vsh[0][dc], D);
            wmma::mma_sync(cO, aP, bV, cO);
            wmma::store_matrix_sync(&Tmp[0][0], cO, TK, wmma::mem_row_major);
            __syncwarp();
            for (int idx = tid; idx < MQ*16; idx += 32) Osh[idx/16][dc + (idx%16)] += Tmp[idx/16][idx%16];
            __syncwarp();
        }
    }

    for (int idx = tid; idx < MQ*D; idx += 32) {
        const int i = idx / D, d = idx % D; const int qq = q0 + i;
        if (qq < N) { const float inv = (lrun[i] > 0.0f) ? 1.0f/lrun[i] : 0.0f;
            out[((int64_t)qq*H+head)*D + d] = __float2half(Osh[i][d] * inv); }
    }
}

static void fill(std::vector<half>& x, uint32_t s, float r=1.0f){ std::mt19937 g(s); std::uniform_real_distribution<float> d(-r,r); for(auto&v:x)v=__float2half(d(g)); }
static float cmp(const std::vector<half>& a, const std::vector<half>& b, int N, int H){
    float m=0; size_t am=0; for(size_t i=0;i<a.size();i++){ float e=std::fabs(__half2float(a[i])-__half2float(b[i])); if(e>m){m=e;am=i;} }
    if(m>0.01f){ int d=am%D,h=(am/D)%H,t=am/(D*H); printf("    worst tok=%d head=%d d=%d new=%.4f ref=%.4f\n",t,h,d,__half2float(a[am]),__half2float(b[am])); }
    return m;
}
static void run(int N,int H,int window,const char* lbl){
    const float scale=1.0f/std::sqrt((float)D);
    std::vector<half> q(N*H*D),k(N*H*D),v(N*H*D),oref(N*H*D),onew(N*H*D);
    fill(q,1);fill(k,2);fill(v,3);
    half*dq,*dk,*dv,*doo; size_t sz=(size_t)N*H*D*sizeof(half);
    CK(cudaMalloc(&dq,sz));CK(cudaMalloc(&dk,sz));CK(cudaMalloc(&dv,sz));CK(cudaMalloc(&doo,sz));
    CK(cudaMemcpy(dq,q.data(),sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(dk,k.data(),sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,v.data(),sz,cudaMemcpyHostToDevice));
    dim3 gN(H,N); int sm=(4096+D)*sizeof(float);
    attn_naive<<<gN,128,sm>>>(doo,dq,dk,dv,N,H,scale,window); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(oref.data(),doo,sz,cudaMemcpyDeviceToHost));
    dim3 gT(H,(N+MQ-1)/MQ);
    attn_tc<<<gT,32>>>(doo,dq,dk,dv,N,H,scale,window); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(onew.data(),doo,sz,cudaMemcpyDeviceToHost));
    float m=cmp(onew,oref,N,H);
    cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
    for(int i=0;i<3;i++)attn_tc<<<gT,32>>>(doo,dq,dk,dv,N,H,scale,window); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);for(int i=0;i<10;i++)attn_tc<<<gT,32>>>(doo,dq,dk,dv,N,H,scale,window);cudaEventRecord(e1);cudaEventSynchronize(e1);
    float ms=0;cudaEventElapsedTime(&ms,e0,e1);ms/=10;
    cudaEventRecord(e0);for(int i=0;i<10;i++)attn_naive<<<gN,128,sm>>>(doo,dq,dk,dv,N,H,scale,window);cudaEventRecord(e1);cudaEventSynchronize(e1);
    float mn=0;cudaEventElapsedTime(&mn,e0,e1);mn/=10;
    printf("  %-14s N=%4d win=%-4d max_abs=%.4f  tc=%.2fms naive=%.2fms  speedup=%.1fx\n",lbl,N,window,m,ms,mn,mn/ms);
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(doo);cudaEventDestroy(e0);cudaEventDestroy(e1);
}
int main(){
    CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));printf("Device: %s SM %d.%d\n",p.name,p.major,p.minor);
    printf("-- tensor-core flash attention vs naive (D=%d) --\n",D);
    run(512,8,0,"causal");
    run(512,8,256,"sliding256");
    run(2048,8,0,"causal-2k");
    run(4096,8,512,"sliding512-4k");
    return 0;
}
