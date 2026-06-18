// test_flash_tiled.cu — tiled flash-attention prefill for genie.
//
// The current genie prefill attention is one CUDA block per (head, query) and
// re-reads the ENTIRE K/V cache for every query -> O(N^2) memory traffic, which
// is what makes long-context prefill cripplingly slow on the 8-SM Orin.
//
// This kernel processes a TILE of queries per block (one warp per query) and
// loads each K/V tile into shared memory ONCE, reused across all queries in the
// block -> K/V traffic cut by WARPS x. Online-softmax + causal/sliding-window
// mask preserved. f16 K/V here (correctness of the algorithm); int8-KV + the
// engine port come after this validates.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} }while(0)

constexpr int D     = 256;   // head_dim (Gemma sliding); test this first
constexpr int WARPS = 8;     // queries per block (K/V reuse factor)
constexpr int TK    = 32;    // KV tile length (f16 tiles: 2*TK*D*2 B must fit 48KB smem)
constexpr int DPL   = D / 32; // head_dim elements per lane

// ── naive reference (mirrors the current genie kernel: 1 block per query) ──
__global__ void attn_naive(half* __restrict__ out, const half* __restrict__ q,
                           const half* __restrict__ k, const half* __restrict__ v,
                           int N, int H, float scale, int window) {
    const int head = blockIdx.x, token = blockIdx.y, tid = threadIdx.x;
    const int seq = token + 1;
    extern __shared__ float sm[];
    float* sscore = sm;            // [seq] (bounded by smem; test uses small seq)
    float* sout   = sm + 4096;
    __shared__ float rmax, rsum;
    for (int d = tid; d < D; d += blockDim.x) sout[d] = 0.0f;
    if (tid == 0) { rmax = -INFINITY; rsum = 0.0f; }
    __syncthreads();
    const half* ql = q + ((int64_t)token * H + head) * D;
    for (int p = tid; p < seq; p += blockDim.x) {
        float s;
        if (window > 0 && p < seq - window) s = -INFINITY;
        else { float dot = 0.0f; for (int d = 0; d < D; d++)
                 dot += __half2float(ql[d]) * __half2float(k[((int64_t)p*H+head)*D+d]);
               s = dot * scale; }
        sscore[p] = s;
    }
    __syncthreads();
    if (tid == 0) {
        float m = -INFINITY; for (int p=0;p<seq;p++) m = fmaxf(m, sscore[p]);
        rmax = m; float su = 0; for (int p=0;p<seq;p++){ float e=expf(sscore[p]-m); sscore[p]=e; su+=e; } rsum=su;
    }
    __syncthreads();
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0; for (int p=0;p<seq;p++) acc += sscore[p]*__half2float(v[((int64_t)p*H+head)*D+d]);
        out[((int64_t)token*H+head)*D+d] = __float2half(acc / rsum);
    }
}

// ── tiled flash attention: one warp per query, shared K/V tiles ──
__global__ void attn_tiled(half* __restrict__ out, const half* __restrict__ q,
                           const half* __restrict__ k, const half* __restrict__ v,
                           int N, int H, float scale, int window) {
    const int head  = blockIdx.x;
    const int qtile = blockIdx.y;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int qpos  = qtile * WARPS + warp;

    __shared__ half  Ksh[TK][D];
    __shared__ half  Vsh[TK][D];
    __shared__ float Ssh[WARPS][TK];

    float qreg[DPL], oreg[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; i++) {
        qreg[i] = (qpos < N) ? __half2float(q[((int64_t)qpos*H+head)*D + lane + i*32]) : 0.0f;
        oreg[i] = 0.0f;
    }
    float run_max = -INFINITY, run_sum = 0.0f;
    const int seq_q = qpos + 1;                                       // this warp's causal end
    const int block_seq = min(qtile * WARPS + WARPS - 1, N - 1) + 1;  // uniform tile range (all warps)

    for (int kt = 0; kt < block_seq; kt += TK) {
        const int tklen = min(TK, block_seq - kt);
        // cooperative load of the K/V tile (all WARPS*32 threads).
        for (int idx = threadIdx.x; idx < TK * D; idx += blockDim.x) {
            const int kk = idx / D, dd = idx % D;
            const int kvpos = kt + kk;
            Ksh[kk][dd] = (kk < tklen) ? k[((int64_t)kvpos*H+head)*D + dd] : __float2half(0.0f);
            Vsh[kk][dd] = (kk < tklen) ? v[((int64_t)kvpos*H+head)*D + dd] : __float2half(0.0f);
        }
        __syncthreads();

        // each warp computes its query's scores vs the K tile.
        if (qpos < N) {
            for (int kk = 0; kk < tklen; kk++) {
                const int kvpos = kt + kk;
                float s;
                if (kvpos > qpos || (window > 0 && kvpos < seq_q - window)) {
                    s = -INFINITY;   // causal + sliding-window mask (per query)
                } else {
                    float p = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < DPL; i++) p += qreg[i] * __half2float(Ksh[kk][lane + i*32]);
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
                    s = p * scale;
                }
                if (lane == 0) Ssh[warp][kk] = s;
            }
            __syncwarp();
            // online-softmax over this tile (warp-cooperative over tklen<=64).
            float lmax = -INFINITY;
            for (int kk = lane; kk < tklen; kk += 32) lmax = fmaxf(lmax, Ssh[warp][kk]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
            if (lmax > -INFINITY) {   // skip fully-masked tiles (else exp(-inf+inf)=NaN)
                const float new_max = fmaxf(run_max, lmax);
                const float corr = (run_max == -INFINITY) ? 0.0f : expf(run_max - new_max);
                run_max = new_max;
                run_sum *= corr;
                #pragma unroll
                for (int i = 0; i < DPL; i++) oreg[i] *= corr;
                float lsum = 0.0f;
                for (int kk = lane; kk < tklen; kk += 32) {
                    float e = expf(Ssh[warp][kk] - run_max);
                    Ssh[warp][kk] = e; lsum += e;
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
                run_sum += lsum;
                __syncwarp();
                // O += P @ V  (each lane owns DPL of head_dim).
                #pragma unroll
                for (int i = 0; i < DPL; i++) {
                    const int d = lane + i*32;
                    float acc = 0.0f;
                    for (int kk = 0; kk < tklen; kk++) acc += Ssh[warp][kk] * __half2float(Vsh[kk][d]);
                    oreg[i] += acc;
                }
            }
        }
        __syncthreads();
    }
    if (qpos < N) {
        const float inv = (run_sum > 0.0f) ? 1.0f / run_sum : 0.0f;
        #pragma unroll
        for (int i = 0; i < DPL; i++)
            out[((int64_t)qpos*H+head)*D + lane + i*32] = __float2half(oreg[i] * inv);
    }
}

static void fill(std::vector<half>& x, uint32_t s, float r=1.0f){ std::mt19937 g(s); std::uniform_real_distribution<float> d(-r,r); for(auto&v:x)v=__float2half(d(g)); }

static float cmp(const std::vector<half>& a, const std::vector<half>& b, int H=0){
    float m=0; size_t am=0;
    for(size_t i=0;i<a.size();i++){ float e=std::fabs(__half2float(a[i])-__half2float(b[i])); if(e>m){m=e;am=i;} }
    if(m>0.01f && H){ int d=am%D; int hf=(am/D)%H; int tok=am/(D*H);
        printf("    worst @ tok=%d head=%d d=%d : new=%.4f ref=%.4f\n", tok, hf, d, __half2float(a[am]), __half2float(b[am])); }
    return m;
}

static void run(int N, int H, int window, const char* lbl){
    const float scale = 1.0f/std::sqrt((float)D);
    std::vector<half> q(N*H*D), k(N*H*D), v(N*H*D), o_ref(N*H*D), o_new(N*H*D);
    fill(q,1,1.0f); fill(k,2,1.0f); fill(v,3,1.0f);
    half *dq,*dk,*dv,*doo; size_t sz=(size_t)N*H*D*sizeof(half);
    CK(cudaMalloc(&dq,sz));CK(cudaMalloc(&dk,sz));CK(cudaMalloc(&dv,sz));CK(cudaMalloc(&doo,sz));
    CK(cudaMemcpy(dq,q.data(),sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(dk,k.data(),sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,v.data(),sz,cudaMemcpyHostToDevice));
    // reference (needs seq<=4096 for the smem score buffer)
    dim3 gN(H,N); int smem=(4096+D)*sizeof(float);
    attn_naive<<<gN,128,smem>>>(doo,dq,dk,dv,N,H,scale,window); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(o_ref.data(),doo,sz,cudaMemcpyDeviceToHost));
    // tiled
    dim3 gT(H,(N+WARPS-1)/WARPS);
    attn_tiled<<<gT,WARPS*32>>>(doo,dq,dk,dv,N,H,scale,window); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(o_new.data(),doo,sz,cudaMemcpyDeviceToHost));
    float m=cmp(o_new,o_ref,H);
    // timing (tiled) at this N
    cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
    for(int i=0;i<3;i++)attn_tiled<<<gT,WARPS*32>>>(doo,dq,dk,dv,N,H,scale,window); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int i=0;i<10;i++)attn_tiled<<<gT,WARPS*32>>>(doo,dq,dk,dv,N,H,scale,window); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0;cudaEventElapsedTime(&ms,e0,e1);ms/=10;
    // naive timing
    cudaEventRecord(e0); for(int i=0;i<10;i++)attn_naive<<<gN,128,smem>>>(doo,dq,dk,dv,N,H,scale,window); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float msn=0;cudaEventElapsedTime(&msn,e0,e1);msn/=10;
    printf("  %-14s N=%4d H=%d win=%-4d : max_abs=%.4f  tiled=%.2fms  naive=%.2fms  speedup=%.1fx\n",
           lbl,N,H,window,m,ms,msn,msn/ms);
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(doo);cudaEventDestroy(e0);cudaEventDestroy(e1);
}

int main(){
    CK(cudaSetDevice(0)); cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0)); printf("Device: %s SM %d.%d\n",p.name,p.major,p.minor);
    printf("-- tiled flash attention vs naive (D=%d, %d warps/block) --\n", D, WARPS);
    run(512,  8, 0,   "causal");
    run(512,  8, 256, "sliding256");
    run(2048, 8, 0,   "causal-2k");
    run(4096, 8, 512, "sliding512-4k");
    return 0;
}
