// test_flash_tc_reg.cu — register-accumulator tensor-core flash attention.
//
// The long-context prefill wall is the global-layer attention (D=512, full
// O(N^2) compute). #116 validated wmma TC attention but was occupancy-bound
// because it kept the O accumulator in SHARED memory. This is the high-occupancy
// version llama's fattn-mma uses: O accumulator lives in MMA REGISTER fragments,
// and the per-query online-softmax rescale is applied directly to those register
// fragments. Raw mma.sync (wmma frags are opaque). m16n8k16 f32 accumulator
// layout: thread `lane` holds rows {gid, gid+8} x cols {2t, 2t+1}, gid=lane>>2,
// t=lane&3 — so the query row is known per-thread -> per-row corr scales it.
//
//   S = Q @ K^T   (m16n8k16, 2 key-subtiles x D/16 contractions)
//   online softmax per query row (group-reduce over the 4 lanes sharing a row)
//   rescale O register fragments by exp(m_old - m_new)
//   O += P @ V    (m16n8k16, head-dim-subtiles, P staged in shared)
//
// D=512: NW=2 warps/block split the head-dim of O (256 each = same 128-float reg
// budget); both warps compute S+softmax redundantly. Dynamic shared (>48KB).
// Sliding window: skip K-tiles entirely below the window (no masked-out MMAs).
// f16 K/V here (algorithm validation); int8-KV + cp.async + engine port next.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} }while(0)

constexpr int MQ = 16;   // queries per tile
constexpr int KT = 16;   // keys per tile

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ uint32_t pack2(half lo, half hi){
    return (uint32_t)*reinterpret_cast<uint16_t*>(&lo) |
           ((uint32_t)*reinterpret_cast<uint16_t*>(&hi) << 16);
}
__device__ __forceinline__ float gmax4(float v){
    v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,1));
    v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,2)); return v;
}
__device__ __forceinline__ float gsum4(float v){
    v+=__shfl_xor_sync(0xffffffffu,v,1);
    v+=__shfl_xor_sync(0xffffffffu,v,2); return v;
}

__global__ void attn_naive(half* __restrict__ out, const half* __restrict__ q,
                           const half* __restrict__ k, const half* __restrict__ v,
                           int N, int H, int D, float scale, int window) {
    const int head = blockIdx.x, token = blockIdx.y, tid = threadIdx.x;
    const int seq = token + 1;
    extern __shared__ float sm[];
    float* sc = sm; float* so = sm + 8192;
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

// async-load one K/V tile (kt) into shared buffers via cp.async (16B chunks).
// Out-of-range rows zero-filled synchronously (V garbage * P=0 would NaN).
template<int D>
__device__ __forceinline__ void load_tile_async(half* Kb, half* Vb,
        const half* k, const half* v, int kt, int block_seq,
        int head, int H, int lane, int bdim) {
    const int nchunk = (KT*D)/8;
    for (int chunk = lane; chunk < nchunk; chunk += bdim) {
        const int hidx = chunk*8, r = hidx/D, c = hidx%D, kp = kt + r;
        if (kp < block_seq) {
            __pipeline_memcpy_async(&Kb[hidx], &k[((int64_t)kp*H+head)*D + c], 16);
            __pipeline_memcpy_async(&Vb[hidx], &v[((int64_t)kp*H+head)*D + c], 16);
        } else {
            #pragma unroll
            for (int z = 0; z < 8; z++) { Kb[hidx+z]=__float2half(0.f); Vb[hidx+z]=__float2half(0.f); }
        }
    }
}

// register-accumulator TC flash attention with cp.async double-buffered K/V.
// NW = D/256 warps/block; each warp owns DPW=256 head-dim of the O accumulator.
template<int D>
__global__ void attn_tc_reg(half* __restrict__ out, const half* __restrict__ q,
                            const half* __restrict__ k, const half* __restrict__ v,
                            int N, int H, float scale, int window) {
    constexpr int NW  = D / 256;       // warps per block (1 or 2)
    constexpr int DPW = 256;           // head-dim of O per warp
    const int head = blockIdx.x, qt = blockIdx.y;
    const int lane = threadIdx.x, warp = lane >> 5, wl = lane & 31;
    const int gid = wl >> 2, t = wl & 3;
    const int hd0 = warp * DPW;
    const int q0  = qt * MQ;

    extern __shared__ char smem[];
    half*  Qsh   = (half*)smem;                 // MQ*D
    half*  Kbuf  = Qsh + MQ*D;                   // 2*KT*D  (double-buffered)
    half*  Vbuf  = Kbuf + 2*KT*D;                // 2*KT*D
    half*  Psh   = Vbuf + 2*KT*D;                // NW*MQ*KT
    float* mrow  = (float*)(Psh + NW*MQ*KT);     // NW*MQ
    float* lrow  = mrow + NW*MQ;                 // NW*MQ
    float* corr  = lrow + NW*MQ;                 // NW*MQ
    half*  Kbufs[2] = { Kbuf, Kbuf + KT*D };
    half*  Vbufs[2] = { Vbuf, Vbuf + KT*D };

    for (int idx = lane; idx < MQ*D; idx += blockDim.x) {
        const int r = idx / D, c = idx % D, qq = q0 + r;
        Qsh[idx] = (qq < N) ? q[((int64_t)qq*H+head)*D + c] : __float2half(0.f);
    }
    if (warp == 0) for (int r = wl; r < MQ; r += 32) { mrow[r]=-INFINITY; lrow[r]=0.f; }
    if (NW > 1 && warp == 1) for (int r = wl; r < MQ; r += 32) { mrow[MQ+r]=-INFINITY; lrow[MQ+r]=0.f; }

    float oacc[DPW/8][4];
    #pragma unroll
    for (int j = 0; j < DPW/8; j++){ oacc[j][0]=oacc[j][1]=oacc[j][2]=oacc[j][3]=0.f; }
    __syncthreads();

    const int block_seq = min(q0 + MQ - 1, N - 1) + 1;
    int kt_start = 0;
    if (window > 0) { kt_start = q0 + 1 - window; if (kt_start < 0) kt_start = 0; kt_start = (kt_start/KT)*KT; }

    float* mw = mrow + warp*MQ; float* lw = lrow + warp*MQ; float* cw = corr + warp*MQ;
    half*  Pw = Psh  + warp*MQ*KT;

    int buf = 0;
    load_tile_async<D>(Kbufs[0], Vbufs[0], k, v, kt_start, block_seq, head, H, lane, blockDim.x);
    __pipeline_commit();

    for (int kt = kt_start; kt < block_seq; kt += KT) {
        const int ktn = kt + KT;
        if (ktn < block_seq) {
            load_tile_async<D>(Kbufs[buf^1], Vbufs[buf^1], k, v, ktn, block_seq, head, H, lane, blockDim.x);
            __pipeline_commit();
            __pipeline_wait_prior(1);   // keep the next-tile prefetch in flight, wait for current
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        half* Ksh = Kbufs[buf];
        half* Vsh = Vbufs[buf];

        // S = Q @ K^T  (each warp computes full S over D, redundant for NW>1)
        float sacc[2][4] = {{0,0,0,0},{0,0,0,0}};
        #pragma unroll
        for (int kc = 0; kc < D; kc += 16) {
            uint32_t a[4] = {
                *(uint32_t*)&Qsh[(gid  )*D + kc + 2*t],
                *(uint32_t*)&Qsh[(gid+8)*D + kc + 2*t],
                *(uint32_t*)&Qsh[(gid  )*D + kc + 2*t + 8],
                *(uint32_t*)&Qsh[(gid+8)*D + kc + 2*t + 8] };
            #pragma unroll
            for (int ns = 0; ns < 2; ns++) {
                uint32_t b[2] = {
                    *(uint32_t*)&Ksh[(ns*8+gid)*D + kc + 2*t],
                    *(uint32_t*)&Ksh[(ns*8+gid)*D + kc + 2*t + 8] };
                mma16816(sacc[ns], a, b);
            }
        }

        const int qa = q0 + gid, qb = q0 + gid + 8;
        float mhi_a = -INFINITY, mhi_b = -INFINITY;
        #pragma unroll
        for (int ns = 0; ns < 2; ns++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = kt + ns*8 + 2*t + e;
                float sa = sacc[ns][e];
                bool oka = (key <= qa) && !(window>0 && key < qa+1-window) && (qa < N);
                sa = oka ? sa*scale : -INFINITY; sacc[ns][e] = sa; mhi_a = fmaxf(mhi_a, sa);
                float sb = sacc[ns][2+e];
                bool okb = (key <= qb) && !(window>0 && key < qb+1-window) && (qb < N);
                sb = okb ? sb*scale : -INFINITY; sacc[ns][2+e] = sb; mhi_b = fmaxf(mhi_b, sb);
            }
        }
        mhi_a = gmax4(mhi_a); mhi_b = gmax4(mhi_b);

        if (t == 0) {
            float om=mw[gid],   nm =fmaxf(om, mhi_a);
            cw[gid]   = (om ==-INFINITY)?0.f:__expf(om - nm);  mw[gid]   = nm;
            float omb=mw[gid+8],nmb=fmaxf(omb,mhi_b);
            cw[gid+8] = (omb==-INFINITY)?0.f:__expf(omb- nmb); mw[gid+8] = nmb;
        }
        __syncwarp();

        const float ca = cw[gid], cb = cw[gid+8];
        #pragma unroll
        for (int j = 0; j < DPW/8; j++){ oacc[j][0]*=ca; oacc[j][1]*=ca; oacc[j][2]*=cb; oacc[j][3]*=cb; }

        const float nma = mw[gid], nmb = mw[gid+8];
        float pa = 0.f, pb = 0.f;
        #pragma unroll
        for (int ns = 0; ns < 2; ns++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = ns*8 + 2*t + e;
                float ea = (sacc[ns][e]  ==-INFINITY)?0.f:__expf(sacc[ns][e]  - nma);
                float eb = (sacc[ns][2+e]==-INFINITY)?0.f:__expf(sacc[ns][2+e]- nmb);
                Pw[(gid  )*KT + key] = __float2half(ea); pa += ea;
                Pw[(gid+8)*KT + key] = __float2half(eb); pb += eb;
            }
        }
        pa = gsum4(pa); pb = gsum4(pb);
        if (t == 0) { lw[gid] = lw[gid]*cw[gid] + pa; lw[gid+8] = lw[gid+8]*cw[gid+8] + pb; }
        __syncwarp();

        uint32_t pf[4] = {
            *(uint32_t*)&Pw[(gid  )*KT + 2*t],
            *(uint32_t*)&Pw[(gid+8)*KT + 2*t],
            *(uint32_t*)&Pw[(gid  )*KT + 2*t + 8],
            *(uint32_t*)&Pw[(gid+8)*KT + 2*t + 8] };
        #pragma unroll
        for (int j = 0; j < DPW/8; j++) {
            const int hd = hd0 + j*8 + gid;
            uint32_t bv[2] = {
                pack2(Vsh[(2*t  )*D + hd], Vsh[(2*t+1)*D + hd]),
                pack2(Vsh[(2*t+8)*D + hd], Vsh[(2*t+9)*D + hd]) };
            mma16816(oacc[j], pf, bv);
        }
        __syncthreads();
        buf ^= 1;
    }

    const float la = lw[gid]   > 0.f ? 1.f/lw[gid]   : 0.f;
    const float lb = lw[gid+8] > 0.f ? 1.f/lw[gid+8] : 0.f;
    #pragma unroll
    for (int j = 0; j < DPW/8; j++) {
        const int qa = q0 + gid, qb = q0 + gid + 8, hd = hd0 + j*8 + 2*t;
        if (qa < N) {
            out[((int64_t)qa*H+head)*D + hd  ] = __float2half(oacc[j][0]*la);
            out[((int64_t)qa*H+head)*D + hd+1] = __float2half(oacc[j][1]*la);
        }
        if (qb < N) {
            out[((int64_t)qb*H+head)*D + hd  ] = __float2half(oacc[j][2]*lb);
            out[((int64_t)qb*H+head)*D + hd+1] = __float2half(oacc[j][3]*lb);
        }
    }
}

static void fill(std::vector<half>& x, uint32_t s, float r=1.0f){ std::mt19937 g(s); std::uniform_real_distribution<float> d(-r,r); for(auto&v:x)v=__float2half(d(g)); }
static float cmp(const std::vector<half>& a, const std::vector<half>& b, int D, int H){
    float m=0; size_t am=0; for(size_t i=0;i<a.size();i++){ float e=std::fabs(__half2float(a[i])-__half2float(b[i])); if(e>m){m=e;am=i;} }
    if(m>0.02f){ int d=am%D,h=(am/D)%H,tk=am/(D*H); printf("    worst tok=%d head=%d d=%d new=%.4f ref=%.4f\n",tk,h,d,__half2float(a[am]),__half2float(b[am])); }
    return m;
}
template<int D>
static void run(int N,int H,int window,const char* lbl){
    constexpr int NW = D/256;
    const float scale=1.0f/std::sqrt((float)D);
    std::vector<half> q(N*H*D),k(N*H*D),v(N*H*D),oref(N*H*D),onew(N*H*D);
    fill(q,1);fill(k,2);fill(v,3);
    half*dq,*dk,*dv,*doo; size_t sz=(size_t)N*H*D*sizeof(half);
    CK(cudaMalloc(&dq,sz));CK(cudaMalloc(&dk,sz));CK(cudaMalloc(&dv,sz));CK(cudaMalloc(&doo,sz));
    CK(cudaMemcpy(dq,q.data(),sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(dk,k.data(),sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,v.data(),sz,cudaMemcpyHostToDevice));
    dim3 gN(H,N); int smnaive=(8192+D)*sizeof(float);
    attn_naive<<<gN,128,smnaive>>>(doo,dq,dk,dv,N,H,D,scale,window); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(oref.data(),doo,sz,cudaMemcpyDeviceToHost));
    size_t smbytes = (size_t)(MQ*D + 4*KT*D + NW*MQ*KT)*sizeof(half) + (size_t)(3*NW*MQ)*sizeof(float);
    CK(cudaFuncSetAttribute(attn_tc_reg<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smbytes));
    dim3 gT(H,(N+MQ-1)/MQ);
    attn_tc_reg<D><<<gT,NW*32,smbytes>>>(doo,dq,dk,dv,N,H,scale,window); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(onew.data(),doo,sz,cudaMemcpyDeviceToHost));
    float m=cmp(onew,oref,D,H);
    cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
    attn_tc_reg<D><<<gT,NW*32,smbytes>>>(doo,dq,dk,dv,N,H,scale,window); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);for(int i=0;i<5;i++)attn_tc_reg<D><<<gT,NW*32,smbytes>>>(doo,dq,dk,dv,N,H,scale,window);cudaEventRecord(e1);cudaEventSynchronize(e1);
    float ms=0;cudaEventElapsedTime(&ms,e0,e1);ms/=5;
    cudaEventRecord(e0);for(int i=0;i<2;i++)attn_naive<<<gN,128,smnaive>>>(doo,dq,dk,dv,N,H,D,scale,window);cudaEventRecord(e1);cudaEventSynchronize(e1);
    float mn=0;cudaEventElapsedTime(&mn,e0,e1);mn/=2;
    printf("  %-16s D=%d N=%4d win=%-4d max_abs=%.4f  tc=%.3fms naive=%.3fms  speedup=%.1fx\n",lbl,D,N,window,m,ms,mn,mn/ms);
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(doo);cudaEventDestroy(e0);cudaEventDestroy(e1);
}
int main(){
    CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));printf("Device: %s SM %d.%d\n",p.name,p.major,p.minor);
    printf("-- register-accumulator TC flash attention vs naive --\n");
    printf("[D=256 sliding layers]\n");
    run<256>(512,8,0,"causal");
    run<256>(512,8,256,"sliding256");
    run<256>(2048,8,512,"sliding512-2k");
    run<256>(4096,8,512,"sliding512-4k");
    printf("[D=512 global layers — the long-ctx wall]\n");
    run<512>(512,8,0,"causal");
    run<512>(2048,8,0,"causal-2k");
    run<512>(4096,8,0,"causal-4k");
    printf("[small / non-multiple-of-16 N — reproduce the Paris(N=6) case]\n");
    run<256>(6,8,0,"d256-N6");
    run<512>(6,8,0,"d512-N6");
    run<512>(20,8,0,"d512-N20");
    run<512>(6,8,512,"d512-N6-sw");
    return 0;
}
