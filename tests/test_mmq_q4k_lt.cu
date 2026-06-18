// test_mmq_q4k_lt.cu — llama.cpp-style int8 Q4_K MMQ for Orin SM 8.7.
//
// nsys + mmq.cuh review showed llama's prefill uses the SAME int8 MMQ as genie
// but ~4-5x more TC-efficient (~26% vs genie's 6%) via: (1) __launch_bounds__(.,2)
// forcing 2 blocks/SM, (2) K pipelined in 32-wide slices so only a small operand
// slice sits in shared (low shared -> high occupancy WITH a big tile), (3) 8
// warps, (4) weights unpacked to a shared int8 tile ONCE per slice (not per-MMA).
// genie's warp-tile got amortization OR occupancy, never both; this gets both.
//
// Y[N x M] = X(q8_1)[N x K] . W(Q4_K)[M x K].

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} }while(0)

#define QK_K 256
struct __align__(2) block_q4_K { uint16_t d_raw, dmin_raw; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q8_1 { uint16_t d_raw, s_raw; int8_t qs[32]; };
__host__ __device__ inline float rawh2f(uint16_t h){ __half x; memcpy(&x,&h,2); return __half2float(x); }
__host__ __device__ inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m){
    if (j<4){ d=q[j]&63; m=q[j+4]&63; } else { d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); m=(q[j+4]>>4)|((q[j]>>6)<<4); }
}
static void ref_gemm(std::vector<float>& Y,const std::vector<block_q4_K>& W,const std::vector<block_q8_1>& X,int M,int N,int K){
    const int nbk=K/QK_K, nsb=K/32;
    for(int n=0;n<N;n++)for(int m=0;m<M;m++){ float acc=0;
        for(int b=0;b<nbk;b++){ const block_q4_K& w=W[(size_t)m*nbk+b]; float dall=rawh2f(w.d_raw),dmin=rawh2f(w.dmin_raw);
            for(int sb=0;sb<8;sb++){ uint8_t sc,mn; get_scale_min_k4(sb,w.scales,sc,mn); float dw=dall*sc,dmw=dmin*mn;
                int il=sb>>1,parity=sb&1; const block_q8_1& x=X[(size_t)n*nsb+b*8+sb]; float dx=rawh2f(x.d_raw),sx=rawh2f(x.s_raw);
                int dot=0; for(int c=0;c<32;c++){ int qw=parity?(w.qs[32*il+c]>>4):(w.qs[32*il+c]&0xF); dot+=qw*(int)x.qs[c]; }
                acc+=dw*dx*dot-dmw*sx; } }
        Y[(size_t)n*M+m]=acc; }
}

// ---- llama-style kernel ----
constexpr int LT_BM=64, LT_BN=64, LT_NW=8;          // 8 warps, 64x64 tile
constexpr int LT_WM=2, LT_WN=4;                      // warp grid 2x4
constexpr int LT_WARP_M=LT_BM/LT_WM, LT_WARP_N=LT_BN/LT_WN;  // 32 x 16
constexpr int LT_RF=LT_WARP_M/16, LT_TF=LT_WARP_N/8;          // 2 x 2 = 4 MMAs/warp/slice

__device__ __forceinline__ void mma(int c[4], const int a[4], const int b[2]){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__launch_bounds__(LT_NW*32, 2)
__global__ void gemm_lt(half* __restrict__ y, const block_q4_K* __restrict__ W,
                        const block_q8_1* __restrict__ XQ, int M, int N, int K){
    const int row_base=blockIdx.y*LT_BM, tok_base=blockIdx.x*LT_BN;
    const int wid=threadIdx.x>>5, lane=threadIdx.x&31;
    const int wm=wid%LT_WM, wn=wid/LT_WM, gid=lane>>2, t=lane&3;
    const int nbk=K/QK_K, nsb=K/32;
    const int wrow0=wm*LT_WARP_M, wtok0=wn*LT_WARP_N;

    // 64-K chunk = qs[32*il] of one block -> sub-block gsb0=2c (low nibbles) +
    // gsb1=2c+1 (high). Read each qs int ONCE, unpack BOTH nibbles to shared.
    __shared__ int8_t Wsh[2][LT_BM][64];
    __shared__ int8_t Xsh[2][LT_BN][64];
    __shared__ float  Wd[2][LT_BM][2], Wm[2][LT_BM][2], Xd[2][LT_BN][2], Xs[2][LT_BN][2];
    const int nchunk=nsb/2;
    auto load=[&](int buf,int c){
        const int blk=c>>2, il=c&3, gsb0=2*c, gsb1=2*c+1;
        for(int idx=threadIdx.x; idx<LT_BM*8; idx+=LT_NW*32){ int r=idx>>3, kc=(idx&7)<<2, g=row_base+r;
            uint32_t w4=(g<M)?*(const uint32_t*)&W[(int64_t)g*nbk+blk].qs[32*il+kc]:0u;
            *(int*)&Wsh[buf][r][kc]=(int)(w4&0x0F0F0F0Fu); *(int*)&Wsh[buf][r][32+kc]=(int)((w4>>4)&0x0F0F0F0Fu); }
        for(int r=threadIdx.x; r<LT_BM; r+=LT_NW*32){ int g=row_base+r; float d0=0,m0=0,d1=0,m1=0;
            if(g<M){ const block_q4_K& b=W[(int64_t)g*nbk+blk]; float dl=rawh2f(b.d_raw),dn=rawh2f(b.dmin_raw); uint8_t s,m;
                get_scale_min_k4(gsb0&7,b.scales,s,m); d0=dl*s; m0=dn*m; get_scale_min_k4(gsb1&7,b.scales,s,m); d1=dl*s; m1=dn*m; }
            Wd[buf][r][0]=d0; Wm[buf][r][0]=m0; Wd[buf][r][1]=d1; Wm[buf][r][1]=m1; }
        for(int idx=threadIdx.x; idx<LT_BN*8; idx+=LT_NW*32){ int tk=idx>>3, kc=(idx&7)<<2, g=tok_base+tk;
            *(int*)&Xsh[buf][tk][kc]=(g<N)?*(const int*)&XQ[(int64_t)g*nsb+gsb0].qs[kc]:0;
            *(int*)&Xsh[buf][tk][32+kc]=(g<N)?*(const int*)&XQ[(int64_t)g*nsb+gsb1].qs[kc]:0; }
        for(int tk=threadIdx.x; tk<LT_BN; tk+=LT_NW*32){ int g=tok_base+tk; float d0=0,s0=0,d1=0,s1=0;
            if(g<N){ d0=rawh2f(XQ[(int64_t)g*nsb+gsb0].d_raw); s0=rawh2f(XQ[(int64_t)g*nsb+gsb0].s_raw);
                     d1=rawh2f(XQ[(int64_t)g*nsb+gsb1].d_raw); s1=rawh2f(XQ[(int64_t)g*nsb+gsb1].s_raw); }
            Xd[buf][tk][0]=d0; Xs[buf][tk][0]=s0; Xd[buf][tk][1]=d1; Xs[buf][tk][1]=s1; }
    };

    float acc[LT_RF][LT_TF][4];
    #pragma unroll
    for(int i=0;i<LT_RF;i++)for(int j=0;j<LT_TF;j++){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.f;}

    int buf=0; load(0,0); __syncthreads();
    for(int c=0;c<nchunk;c++){
        if(c+1<nchunk) load(buf^1,c+1);
        #pragma unroll
        for(int s=0;s<2;s++){ const int ko=s*32;
            #pragma unroll
            for(int rf=0;rf<LT_RF;rf++){ const int r0=wrow0+rf*16;
                int a[4]={ *(int*)&Wsh[buf][r0+gid][ko+4*t], *(int*)&Wsh[buf][r0+gid+8][ko+4*t],
                           *(int*)&Wsh[buf][r0+gid][ko+16+4*t], *(int*)&Wsh[buf][r0+gid+8][ko+16+4*t] };
                const float dwA=Wd[buf][r0+gid][s],dmA=Wm[buf][r0+gid][s],dwB=Wd[buf][r0+gid+8][s],dmB=Wm[buf][r0+gid+8][s];
                #pragma unroll
                for(int tf=0;tf<LT_TF;tf++){ const int n0=wtok0+tf*8;
                    int b[2]={ *(int*)&Xsh[buf][n0+gid][ko+4*t], *(int*)&Xsh[buf][n0+gid][ko+16+4*t] };
                    int cc[4]={0,0,0,0}; mma(cc,a,b);
                    const float d8_0=Xd[buf][n0+2*t][s],s8_0=Xs[buf][n0+2*t][s],d8_1=Xd[buf][n0+2*t+1][s],s8_1=Xs[buf][n0+2*t+1][s];
                    acc[rf][tf][0]+=dwA*d8_0*cc[0]-dmA*s8_0; acc[rf][tf][1]+=dwA*d8_1*cc[1]-dmA*s8_1;
                    acc[rf][tf][2]+=dwB*d8_0*cc[2]-dmB*s8_0; acc[rf][tf][3]+=dwB*d8_1*cc[3]-dmB*s8_1;
                } } }
        __syncthreads(); buf^=1;
    }
    #pragma unroll
    for(int rf=0;rf<LT_RF;rf++)for(int tf=0;tf<LT_TF;tf++){
        const int rA=row_base+wrow0+rf*16+gid, rB=rA+8, t0=tok_base+wtok0+tf*8+2*t, t1=t0+1;
        if(rA<M){ if(t0<N)y[(int64_t)t0*M+rA]=__float2half(acc[rf][tf][0]); if(t1<N)y[(int64_t)t1*M+rA]=__float2half(acc[rf][tf][1]); }
        if(rB<M){ if(t0<N)y[(int64_t)t0*M+rB]=__float2half(acc[rf][tf][2]); if(t1<N)y[(int64_t)t1*M+rB]=__float2half(acc[rf][tf][3]); }
    }
}

int main(){
    CK(cudaSetDevice(0)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); printf("Device %s SM %d.%d\n",p.name,p.major,p.minor);
    std::mt19937 g(7); std::uniform_int_distribution<int> bd(0,255); std::uniform_real_distribution<float> fd(-1,1);
    auto test=[&](int M,int N,int K){
        const int nbk=K/QK_K, nsb=K/32;
        std::vector<block_q4_K> W((size_t)M*nbk); std::vector<block_q8_1> X((size_t)N*nsb);
        for(auto&w:W){ __half d=__float2half(0.05f*fabsf(fd(g))+0.01f),dm=__float2half(0.03f*fabsf(fd(g))); memcpy(&w.d_raw,&d,2);memcpy(&w.dmin_raw,&dm,2);
            for(int i=0;i<12;i++)w.scales[i]=bd(g); for(int i=0;i<128;i++)w.qs[i]=bd(g); }
        for(auto&x:X){ __half d=__float2half(0.02f*fabsf(fd(g))+0.005f); float sum=0; for(int i=0;i<32;i++){x.qs[i]=(int8_t)(fd(g)*100);sum+=x.qs[i];}
            __half s=__float2half(__half2float(d)*sum); memcpy(&x.d_raw,&d,2);memcpy(&x.s_raw,&s,2); }
        std::vector<float> Yref((size_t)N*M); ref_gemm(Yref,W,X,M,N,K);
        block_q4_K* dW; block_q8_1* dX; half* dY;
        CK(cudaMalloc(&dW,W.size()*144));CK(cudaMalloc(&dX,X.size()*sizeof(block_q8_1)));CK(cudaMalloc(&dY,(size_t)N*M*2));
        CK(cudaMemcpy(dW,W.data(),W.size()*144,cudaMemcpyHostToDevice));CK(cudaMemcpy(dX,X.data(),X.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice));
        dim3 grid((N+LT_BN-1)/LT_BN,(M+LT_BM-1)/LT_BM); int blk=LT_NW*32;
        gemm_lt<<<grid,blk>>>(dY,dW,dX,M,N,K); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<half> Yh((size_t)N*M); CK(cudaMemcpy(Yh.data(),dY,(size_t)N*M*2,cudaMemcpyDeviceToHost));
        float mx=0,rel=0; for(size_t i=0;i<Yh.size();i++){ float e=fabsf(__half2float(Yh[i])-Yref[i]); mx=fmaxf(mx,e); rel=fmaxf(rel,e/(fabsf(Yref[i])+1e-3f)); }
        cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
        for(int i=0;i<3;i++)gemm_lt<<<grid,blk>>>(dY,dW,dX,M,N,K); CK(cudaDeviceSynchronize());
        cudaEventRecord(e0); for(int i=0;i<20;i++)gemm_lt<<<grid,blk>>>(dY,dW,dX,M,N,K); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=20;
        printf("  M=%d N=%d K=%d  max_abs=%.3f rel=%.4f  %.3fms  %.1f GOP/s\n",M,N,K,mx,rel,ms,2.0*M*N*K/1e9/(ms/1e3));
        cudaFree(dW);cudaFree(dX);cudaFree(dY);cudaEventDestroy(e0);cudaEventDestroy(e1);
    };
    printf("-- llama-style MMQ (8 warps, __launch_bounds__(.,2), K-slice, %dx%d tile) --\n",LT_BM,LT_BN);
    test(1536,512,1536); test(4096,512,1536); test(1536,2048,1536);
    return 0;
}
