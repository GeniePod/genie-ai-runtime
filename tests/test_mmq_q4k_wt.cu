// test_mmq_q4k_wt.cu — CUTLASS/CODA-style WARP-TILED int8 Q4_K MMQ GEMM for Orin.
//
// The engine kernel (gemm_mmq_q4k_i8) does ONE m16n8k32 MMA per Q4_K sub-block
// then unpacks/loads/dequants -> ncu showed int8-TC util ~6%: the MMA is starved
// by per-MMA overhead. This is the CUTLASS architecture (which CODA builds on):
// a LARGE register-accumulator warp-tile so each sub-block does MANY MMAs that
// reuse shared-staged operands, with the Q4_K dequant applied to the register
// accumulator (CODA's "walk the accumulator sub-tile, mutate in registers").
// Ampere adaptation: cp.async (not TMA), hand-written CUDA (not CuTeDSL), int8.
//
// Y[N tokens x M rows] = X(q8_1)[N x K] . W(Q4_K)[M x K].
// Block tile BM x BN; WARPS_M x WARPS_N warps; warp-tile (BM/WARPS_M)x(BN/WARPS_N);
// each warp accumulates RF=warpM/16 row-frags x TF=warpN/8 token-frags MMAs.

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

#define QK_K 256
struct __align__(2) block_q4_K { uint16_t d_raw, dmin_raw; uint8_t scales[12]; uint8_t qs[128]; }; // 144B
struct block_q8_1 { uint16_t d_raw, s_raw; int8_t qs[32]; };                                        // 36B

__host__ __device__ inline float rawh2f(uint16_t h){ __half x; memcpy(&x,&h,2); return __half2float(x); }
__host__ __device__ inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m){
    if (j < 4){ d = q[j] & 63; m = q[j+4] & 63; }
    else { d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

// ---------- CPU reference (exact int8-MMQ math) ----------
static void ref_gemm(std::vector<float>& Y, const std::vector<block_q4_K>& W,
                     const std::vector<block_q8_1>& X, int M, int N, int K){
    const int nbk = K/QK_K, nsb = K/32;
    for (int n=0;n<N;n++) for (int m=0;m<M;m++){
        float acc=0;
        for (int b=0;b<nbk;b++){
            const block_q4_K& w = W[(size_t)m*nbk+b];
            const float dall=rawh2f(w.d_raw), dmin=rawh2f(w.dmin_raw);
            for (int sb=0;sb<8;sb++){
                uint8_t sc,mn; get_scale_min_k4(sb,w.scales,sc,mn);
                const float dw=dall*sc, dmw=dmin*mn;
                const int il=sb>>1, parity=sb&1;
                const block_q8_1& x = X[(size_t)n*nsb + b*8+sb];
                const float dx=rawh2f(x.d_raw), sx=rawh2f(x.s_raw);
                int idot=0;
                for (int c=0;c<32;c++){
                    int qw = parity ? (w.qs[32*il+c]>>4) : (w.qs[32*il+c]&0xF);
                    idot += qw * (int)x.qs[c];
                }
                acc += dw*dx*idot - dmw*sx;
            }
        }
        Y[(size_t)n*M+m]=acc;
    }
}

// ---------- warp-tiled kernel ----------
#ifndef CFG_BM
#define CFG_BM 64
#define CFG_BN 64
#endif
constexpr int BM=CFG_BM, BN=CFG_BN, WARPS_M=2, WARPS_N=2;
constexpr int WARP_M=BM/WARPS_M, WARP_N=BN/WARPS_N;   // 32 x 16
constexpr int RF=WARP_M/16, TF=WARP_N/8;              // 2 row-frags x 2 token-frags = 4 MMAs

__device__ __forceinline__ int unpack4(const uint8_t* qs, int parity){
    const uint32_t w=*reinterpret_cast<const uint32_t*>(qs);
    return (int)(parity ? ((w>>4)&0x0F0F0F0Fu) : (w&0x0F0F0F0Fu));
}
__device__ __forceinline__ void mma(int c[4], const int a[4], const int b[2]){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__global__ void gemm_wt(half* __restrict__ Y, const block_q4_K* __restrict__ W,
                        const block_q8_1* __restrict__ X, int M, int N, int K){
    const int row_base = blockIdx.y*BM;
    const int tok_base = blockIdx.x*BN;
    const int wid = threadIdx.x>>5, lane=threadIdx.x&31;
    const int wm = wid % WARPS_M, wn = wid / WARPS_M;     // warp coords
    const int gid = lane>>2, t = lane&3;
    const int nbk = K/QK_K, nsb=K/32;

    const int wrow0 = wm*WARP_M;   // warp's local row base (0..BM)
    const int wtok0 = wn*WARP_N;   // warp's local token base (0..BN)

    __align__(16) __shared__ block_q4_K Wsh[2][BM];
    auto prefetch=[&](int buf,int blk){
        for (int ci=threadIdx.x; ci<BM*9; ci+=blockDim.x){
            const int row=ci/9, c16=ci%9, g=row_base+row;
            const char* src=(const char*)((g<M)?&W[(size_t)g*nbk+blk]:&W[0])+c16*16;
            __pipeline_memcpy_async((char*)&Wsh[buf][row]+c16*16, src, 16);
        }
        __pipeline_commit();
    };

    float acc[RF][TF][4];
    #pragma unroll
    for(int i=0;i<RF;i++) for(int j=0;j<TF;j++){ acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.f; }

    prefetch(0,0);
    for (int b=0;b<nbk;b++){
        if (b+1<nbk) prefetch((b+1)&1,b+1);
        __pipeline_wait_prior(b+1<nbk?1:0);
        __syncthreads();
        const block_q4_K* Wb=Wsh[b&1];

        for (int sb=0;sb<8;sb++){
            const int il=sb>>1, parity=sb&1, gsb=b*8+sb;
            // B fragments (activations) for this warp's TF token-frags: load once, reuse across RF.
            int bf[TF][2]; float dx[TF][2], sx[TF][2];
            #pragma unroll
            for(int tf=0;tf<TF;tf++){
                const int gtok = tok_base + wtok0 + tf*8 + gid;
                bf[tf][0]=bf[tf][1]=0;
                if (gtok<N){ const block_q8_1* q=&X[(size_t)gtok*nsb+gsb];
                    bf[tf][0]=*reinterpret_cast<const int*>(q->qs+4*t);
                    bf[tf][1]=*reinterpret_cast<const int*>(q->qs+16+4*t); }
                // dequant scales for the two tokens this lane owns in c-cols (tok0,tok1 = wtok0+tf*8+2t,+1)
                const int tk0=tok_base+wtok0+tf*8+2*t, tk1=tk0+1;
                dx[tf][0]=sx[tf][0]=dx[tf][1]=sx[tf][1]=0.f;
                if(tk0<N){ dx[tf][0]=rawh2f(X[(size_t)tk0*nsb+gsb].d_raw); sx[tf][0]=rawh2f(X[(size_t)tk0*nsb+gsb].s_raw);}
                if(tk1<N){ dx[tf][1]=rawh2f(X[(size_t)tk1*nsb+gsb].d_raw); sx[tf][1]=rawh2f(X[(size_t)tk1*nsb+gsb].s_raw);}
            }
            #pragma unroll
            for(int rf=0;rf<RF;rf++){
                const int lrowA = wrow0 + rf*16 + gid;       // local rows in Wsh
                const int lrowB = wrow0 + rf*16 + gid + 8;
                const block_q4_K& WA=Wb[lrowA]; const block_q4_K& WB=Wb[lrowB];
                int a[4]={ unpack4(WA.qs+32*il+4*t,parity), unpack4(WB.qs+32*il+4*t,parity),
                           unpack4(WA.qs+32*il+16+4*t,parity), unpack4(WB.qs+32*il+16+4*t,parity) };
                uint8_t scA,mnA,scB,mnB;
                get_scale_min_k4(sb,WA.scales,scA,mnA); get_scale_min_k4(sb,WB.scales,scB,mnB);
                const float dwA=rawh2f(WA.d_raw)*scA, dmA=rawh2f(WA.dmin_raw)*mnA;
                const float dwB=rawh2f(WB.d_raw)*scB, dmB=rawh2f(WB.dmin_raw)*mnB;
                #pragma unroll
                for(int tf=0;tf<TF;tf++){
                    int c[4]={0,0,0,0};
                    mma(c,a,bf[tf]);
                    // c0=(rowA,tok0) c1=(rowA,tok1) c2=(rowB,tok0) c3=(rowB,tok1)
                    acc[rf][tf][0]+=dwA*dx[tf][0]*c[0]-dmA*sx[tf][0];
                    acc[rf][tf][1]+=dwA*dx[tf][1]*c[1]-dmA*sx[tf][1];
                    acc[rf][tf][2]+=dwB*dx[tf][0]*c[2]-dmB*sx[tf][0];
                    acc[rf][tf][3]+=dwB*dx[tf][1]*c[3]-dmB*sx[tf][1];
                }
            }
        }
        __syncthreads();
    }

    // store
    #pragma unroll
    for(int rf=0;rf<RF;rf++) for(int tf=0;tf<TF;tf++){
        const int rA=row_base+wrow0+rf*16+gid, rB=rA+8;
        const int t0=tok_base+wtok0+tf*8+2*t, t1=t0+1;
        if(rA<M){ if(t0<N)Y[(size_t)t0*M+rA]=__float2half(acc[rf][tf][0]); if(t1<N)Y[(size_t)t1*M+rA]=__float2half(acc[rf][tf][1]); }
        if(rB<M){ if(t0<N)Y[(size_t)t0*M+rB]=__float2half(acc[rf][tf][2]); if(t1<N)Y[(size_t)t1*M+rB]=__float2half(acc[rf][tf][3]); }
    }
}

// ---------- harness ----------
int main(){
    CK(cudaSetDevice(0)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("Device %s SM %d.%d\n",p.name,p.major,p.minor);
    std::mt19937 g(123); std::uniform_int_distribution<int> bd(0,255); std::uniform_real_distribution<float> fd(-1,1);
    auto test=[&](int M,int N,int K){
        const int nbk=K/QK_K, nsb=K/32;
        std::vector<block_q4_K> W((size_t)M*nbk); std::vector<block_q8_1> X((size_t)N*nsb);
        for(auto&w:W){ __half d=__float2half(0.05f*fabsf(fd(g))+0.01f),dm=__float2half(0.03f*fabsf(fd(g)));
            memcpy(&w.d_raw,&d,2); memcpy(&w.dmin_raw,&dm,2); for(int i=0;i<12;i++)w.scales[i]=bd(g); for(int i=0;i<128;i++)w.qs[i]=bd(g); }
        for(auto&x:X){ __half d=__float2half(0.02f*fabsf(fd(g))+0.005f); float sum=0; for(int i=0;i<32;i++){x.qs[i]=(int8_t)(fd(g)*100); sum+=x.qs[i];}
            __half s=__float2half(__half2float(d)*sum); memcpy(&x.d_raw,&d,2); memcpy(&x.s_raw,&s,2); }
        std::vector<float> Yref((size_t)N*M); ref_gemm(Yref,W,X,M,N,K);
        block_q4_K* dW; block_q8_1* dX; half* dY;
        CK(cudaMalloc(&dW,W.size()*144)); CK(cudaMalloc(&dX,X.size()*sizeof(block_q8_1))); CK(cudaMalloc(&dY,(size_t)N*M*2));
        CK(cudaMemcpy(dW,W.data(),W.size()*144,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dX,X.data(),X.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice));
        dim3 grid((N+BN-1)/BN,(M+BM-1)/BM); int blk=WARPS_M*WARPS_N*32;
        gemm_wt<<<grid,blk>>>(dY,dW,dX,M,N,K); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<half> Yh((size_t)N*M); CK(cudaMemcpy(Yh.data(),dY,(size_t)N*M*2,cudaMemcpyDeviceToHost));
        float mx=0,rel=0; for(size_t i=0;i<Yh.size();i++){ float e=fabsf(__half2float(Yh[i])-Yref[i]); mx=fmaxf(mx,e); rel=fmaxf(rel,e/(fabsf(Yref[i])+1e-3f)); }
        cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
        for(int i=0;i<3;i++)gemm_wt<<<grid,blk>>>(dY,dW,dX,M,N,K); CK(cudaDeviceSynchronize());
        cudaEventRecord(e0); for(int i=0;i<20;i++)gemm_wt<<<grid,blk>>>(dY,dW,dX,M,N,K); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=20;
        double gops=2.0*M*N*K/1e9; printf("  M=%d N=%d K=%d  max_abs=%.3f rel=%.4f  %.3fms  %.1f GOP/s\n",M,N,K,mx,rel,ms,gops/(ms/1e3));
        cudaFree(dW);cudaFree(dX);cudaFree(dY);cudaEventDestroy(e0);cudaEventDestroy(e1);
    };
    printf("-- warp-tiled int8 Q4_K GEMM (BM=%d BN=%d, %dx%d MMAs/sub-block/warp) --\n",BM,BN,RF,TF);
    test(1536,512,1536);   // wo-ish
    test(4096,512,1536);   // gate/up-ish
    test(1536,2048,1536);  // larger N
    return 0;
}
