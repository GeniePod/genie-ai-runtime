// test_mmq_q4k_cpasync.cu — cp.async double-buffering for the int8-MMA Q4_K
// prefill GEMM. Compares the current int8 kernel (#107) against a version that
// prefetches the next K-block's Q4_K weight tile into double-buffered shared
// via cp.async (__pipeline_memcpy_async), so the int8 tensor cores process
// block b while block b+1 streams in. Q4_K blocks are 144 B (=16x9, 16-aligned
// in the weight array), so the async copy is well-aligned.

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

constexpr int QK_K = 256;
struct __attribute__((packed)) block_q4_K { uint16_t d_raw, dmin_raw; uint8_t scales[12], qs[QK_K/2]; };
static_assert(sizeof(block_q4_K) == 144, "");
struct __attribute__((packed)) block_q8_1 { __half d, s; int8_t qs[32]; };
static_assert(sizeof(block_q8_1) == 36, "");

__host__ __device__ __forceinline__ float raw_fp16_to_float(uint16_t h){
#ifdef __CUDA_ARCH__
  return __half2float(__ushort_as_half(h));
#else
  uint32_t s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF; float r;
  if(e==0)r=std::ldexp((float)m,-24); else if(e==31)r=m?NAN:INFINITY; else r=std::ldexp((float)(m+1024),(int)e-25);
  return s?-r:r;
#endif
}
__host__ __device__ __forceinline__ void get_scale_min_k4(int j,const uint8_t*q,uint8_t&d,uint8_t&m){
  if(j<4){d=q[j]&63;m=q[j+4]&63;}else{d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}

__global__ void quantize_q8_1(const half* __restrict__ x, block_q8_1* __restrict__ y, int ng){
  const int g=blockIdx.x; if(g>=ng)return; const int l=threadIdx.x;
  const float v=__half2float(x[(int64_t)g*32+l]); float a=fabsf(v);
  #pragma unroll
  for(int o=16;o>0;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
  const float d=a/127.f,id=d>0?1.f/d:0.f; int q=max(-127,min(127,__float2int_rn(v*id))); int s=q;
  #pragma unroll
  for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
  y[g].qs[l]=(int8_t)q; if(l==0){y[g].d=__float2half(d);y[g].s=__float2half(d*(float)s);}
}

constexpr int TM=16, TN=8, NW=4, BN=TN*NW;

// ── baseline int8 kernel (matches the merged #107) ──
__global__ void mmq_base(half* __restrict__ y, const block_q4_K* __restrict__ W,
                         const block_q8_1* __restrict__ XQ, int M, int N, int K){
  const int rb=blockIdx.y*TM, ktb=blockIdx.x*BN; if(rb>=M)return;
  const int t=threadIdx.x, wid=t>>5, lane=t&31, gid=lane>>2, tg=lane&3;
  const int tb=ktb+wid*TN, nb=K/QK_K, nsb=nb*8, tok0=tb+tg*2, tok1=tok0+1;
  float d0=0,d1=0,d2=0,d3=0;
  __shared__ int8_t A[TM][32]; __shared__ float pd[TM][8], pm[TM][8];
  for(int b=0;b<nb;b++){
    { int row=t>>3,sb=t&7,gr=rb+row; float d=0,dm=0;
      if(gr<M){const block_q4_K&bk=W[(int64_t)gr*nb+b]; float da=raw_fp16_to_float(bk.d_raw),dn=raw_fp16_to_float(bk.dmin_raw);
        uint8_t sc,mn; get_scale_min_k4(sb,bk.scales,sc,mn); d=da*sc; dm=dn*mn;}
      pd[row][sb]=d; pm[row][sb]=dm; }
    __syncthreads();
    for(int sb=0;sb<8;sb++){
      int il=sb>>1,par=sb&1;
      { int row=t>>3,c4=(t&7)<<2,gr=rb+row; const block_q4_K*bp=(gr<M)?&W[(int64_t)gr*nb+b]:nullptr;
        #pragma unroll
        for(int s=0;s<4;s++){int c=c4+s; int8_t q=0; if(gr<M){uint8_t qb=bp->qs[32*il+c]; q=(int8_t)(par?(qb>>4):(qb&0xF));} A[row][c]=q;} }
      __syncthreads();
      int gsb=b*8+sb;
      int a0=*(const int*)&A[gid][4*tg], a1=*(const int*)&A[gid+8][4*tg], a2=*(const int*)&A[gid][16+4*tg], a3=*(const int*)&A[gid+8][16+4*tg];
      int gt=tb+gid, bb0=0,bb1=0;
      if(gt<N){const block_q8_1*q=&XQ[(int64_t)gt*nsb+gsb]; bb0=*(const int*)(q->qs+4*tg); bb1=*(const int*)(q->qs+16+4*tg);}
      int c0=0,c1=0,c2=0,c3=0;
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(bb0),"r"(bb1));
      float dA=pd[gid][sb],dmA=pm[gid][sb],dB=pd[gid+8][sb],dmB=pm[gid+8][sb];
      float d80=0,s80=0,d81=0,s81=0;
      if(tok0<N){d80=__half2float(XQ[(int64_t)tok0*nsb+gsb].d); s80=__half2float(XQ[(int64_t)tok0*nsb+gsb].s);}
      if(tok1<N){d81=__half2float(XQ[(int64_t)tok1*nsb+gsb].d); s81=__half2float(XQ[(int64_t)tok1*nsb+gsb].s);}
      d0+=dA*d80*c0-dmA*s80; d1+=dA*d81*c1-dmA*s81; d2+=dB*d80*c2-dmB*s80; d3+=dB*d81*c3-dmB*s81;
      __syncthreads();
    }
  }
  int ra=rb+gid,rbb=rb+gid+8;
  if(ra<M){if(tok0<N)y[(int64_t)tok0*M+ra]=__float2half(d0); if(tok1<N)y[(int64_t)tok1*M+ra]=__float2half(d1);}
  if(rbb<M){if(tok0<N)y[(int64_t)tok0*M+rbb]=__float2half(d2); if(tok1<N)y[(int64_t)tok1*M+rbb]=__float2half(d3);}
}

// ── cp.async double-buffered version ──
// Prefetch the next K-block's 16-row Q4_K tile into shared while MMAing.
__global__ void mmq_cpasync(half* __restrict__ y, const block_q4_K* __restrict__ W,
                            const block_q8_1* __restrict__ XQ, int M, int N, int K){
  const int rb=blockIdx.y*TM, ktb=blockIdx.x*BN; if(rb>=M)return;
  const int t=threadIdx.x, wid=t>>5, lane=t&31, gid=lane>>2, tg=lane&3;
  const int tb=ktb+wid*TN, nb=K/QK_K, nsb=nb*8, tok0=tb+tg*2, tok1=tok0+1;
  float d0=0,d1=0,d2=0,d3=0;
  __align__(16) __shared__ block_q4_K Wsh[2][TM];   // double-buffered, 16B-aligned
  __shared__ int8_t A[TM][32]; __shared__ float pd[TM][8], pm[TM][8];

  // Prefetch a 16-row Q4_K tile via cp.async. block_q4_K is 144 B = 9x16,
  // and blocks sit at 144*k (16-aligned) in the weight array, so copy in
  // 16-byte chunks (__pipeline_memcpy_async only supports 4/8/16 B).
  auto prefetch=[&](int buf,int b){
    for(int ci=t; ci<TM*9; ci+=blockDim.x){
      int row=ci/9, c16=ci%9, gr=rb+row;
      const char* src=(const char*)((gr<M)? &W[(int64_t)gr*nb+b] : &W[0])+c16*16;
      char* dst=(char*)&Wsh[buf][row]+c16*16;
      __pipeline_memcpy_async(dst, src, 16);
    }
    __pipeline_commit();
  };
  prefetch(0,0);

  for(int b=0;b<nb;b++){
    if(b+1<nb) prefetch((b+1)&1, b+1);
    __pipeline_wait_prior(b+1<nb ? 1 : 0);
    __syncthreads();
    const block_q4_K* Wb = Wsh[b&1];
    { int row=t>>3,sb=t&7,gr=rb+row; float d=0,dm=0;
      if(gr<M){const block_q4_K&bk=Wb[row]; float da=raw_fp16_to_float(bk.d_raw),dn=raw_fp16_to_float(bk.dmin_raw);
        uint8_t sc,mn; get_scale_min_k4(sb,bk.scales,sc,mn); d=da*sc; dm=dn*mn;}
      pd[row][sb]=d; pm[row][sb]=dm; }
    __syncthreads();
    for(int sb=0;sb<8;sb++){
      int il=sb>>1,par=sb&1;
      { int row=t>>3,c4=(t&7)<<2,gr=rb+row;
        #pragma unroll
        for(int s=0;s<4;s++){int c=c4+s; int8_t q=0; if(gr<M){uint8_t qb=Wb[row].qs[32*il+c]; q=(int8_t)(par?(qb>>4):(qb&0xF));} A[row][c]=q;} }
      __syncthreads();
      int gsb=b*8+sb;
      int a0=*(const int*)&A[gid][4*tg], a1=*(const int*)&A[gid+8][4*tg], a2=*(const int*)&A[gid][16+4*tg], a3=*(const int*)&A[gid+8][16+4*tg];
      int gt=tb+gid, bb0=0,bb1=0;
      if(gt<N){const block_q8_1*q=&XQ[(int64_t)gt*nsb+gsb]; bb0=*(const int*)(q->qs+4*tg); bb1=*(const int*)(q->qs+16+4*tg);}
      int c0=0,c1=0,c2=0,c3=0;
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(bb0),"r"(bb1));
      float dA=pd[gid][sb],dmA=pm[gid][sb],dB=pd[gid+8][sb],dmB=pm[gid+8][sb];
      float d80=0,s80=0,d81=0,s81=0;
      if(tok0<N){d80=__half2float(XQ[(int64_t)tok0*nsb+gsb].d); s80=__half2float(XQ[(int64_t)tok0*nsb+gsb].s);}
      if(tok1<N){d81=__half2float(XQ[(int64_t)tok1*nsb+gsb].d); s81=__half2float(XQ[(int64_t)tok1*nsb+gsb].s);}
      d0+=dA*d80*c0-dmA*s80; d1+=dA*d81*c1-dmA*s81; d2+=dB*d80*c2-dmB*s80; d3+=dB*d81*c3-dmB*s81;
      __syncthreads();
    }
  }
  int ra=rb+gid,rbb=rb+gid+8;
  if(ra<M){if(tok0<N)y[(int64_t)tok0*M+ra]=__float2half(d0); if(tok1<N)y[(int64_t)tok1*M+ra]=__float2half(d1);}
  if(rbb<M){if(tok0<N)y[(int64_t)tok0*M+rbb]=__float2half(d2); if(tok1<N)y[(int64_t)tok1*M+rbb]=__float2half(d3);}
}

// ── int8 reference ──
static void host_q8(const half* x,int8_t* q,float& d8){
  float a=0; for(int i=0;i<32;i++)a=std::fmax(a,std::fabs(__half2float(x[i])));
  float d=a/127.f,id=d>0?1.f/d:0.f; for(int i=0;i<32;i++){int v=(int)std::lrint(__half2float(x[i])*id); q[i]=(int8_t)std::max(-127,std::min(127,v));} d8=d;
}
static void ref_i8(const std::vector<block_q4_K>& W,const std::vector<half>& X,std::vector<half>& Y,int M,int N,int K){
  int nb=K/QK_K;
  for(int t=0;t<N;t++)for(int r=0;r<M;r++){float acc=0;
    for(int b=0;b<nb;b++){const block_q4_K&bk=W[(int64_t)r*nb+b]; float da=raw_fp16_to_float(bk.d_raw),dn=raw_fp16_to_float(bk.dmin_raw);
      for(int sb=0;sb<8;sb++){int il=sb>>1,par=sb&1; uint8_t sc,mn; get_scale_min_k4(sb,bk.scales,sc,mn); float d=da*sc,dm=dn*mn;
        int8_t q8[32]; float d8; host_q8(&X[(int64_t)t*K+b*QK_K+sb*32],q8,d8); int dot=0;
        for(int c=0;c<32;c++){uint8_t qb=bk.qs[32*il+c]; int w=par?(qb>>4):(qb&0xF); dot+=w*(int)q8[c];}
        acc+=d*d8*(float)dot-dm*(d8*[&]{int s=0;for(int c=0;c<32;c++)s+=q8[c];return (float)s;}());}}
    Y[(int64_t)t*M+r]=__float2half(acc);}
}
static void fillW(std::vector<block_q4_K>& W,uint32_t s){std::mt19937 r(s);std::uniform_int_distribution<int>b(0,255);std::uniform_real_distribution<float>da(.005f,.02f),dn(.001f,.004f);
  for(auto&k:W){k.d_raw=__half_as_ushort(__float2half(da(r)));k.dmin_raw=__half_as_ushort(__float2half(dn(r)));for(int i=0;i<12;i++)k.scales[i]=b(r);for(int i=0;i<128;i++)k.qs[i]=b(r);}}
static void fillX(std::vector<half>& X,uint32_t s){std::mt19937 r(s);std::uniform_real_distribution<float>d(-1,1);for(auto&v:X)v=__float2half(d(r));}
static block_q8_1* quantX(const std::vector<half>& hX,int N,int K){int ng=N*K/32;half*dX;block_q8_1*dXQ;
  CK(cudaMalloc(&dX,hX.size()*sizeof(half)));CK(cudaMalloc(&dXQ,(size_t)ng*sizeof(block_q8_1)));
  CK(cudaMemcpy(dX,hX.data(),hX.size()*sizeof(half),cudaMemcpyHostToDevice));quantize_q8_1<<<ng,32>>>(dX,dXQ,ng);CK(cudaDeviceSynchronize());cudaFree(dX);return dXQ;}

static int check(const char* lbl,void(*kern)(half*,const block_q4_K*,const block_q8_1*,int,int,int),
                 int M,int N,int K,const std::vector<block_q4_K>&hW,const std::vector<half>&hX,const std::vector<half>&hRef){
  block_q4_K*dW;half*dY; CK(cudaMalloc(&dW,hW.size()*sizeof(block_q4_K)));CK(cudaMalloc(&dY,(size_t)N*M*sizeof(half)));
  CK(cudaMemcpy(dW,hW.data(),hW.size()*sizeof(block_q4_K),cudaMemcpyHostToDevice));CK(cudaMemset(dY,0,(size_t)N*M*sizeof(half)));
  block_q8_1*dXQ=quantX(hX,N,K); dim3 g((N+BN-1)/BN,(M+TM-1)/TM,1);
  kern<<<g,NW*32>>>(dY,dW,dXQ,M,N,K); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<half> hY(N*M); CK(cudaMemcpy(hY.data(),dY,(size_t)N*M*sizeof(half),cudaMemcpyDeviceToHost));
  float mi=0; int bad=0; for(int i=0;i<N*M;i++){float e=std::fabs(__half2float(hY[i])-__half2float(hRef[i])); mi=std::fmax(mi,e); if(e>0.5f)bad++;}
  printf("  %-8s vs int8ref: max_abs=%g  %d/%d outside 0.5  %s\n",lbl,mi,bad,N*M,bad?"FAIL":"PASS");
  cudaFree(dW);cudaFree(dY);cudaFree(dXQ); return bad?1:0;
}
static void bench(const char* lbl,void(*kern)(half*,const block_q4_K*,const block_q8_1*,int,int,int),int M,int N,int K){
  int nb=K/QK_K; std::vector<block_q4_K>hW(M*nb);std::vector<half>hX(N*K); fillW(hW,7+M+K);fillX(hX,9+N);
  block_q4_K*dW;half*dY; CK(cudaMalloc(&dW,hW.size()*sizeof(block_q4_K)));CK(cudaMalloc(&dY,(size_t)N*M*sizeof(half)));
  CK(cudaMemcpy(dW,hW.data(),hW.size()*sizeof(block_q4_K),cudaMemcpyHostToDevice)); block_q8_1*dXQ=quantX(hX,N,K);
  dim3 g((N+BN-1)/BN,(M+TM-1)/TM,1);
  for(int i=0;i<5;i++)kern<<<g,NW*32>>>(dY,dW,dXQ,M,N,K); CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventRecord(e0);
  for(int i=0;i<100;i++)kern<<<g,NW*32>>>(dY,dW,dXQ,M,N,K); cudaEventRecord(e1);cudaEventSynchronize(e1);
  float ms=0;cudaEventElapsedTime(&ms,e0,e1);ms/=100;
  printf("  %-8s M=%5d N=%3d K=%5d  ms=%6.3f  GFLOPS=%6.1f\n",lbl,M,N,K,ms,2.0*M*N*K/(ms*1e6));
  cudaEventDestroy(e0);cudaEventDestroy(e1);cudaFree(dW);cudaFree(dY);cudaFree(dXQ);
}

int main(){
  CK(cudaSetDevice(0)); cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0)); printf("Device: %s SM %d.%d\n",p.name,p.major,p.minor);
  const int M=128,N=16,K=1024,nb=K/QK_K; std::vector<block_q4_K>hW(M*nb);std::vector<half>hX(N*K),hRef(N*M);
  fillW(hW,0xC0FFEE);fillX(hX,0xFEED); ref_i8(hW,hX,hRef,M,N,K);
  printf("\n-- correctness (M=%d N=%d K=%d) --\n",M,N,K);
  int bad=0; bad+=check("base",mmq_base,M,N,K,hW,hX,hRef); bad+=check("cpasync",mmq_cpasync,M,N,K,hW,hX,hRef);
  printf("\n-- throughput: base vs cp.async --\n");
  int shapes[][3]={{2048,256,1536},{1536,256,2048},{12288,256,1536},{2560,256,2560}};
  for(auto&s:shapes){bench("base",mmq_base,s[0],s[1],s[2]); bench("cpasync",mmq_cpasync,s[0],s[1],s[2]);}
  return bad?1:0;
}
