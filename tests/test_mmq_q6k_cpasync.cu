// test_mmq_q6k_cpasync.cu — cp.async for the int8-MMA Q6_K (ffn_down) GEMM.
// Q6_K's 210 B block is not 16-aligned in the weight array (210 = 105*2), so
// cp.async (4/8/16 B) can't copy it cleanly. Fix: repack once into a padded
// 224 B stride (=14*16, 16-aligned), then double-buffer the next K-block's
// tile into shared via __pipeline_memcpy_async while the tensor cores work the
// current one. Compares the baseline int8 Q6_K kernel vs the cp.async one.

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
struct __attribute__((packed)) block_q6_K { uint8_t ql[128], qh[64]; int8_t scales[16]; uint16_t d_raw; };
static_assert(sizeof(block_q6_K) == 210, "");
struct block_q6_K_pad { block_q6_K blk; uint8_t pad[14]; };   // 224 = 14*16
static_assert(sizeof(block_q6_K_pad) == 224, "");
struct __attribute__((packed)) block_q8_1 { __half d, s; int8_t qs[32]; };

__host__ __device__ __forceinline__ float raw_fp16_to_float(uint16_t h){
#ifdef __CUDA_ARCH__
  return __half2float(__ushort_as_half(h));
#else
  uint32_t s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF; float r;
  if(e==0)r=std::ldexp((float)m,-24); else if(e==31)r=m?NAN:INFINITY; else r=std::ldexp((float)(m+1024),(int)e-25);
  return s?-r:r;
#endif
}
__host__ __device__ __forceinline__ int load_int_u16(const void* p){const uint16_t* s=(const uint16_t*)p; return (int)((uint32_t)s[0]|((uint32_t)s[1]<<16));}
#ifndef __CUDA_ARCH__
static int host_vsubss4(int a,int b){int r=0;for(int i=0;i<4;i++){int x=(int8_t)((a>>(8*i))&0xFF),y=(int8_t)((b>>(8*i))&0xFF),s=x-y;if(s>127)s=127;if(s<-128)s=-128;r|=(s&0xFF)<<(8*i);}return r;}
#endif
__host__ __device__ __forceinline__ int q6_unpack4(const block_q6_K& blk,int gi){
  const int p=4*gi,n=(p>=128)?128:0,ph=p-n,g=ph>>5,r=ph&31;
  const uint8_t* ql_h=blk.ql+(n>>1); const uint8_t* qh_h=blk.qh+(n>>2);
  const int qlbyte=(ph<64)?ph:(ph-64); const int ql_int=load_int_u16(ql_h+qlbyte);
  const int vil=(ph>=64)?((ql_int>>4)&0x0F0F0F0F):(ql_int&0x0F0F0F0F);
  const int qh_int=load_int_u16(qh_h+r); const int sh=2*g; const int sel=qh_int&(0x03030303<<sh);
  const int vih=(sh<=4?(sel<<(4-sh)):(sel>>(sh-4)))&0x30303030;
#ifdef __CUDA_ARCH__
  return __vsubss4(vil|vih,0x20202020);
#else
  return host_vsubss4(vil|vih,0x20202020);
#endif
}
__global__ void quantize_q8_1(const half* __restrict__ x,block_q8_1* __restrict__ y,int ng){
  const int g=blockIdx.x; if(g>=ng)return; const int l=threadIdx.x; const float v=__half2float(x[(int64_t)g*32+l]); float a=fabsf(v);
  #pragma unroll
  for(int o=16;o>0;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
  const float d=a/127.f,id=d>0?1.f/d:0.f; int q=max(-127,min(127,__float2int_rn(v*id))); int s=q;
  #pragma unroll
  for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
  y[g].qs[l]=(int8_t)q; if(l==0){y[g].d=__float2half(d);y[g].s=__float2half(d*(float)s);}
}

constexpr int TM=16,TN=8,NW=4,BN=TN*NW;

// baseline int8 Q6_K (matches merged #108; reads W from global 210-stride)
__global__ void q6_base(half* __restrict__ y,const block_q6_K* __restrict__ W,const block_q8_1* __restrict__ XQ,int M,int N,int K){
  const int rb=blockIdx.y*TM,ktb=blockIdx.x*BN; if(rb>=M)return;
  const int t=threadIdx.x,wid=t>>5,lane=t&31,gid=lane>>2,tg=lane&3;
  const int tb=ktb+wid*TN,nb=K/QK_K,nsb=nb*8,tok0=tb+tg*2,tok1=tok0+1;
  float d0=0,d1=0,d2=0,d3=0; __shared__ int8_t A[TM][16]; __shared__ float pdsc[TM][16];
  for(int b=0;b<nb;b++){
    {int row=t>>3,two=t&7,gr=rb+row; float d=0; const block_q6_K*bp=nullptr; if(gr<M){bp=&W[(int64_t)gr*nb+b];d=raw_fp16_to_float(bp->d_raw);}
     #pragma unroll
     for(int h=0;h<2;h++){int j=two+8*h; pdsc[row][j]=(gr<M)?d*(float)bp->scales[j]:0.f;}}
    __syncthreads();
    for(int j=0;j<16;j++){
      if(t<64){int row=t>>2,gl=t&3,gr=rb+row; int vi=0; if(gr<M)vi=q6_unpack4(W[(int64_t)gr*nb+b],4*j+gl); *(int*)&A[row][4*gl]=vi;}
      __syncthreads();
      int a0=*(const int*)&A[gid][4*tg],a1=*(const int*)&A[gid+8][4*tg];
      int gt=tb+gid,qsb=b*8+(j>>1),qoff=(j&1)*16,bb0=0;
      if(gt<N)bb0=*(const int*)(XQ[(int64_t)gt*nsb+qsb].qs+qoff+4*tg);
      int c0=0,c1=0,c2=0,c3=0;
      asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};\n":"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(bb0));
      float dA=pdsc[gid][j],dB=pdsc[gid+8][j],d80=0,d81=0;
      if(tok0<N)d80=__half2float(XQ[(int64_t)tok0*nsb+qsb].d); if(tok1<N)d81=__half2float(XQ[(int64_t)tok1*nsb+qsb].d);
      d0+=dA*d80*c0; d1+=dA*d81*c1; d2+=dB*d80*c2; d3+=dB*d81*c3; __syncthreads();
    }
  }
  int ra=rb+gid,rbb=rb+gid+8;
  if(ra<M){if(tok0<N)y[(int64_t)tok0*M+ra]=__float2half(d0); if(tok1<N)y[(int64_t)tok1*M+ra]=__float2half(d1);}
  if(rbb<M){if(tok0<N)y[(int64_t)tok0*M+rbb]=__float2half(d2); if(tok1<N)y[(int64_t)tok1*M+rbb]=__float2half(d3);}
}

// cp.async: prefetch the next K-block's 16-row tile from the PADDED (224 B,
// 16-aligned) weight array into double-buffered shared.
__global__ void q6_cpasync(half* __restrict__ y,const block_q6_K_pad* __restrict__ Wp,const block_q8_1* __restrict__ XQ,int M,int N,int K){
  const int rb=blockIdx.y*TM,ktb=blockIdx.x*BN; if(rb>=M)return;
  const int t=threadIdx.x,wid=t>>5,lane=t&31,gid=lane>>2,tg=lane&3;
  const int tb=ktb+wid*TN,nb=K/QK_K,nsb=nb*8,tok0=tb+tg*2,tok1=tok0+1;
  float d0=0,d1=0,d2=0,d3=0;
  __align__(16) __shared__ block_q6_K_pad Wsh[2][TM];
  __shared__ int8_t A[TM][16]; __shared__ float pdsc[TM][16];
  auto prefetch=[&](int buf,int b){
    for(int ci=t;ci<TM*14;ci+=blockDim.x){int row=ci/14,c16=ci%14,gr=rb+row;
      const char* src=(const char*)((gr<M)?&Wp[(int64_t)gr*nb+b]:&Wp[0])+c16*16;
      __pipeline_memcpy_async((char*)&Wsh[buf][row]+c16*16,src,16);}
    __pipeline_commit();
  };
  prefetch(0,0);
  for(int b=0;b<nb;b++){
    if(b+1<nb)prefetch((b+1)&1,b+1);
    __pipeline_wait_prior(b+1<nb?1:0); __syncthreads();
    const block_q6_K* Wb=&Wsh[b&1][0].blk;   // stride is sizeof(block_q6_K_pad)
    auto blk=[&](int row)->const block_q6_K&{return Wsh[b&1][row].blk;};
    {int row=t>>3,two=t&7,gr=rb+row; float d=0; if(gr<M)d=raw_fp16_to_float(blk(row).d_raw);
     #pragma unroll
     for(int h=0;h<2;h++){int j=two+8*h; pdsc[row][j]=(gr<M)?d*(float)blk(row).scales[j]:0.f;}}
    __syncthreads();
    for(int j=0;j<16;j++){
      if(t<64){int row=t>>2,gl=t&3,gr=rb+row; int vi=0; if(gr<M)vi=q6_unpack4(blk(row),4*j+gl); *(int*)&A[row][4*gl]=vi;}
      __syncthreads();
      int a0=*(const int*)&A[gid][4*tg],a1=*(const int*)&A[gid+8][4*tg];
      int gt=tb+gid,qsb=b*8+(j>>1),qoff=(j&1)*16,bb0=0;
      if(gt<N)bb0=*(const int*)(XQ[(int64_t)gt*nsb+qsb].qs+qoff+4*tg);
      int c0=0,c1=0,c2=0,c3=0;
      asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};\n":"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(bb0));
      float dA=pdsc[gid][j],dB=pdsc[gid+8][j],d80=0,d81=0;
      if(tok0<N)d80=__half2float(XQ[(int64_t)tok0*nsb+qsb].d); if(tok1<N)d81=__half2float(XQ[(int64_t)tok1*nsb+qsb].d);
      d0+=dA*d80*c0; d1+=dA*d81*c1; d2+=dB*d80*c2; d3+=dB*d81*c3; __syncthreads();
    }
  }
  int ra=rb+gid,rbb=rb+gid+8;
  if(ra<M){if(tok0<N)y[(int64_t)tok0*M+ra]=__float2half(d0); if(tok1<N)y[(int64_t)tok1*M+ra]=__float2half(d1);}
  if(rbb<M){if(tok0<N)y[(int64_t)tok0*M+rbb]=__float2half(d2); if(tok1<N)y[(int64_t)tok1*M+rbb]=__float2half(d3);}
}

static void host_q8(const half* x,int8_t* q,float& d8){float a=0;for(int i=0;i<32;i++)a=std::fmax(a,std::fabs(__half2float(x[i])));float d=a/127.f,id=d>0?1.f/d:0.f;for(int i=0;i<32;i++){int v=(int)std::lrint(__half2float(x[i])*id);q[i]=(int8_t)std::max(-127,std::min(127,v));}d8=d;}
static void ref_i8(const std::vector<block_q6_K>& W,const std::vector<half>& X,std::vector<half>& Y,int M,int N,int K){
  int nb=K/QK_K;
  for(int t=0;t<N;t++)for(int r=0;r<M;r++){float acc=0;
    for(int b=0;b<nb;b++){const block_q6_K&bk=W[(int64_t)r*nb+b]; float d=raw_fp16_to_float(bk.d_raw);
      for(int j=0;j<16;j++){int8_t q8[32];float d8; host_q8(&X[(int64_t)t*K+b*QK_K+((16*j)&~31)],q8,d8); int qoff=(16*j)&31,dot=0;
        for(int m=0;m<16;m++){int k=16*j+m,gi=k>>2,mm=k&3; int q6=(int8_t)((q6_unpack4(bk,gi)>>(8*mm))&0xFF); dot+=q6*(int)q8[qoff+m];}
        acc+=d*(float)bk.scales[j]*d8*(float)dot;}}
    Y[(int64_t)t*M+r]=__float2half(acc);}
}
static void fillW(std::vector<block_q6_K>& W,uint32_t s){std::mt19937 r(s);std::uniform_int_distribution<int>b(0,255),sc(-32,32);std::uniform_real_distribution<float>dd(.001f,.01f);
  for(auto&k:W){for(int i=0;i<128;i++)k.ql[i]=b(r);for(int i=0;i<64;i++)k.qh[i]=b(r);for(int i=0;i<16;i++)k.scales[i]=(int8_t)sc(r);k.d_raw=__half_as_ushort(__float2half(dd(r)));}}
static void fillX(std::vector<half>& X,uint32_t s){std::mt19937 r(s);std::uniform_real_distribution<float>d(-1,1);for(auto&v:X)v=__float2half(d(r));}
static block_q8_1* quantX(const std::vector<half>& hX,int N,int K){int ng=N*K/32;half*dX;block_q8_1*dXQ;CK(cudaMalloc(&dX,hX.size()*sizeof(half)));CK(cudaMalloc(&dXQ,(size_t)ng*sizeof(block_q8_1)));CK(cudaMemcpy(dX,hX.data(),hX.size()*sizeof(half),cudaMemcpyHostToDevice));quantize_q8_1<<<ng,32>>>(dX,dXQ,ng);CK(cudaDeviceSynchronize());cudaFree(dX);return dXQ;}

static int check_base(int M,int N,int K,const std::vector<block_q6_K>&hW,const std::vector<half>&hX,const std::vector<half>&hRef){
  int nb=K/QK_K; block_q6_K*dW;half*dY;CK(cudaMalloc(&dW,hW.size()*sizeof(block_q6_K)));CK(cudaMalloc(&dY,(size_t)N*M*sizeof(half)));
  CK(cudaMemcpy(dW,hW.data(),hW.size()*sizeof(block_q6_K),cudaMemcpyHostToDevice));CK(cudaMemset(dY,0,(size_t)N*M*sizeof(half)));
  block_q8_1*dXQ=quantX(hX,N,K); dim3 g((N+BN-1)/BN,(M+TM-1)/TM,1); q6_base<<<g,NW*32>>>(dY,dW,dXQ,M,N,K); CK(cudaGetLastError());CK(cudaDeviceSynchronize());
  std::vector<half>hY(N*M);CK(cudaMemcpy(hY.data(),dY,(size_t)N*M*sizeof(half),cudaMemcpyDeviceToHost));
  float mi=0;int bad=0;for(int i=0;i<N*M;i++){float e=std::fabs(__half2float(hY[i])-__half2float(hRef[i]));mi=std::fmax(mi,e);if(e>0.5f)bad++;}
  printf("  base    vs int8ref: max_abs=%g  %d/%d outside 0.5  %s\n",mi,bad,N*M,bad?"FAIL":"PASS");
  cudaFree(dW);cudaFree(dY);cudaFree(dXQ);return bad?1:0;
}
static int check_cpa(int M,int N,int K,const std::vector<block_q6_K>&hW,const std::vector<half>&hX,const std::vector<half>&hRef){
  int nb=K/QK_K; std::vector<block_q6_K_pad>hWp(hW.size()); for(size_t i=0;i<hW.size();i++)hWp[i].blk=hW[i];
  block_q6_K_pad*dWp;half*dY;CK(cudaMalloc(&dWp,hWp.size()*sizeof(block_q6_K_pad)));CK(cudaMalloc(&dY,(size_t)N*M*sizeof(half)));
  CK(cudaMemcpy(dWp,hWp.data(),hWp.size()*sizeof(block_q6_K_pad),cudaMemcpyHostToDevice));CK(cudaMemset(dY,0,(size_t)N*M*sizeof(half)));
  block_q8_1*dXQ=quantX(hX,N,K); dim3 g((N+BN-1)/BN,(M+TM-1)/TM,1); q6_cpasync<<<g,NW*32>>>(dY,dWp,dXQ,M,N,K); CK(cudaGetLastError());CK(cudaDeviceSynchronize());
  std::vector<half>hY(N*M);CK(cudaMemcpy(hY.data(),dY,(size_t)N*M*sizeof(half),cudaMemcpyDeviceToHost));
  float mi=0;int bad=0;for(int i=0;i<N*M;i++){float e=std::fabs(__half2float(hY[i])-__half2float(hRef[i]));mi=std::fmax(mi,e);if(e>0.5f)bad++;}
  printf("  cpasync vs int8ref: max_abs=%g  %d/%d outside 0.5  %s\n",mi,bad,N*M,bad?"FAIL":"PASS");
  cudaFree(dWp);cudaFree(dY);cudaFree(dXQ);return bad?1:0;
}
static void bench_base(int M,int N,int K){int nb=K/QK_K;std::vector<block_q6_K>hW(M*nb);std::vector<half>hX(N*K);fillW(hW,5+M+K);fillX(hX,3+N);
  block_q6_K*dW;half*dY;CK(cudaMalloc(&dW,hW.size()*sizeof(block_q6_K)));CK(cudaMalloc(&dY,(size_t)N*M*sizeof(half)));CK(cudaMemcpy(dW,hW.data(),hW.size()*sizeof(block_q6_K),cudaMemcpyHostToDevice));
  block_q8_1*dXQ=quantX(hX,N,K);dim3 g((N+BN-1)/BN,(M+TM-1)/TM,1);for(int i=0;i<5;i++)q6_base<<<g,NW*32>>>(dY,dW,dXQ,M,N,K);CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventRecord(e0);for(int i=0;i<100;i++)q6_base<<<g,NW*32>>>(dY,dW,dXQ,M,N,K);cudaEventRecord(e1);cudaEventSynchronize(e1);
  float ms=0;cudaEventElapsedTime(&ms,e0,e1);ms/=100;printf("  base    M=%5d N=%3d K=%5d ms=%6.3f GFLOPS=%6.1f\n",M,N,K,ms,2.0*M*N*K/(ms*1e6));cudaFree(dW);cudaFree(dY);cudaFree(dXQ);}
static void bench_cpa(int M,int N,int K){int nb=K/QK_K;std::vector<block_q6_K>hW(M*nb);std::vector<half>hX(N*K);fillW(hW,5+M+K);fillX(hX,3+N);
  std::vector<block_q6_K_pad>hWp(hW.size());for(size_t i=0;i<hW.size();i++)hWp[i].blk=hW[i];
  block_q6_K_pad*dWp;half*dY;CK(cudaMalloc(&dWp,hWp.size()*sizeof(block_q6_K_pad)));CK(cudaMalloc(&dY,(size_t)N*M*sizeof(half)));CK(cudaMemcpy(dWp,hWp.data(),hWp.size()*sizeof(block_q6_K_pad),cudaMemcpyHostToDevice));
  block_q8_1*dXQ=quantX(hX,N,K);dim3 g((N+BN-1)/BN,(M+TM-1)/TM,1);for(int i=0;i<5;i++)q6_cpasync<<<g,NW*32>>>(dY,dWp,dXQ,M,N,K);CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventRecord(e0);for(int i=0;i<100;i++)q6_cpasync<<<g,NW*32>>>(dY,dWp,dXQ,M,N,K);cudaEventRecord(e1);cudaEventSynchronize(e1);
  float ms=0;cudaEventElapsedTime(&ms,e0,e1);ms/=100;printf("  cpasync M=%5d N=%3d K=%5d ms=%6.3f GFLOPS=%6.1f\n",M,N,K,ms,2.0*M*N*K/(ms*1e6));cudaFree(dWp);cudaFree(dY);cudaFree(dXQ);}

int main(){
  CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));printf("Device: %s SM %d.%d\n",p.name,p.major,p.minor);
  const int M=128,N=16,K=512,nb=K/QK_K;std::vector<block_q6_K>hW(M*nb);std::vector<half>hX(N*K),hRef(N*M);
  fillW(hW,0xABCDEF);fillX(hX,0x12345);ref_i8(hW,hX,hRef,M,N,K);
  printf("\n-- correctness (M=%d N=%d K=%d) --\n",M,N,K);
  int bad=check_base(M,N,K,hW,hX,hRef)+check_cpa(M,N,K,hW,hX,hRef);
  printf("\n-- throughput: base vs cp.async (Gemma down K=12288) --\n");
  bench_base(1536,256,12288); bench_cpa(1536,256,12288);
  bench_base(2048,256,9728);  bench_cpa(2048,256,9728);
  return bad?1:0;
}
