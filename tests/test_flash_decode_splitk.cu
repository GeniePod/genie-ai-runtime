// test_flash_decode_splitk.cu — validate split-K (flash-decoding) decode
// attention vs the engine's naive 1-block-per-head kernel.
//
// Decode attention in genie launches grid=(n_heads) — only 8 blocks on the
// 8-SM Orin (ncu: ~11% occupancy, 3% throughput). At depth the per-head block
// walks the whole KV serially. Split-K splits the KV range across n_splits
// blocks per head (grid = n_heads*n_splits), each computing a partial online
// softmax; a reduce kernel combines them. Fills the SMs at depth.
//
// Compile: /usr/local/cuda-12.6/bin/nvcc -arch=sm_87 -O3 test_flash_decode_splitk.cu -o /tmp/tsk
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define ATTN_TILE_KV 64
#define ATTN_BLOCK   128
#define FLT_NINF (-3.4e38f)

__device__ __forceinline__ float warp_red_max(float v){
  for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o)); return v;
}
__device__ __forceinline__ float warp_red_sum(float v){
  for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o); return v;
}
__device__ __forceinline__ float block_max(float v,float*s,int wid,int lane){
  v=warp_red_max(v); if(lane==0)s[wid]=v; __syncthreads();
  if(wid==0){ float x=(lane<(ATTN_BLOCK/32))?s[lane]:FLT_NINF; x=warp_red_max(x); if(lane==0)s[0]=x; }
  __syncthreads(); return s[0];
}
__device__ __forceinline__ float block_sum(float v,float*s,int wid,int lane){
  v=warp_red_sum(v); if(lane==0)s[wid]=v; __syncthreads();
  if(wid==0){ float x=(lane<(ATTN_BLOCK/32))?s[lane]:0.0f; x=warp_red_sum(x); if(lane==0)s[0]=x; }
  __syncthreads(); return s[0];
}

// ── Naive: 1 block per head, online softmax over [0,seq_len) (mirrors engine) ──
__global__ void naive_decode(half* out,const half* q,const half* k,const half* v,
                             int n_heads,int head_dim,int seq_len,float scale){
  const int head=blockIdx.x, tid=threadIdx.x, wid=tid>>5, lane=tid&31;
  extern __shared__ float sm[];
  float* s_scores=sm; float* s_out=sm+ATTN_TILE_KV; float* s_q=sm+ATTN_TILE_KV+head_dim;
  __shared__ float rmax,rsum,corr,red[4];
  for(int d=tid;d<head_dim;d+=blockDim.x){ s_out[d]=0; s_q[d]=__half2float(q[head*head_dim+d]); }
  if(tid==0){ rmax=FLT_NINF; rsum=0; corr=1; } __syncthreads();
  for(int ks=0;ks<seq_len;ks+=ATTN_TILE_KV){
    int tl=min(ATTN_TILE_KV,seq_len-ks);
    for(int t=wid;t<tl;t+=(blockDim.x>>5)){
      const half* kr=k+(int64_t)(ks+t)*head_dim; float dot=0;
      for(int d=lane;d<head_dim;d+=32) dot+=s_q[d]*__half2float(kr[d]);
      dot=warp_red_sum(dot); if(lane==0)s_scores[t]=dot*scale;
    } __syncthreads();
    float lm=FLT_NINF; for(int t=tid;t<tl;t+=blockDim.x) lm=fmaxf(lm,s_scores[t]);
    float tm=block_max(lm,red,wid,lane);
    if(tid==0){ float om=rmax; rmax=fmaxf(rmax,tm); corr=expf(om-rmax); rsum*=corr; } __syncthreads();
    for(int d=tid;d<head_dim;d+=blockDim.x) s_out[d]*=corr;
    float ls=0; for(int t=tid;t<tl;t+=blockDim.x){ float p=expf(s_scores[t]-rmax); s_scores[t]=p; ls+=p; }
    float tsm=block_sum(ls,red,wid,lane); if(tid==0)rsum+=tsm; __syncthreads();
    for(int d=tid;d<head_dim;d+=blockDim.x){ float val=0;
      for(int t=0;t<tl;t++) val+=s_scores[t]*__half2float(v[(int64_t)(ks+t)*head_dim+d]);
      s_out[d]+=val; } __syncthreads();
  }
  float inv=(rsum>0)?1.0f/rsum:0; for(int d=tid;d<head_dim;d+=blockDim.x) out[head*head_dim+d]=__float2half(s_out[d]*inv);
}

// ── Split-K partial: grid (n_heads, n_splits); each owns a KV slice ──
__global__ void splitk_partial(float* pm,float* pl,float* pacc,
                               const half* q,const half* k,const half* v,
                               int n_heads,int head_dim,int seq_len,int n_splits,float scale){
  const int head=blockIdx.x, split=blockIdx.y, tid=threadIdx.x, wid=tid>>5, lane=tid&31;
  const int per=(seq_len+n_splits-1)/n_splits;
  const int kv0=split*per, kv1=min(kv0+per,seq_len);
  extern __shared__ float sm[];
  float* s_scores=sm; float* s_out=sm+ATTN_TILE_KV; float* s_q=sm+ATTN_TILE_KV+head_dim;
  __shared__ float rmax,rsum,corr,red[4];
  for(int d=tid;d<head_dim;d+=blockDim.x){ s_out[d]=0; s_q[d]=__half2float(q[head*head_dim+d]); }
  if(tid==0){ rmax=FLT_NINF; rsum=0; corr=1; } __syncthreads();
  const int base=head*n_splits+split;
  if(kv0>=kv1){ if(tid==0){ pm[base]=FLT_NINF; pl[base]=0; } for(int d=tid;d<head_dim;d+=blockDim.x) pacc[(int64_t)base*head_dim+d]=0; return; }
  for(int ks=kv0;ks<kv1;ks+=ATTN_TILE_KV){
    int tl=min(ATTN_TILE_KV,kv1-ks);
    for(int t=wid;t<tl;t+=(blockDim.x>>5)){
      const half* kr=k+(int64_t)(ks+t)*head_dim; float dot=0;
      for(int d=lane;d<head_dim;d+=32) dot+=s_q[d]*__half2float(kr[d]);
      dot=warp_red_sum(dot); if(lane==0)s_scores[t]=dot*scale;
    } __syncthreads();
    float lm=FLT_NINF; for(int t=tid;t<tl;t+=blockDim.x) lm=fmaxf(lm,s_scores[t]);
    float tm=block_max(lm,red,wid,lane);
    if(tid==0){ float om=rmax; rmax=fmaxf(rmax,tm); corr=expf(om-rmax); rsum*=corr; } __syncthreads();
    for(int d=tid;d<head_dim;d+=blockDim.x) s_out[d]*=corr;
    float ls=0; for(int t=tid;t<tl;t+=blockDim.x){ float p=expf(s_scores[t]-rmax); s_scores[t]=p; ls+=p; }
    float tsm=block_sum(ls,red,wid,lane); if(tid==0)rsum+=tsm; __syncthreads();
    for(int d=tid;d<head_dim;d+=blockDim.x){ float val=0;
      for(int t=0;t<tl;t++) val+=s_scores[t]*__half2float(v[(int64_t)(ks+t)*head_dim+d]);
      s_out[d]+=val; } __syncthreads();
  }
  // write UNNORMALIZED partial: m, l, acc (acc = sum p*V, not divided by l)
  if(tid==0){ pm[base]=rmax; pl[base]=rsum; }
  for(int d=tid;d<head_dim;d+=blockDim.x) pacc[(int64_t)base*head_dim+d]=s_out[d];
}

// ── Split-K reduce: grid (n_heads); combine n_splits partials per head ──
__global__ void splitk_reduce(half* out,const float* pm,const float* pl,const float* pacc,
                              int n_heads,int head_dim,int n_splits){
  const int head=blockIdx.x, tid=threadIdx.x;
  __shared__ float M,L;
  if(tid==0){ float m=FLT_NINF; for(int s=0;s<n_splits;s++) m=fmaxf(m,pm[head*n_splits+s]);
    float l=0; for(int s=0;s<n_splits;s++) l+=pl[head*n_splits+s]*expf(pm[head*n_splits+s]-m);
    M=m; L=l; } __syncthreads();
  float inv=(L>0)?1.0f/L:0;
  for(int d=tid;d<head_dim;d+=blockDim.x){
    float acc=0;
    for(int s=0;s<n_splits;s++){ int base=head*n_splits+s;
      acc+=pacc[(int64_t)base*head_dim+d]*expf(pm[base]-M); }
    out[head*head_dim+d]=__float2half(acc*inv);
  }
}

static half h(float x){ return __float2half(x); }

int main(int argc,char**argv){
  int head_dim = argc>1?atoi(argv[1]):512;
  int seq_len  = argc>2?atoi(argv[2]):4096;
  int n_splits = argc>3?atoi(argv[3]):8;
  int n_heads=8;
  srand(1234);
  std::vector<half> q(n_heads*head_dim), k(seq_len*head_dim), v(seq_len*head_dim);
  for(auto&x:q) x=h((rand()/(float)RAND_MAX-0.5f));
  for(auto&x:k) x=h((rand()/(float)RAND_MAX-0.5f));
  for(auto&x:v) x=h((rand()/(float)RAND_MAX-0.5f));
  half *dq,*dk,*dv,*on,*os; float *pm,*pl,*pa;
  cudaMalloc(&dq,q.size()*2); cudaMalloc(&dk,k.size()*2); cudaMalloc(&dv,v.size()*2);
  cudaMalloc(&on,n_heads*head_dim*2); cudaMalloc(&os,n_heads*head_dim*2);
  cudaMalloc(&pm,n_heads*n_splits*4); cudaMalloc(&pl,n_heads*n_splits*4); cudaMalloc(&pa,(int64_t)n_heads*n_splits*head_dim*4);
  cudaMemcpy(dq,q.data(),q.size()*2,cudaMemcpyHostToDevice);
  cudaMemcpy(dk,k.data(),k.size()*2,cudaMemcpyHostToDevice);
  cudaMemcpy(dv,v.data(),v.size()*2,cudaMemcpyHostToDevice);
  float scale=1.0f/sqrtf((float)head_dim);
  int smem=(ATTN_TILE_KV+2*head_dim)*sizeof(float);
  cudaFuncSetAttribute(naive_decode,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  cudaFuncSetAttribute(splitk_partial,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);

  // correctness
  naive_decode<<<n_heads,ATTN_BLOCK,smem>>>(on,dq,dk,dv,n_heads,head_dim,seq_len,scale);
  dim3 g(n_heads,n_splits);
  splitk_partial<<<g,ATTN_BLOCK,smem>>>(pm,pl,pa,dq,dk,dv,n_heads,head_dim,seq_len,n_splits,scale);
  splitk_reduce<<<n_heads,ATTN_BLOCK>>>(os,pm,pl,pa,n_heads,head_dim,n_splits);
  cudaDeviceSynchronize();
  cudaError_t e=cudaGetLastError(); if(e){ printf("CUDA err: %s\n",cudaGetErrorString(e)); return 1; }
  std::vector<half> rn(n_heads*head_dim), rs(n_heads*head_dim);
  cudaMemcpy(rn.data(),on,rn.size()*2,cudaMemcpyDeviceToHost);
  cudaMemcpy(rs.data(),os,rs.size()*2,cudaMemcpyDeviceToHost);
  float maxabs=0; for(size_t i=0;i<rn.size();i++) maxabs=fmaxf(maxabs,fabsf(__half2float(rn[i])-__half2float(rs[i])));

  // timing
  int iters=200; cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  for(int i=0;i<iters;i++) naive_decode<<<n_heads,ATTN_BLOCK,smem>>>(on,dq,dk,dv,n_heads,head_dim,seq_len,scale);
  cudaEventRecord(b); cudaEventSynchronize(b); float tn; cudaEventElapsedTime(&tn,a,b);
  cudaEventRecord(a);
  for(int i=0;i<iters;i++){ splitk_partial<<<g,ATTN_BLOCK,smem>>>(pm,pl,pa,dq,dk,dv,n_heads,head_dim,seq_len,n_splits,scale);
    splitk_reduce<<<n_heads,ATTN_BLOCK>>>(os,pm,pl,pa,n_heads,head_dim,n_splits); }
  cudaEventRecord(b); cudaEventSynchronize(b); float ts; cudaEventElapsedTime(&ts,a,b);

  printf("head_dim=%d seq_len=%d n_splits=%d | max_abs_diff=%.5f | naive=%.1fus splitk=%.1fus speedup=%.2fx\n",
         head_dim,seq_len,n_splits, maxabs, tn/iters*1000, ts/iters*1000, tn/ts);
  return 0;
}
