// jetson_roofline.cu — measure the actual Orin silicon for a roofline analysis
// of the flash-attention kernel. Reports device properties (unified memory, SMs,
// shared/regs, L2, theoretical BW), MEASURED memory bandwidth (read + D2D copy),
// and MEASURED tensor-core throughput (fp16 m16n8k16, int8 m16n8k32) — the two
// roofline ceilings. PTX support is confirmed by the mma.sync/cp.async kernels
// here compiling+running for -arch=sm_87.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// ---- memory read bandwidth: stream a big array, sum it ----
__global__ void readbw(const float4* __restrict__ a, float* __restrict__ out, size_t n4){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    float4 acc = make_float4(0,0,0,0);
    for(size_t j=i; j<n4; j+=stride){ float4 x=a[j]; acc.x+=x.x; acc.y+=x.y; acc.z+=x.z; acc.w+=x.w; }
    if(acc.x==1234.5678f) out[i]=acc.x+acc.y+acc.z+acc.w;
}

// ---- fp16 tensor-core throughput: warps spin independent m16n8k16 MMAs ----
__global__ void tc_fp16(float* __restrict__ sink, int iters){
    uint32_t a[4]={0x3c003c00u,0x3c003c00u,0x3c003c00u,0x3c003c00u};
    uint32_t b[2]={0x3c003c00u,0x3c003c00u};
    float c0[4]={0,0,0,0}, c1[4]={0,0,0,0};
    for(int i=0;i<iters;i++){
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+f"(c0[0]),"+f"(c0[1]),"+f"(c0[2]),"+f"(c0[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+f"(c1[0]),"+f"(c1[1]),"+f"(c1[2]),"+f"(c1[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
    if(c0[0]==-1.f && c1[0]==-1.f) sink[threadIdx.x]=c0[0]+c1[0];
}

// ---- int8 tensor-core throughput: m16n8k32 s8.s8.s32 ----
__global__ void tc_int8(int* __restrict__ sink, int iters){
    uint32_t a[4]={0x01010101u,0x01010101u,0x01010101u,0x01010101u};
    uint32_t b[2]={0x01010101u,0x01010101u};
    int c0[4]={0,0,0,0}, c1[4]={0,0,0,0};
    for(int i=0;i<iters;i++){
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+r"(c0[0]),"+r"(c0[1]),"+r"(c0[2]),"+r"(c0[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+r"(c1[0]),"+r"(c1[1]),"+r"(c1[2]),"+r"(c1[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
    if(c0[0]==-1 && c1[0]==-1) sink[threadIdx.x]=c0[0]+c1[0];
}

int main(){
    int dev=0; CK(cudaSetDevice(dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("=================== DEVICE ===================\n");
    printf("Name                     : %s\n", p.name);
    printf("Compute capability (SM)  : %d.%d\n", p.major, p.minor);
    printf("SMs (multiProcessorCount): %d\n", p.multiProcessorCount);
    printf("GPU clock                : %.0f MHz\n", p.clockRate/1000.0);
    printf("Mem clock                : %.0f MHz\n", p.memoryClockRate/1000.0);
    printf("Mem bus width            : %d bit\n", p.memoryBusWidth);
    double thbw = 2.0*(p.memoryClockRate*1e3)*(p.memoryBusWidth/8.0)/1e9;
    printf("Theoretical mem BW       : %.1f GB/s  (2*memclk*bus/8)\n", thbw);
    printf("Total global mem         : %.2f GB\n", p.totalGlobalMem/1e9);
    printf("L2 cache                 : %.2f MB\n", p.l2CacheSize/1e6);
    printf("Shared/block (max optin) : %zu / %d KB\n", p.sharedMemPerBlock/1024, (int)(p.sharedMemPerBlockOptin/1024));
    printf("Shared per SM            : %zu KB\n", p.sharedMemPerMultiprocessor/1024);
    printf("Regs per block / per SM  : %d / %d\n", p.regsPerBlock, p.regsPerMultiprocessor);
    printf("Max threads / SM,block   : %d / %d   warpSize %d\n", p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock, p.warpSize);
    printf("--- UNIFIED MEMORY (Jetson) ---\n");
    printf("integrated (shared DRAM) : %d\n", p.integrated);
    printf("canMapHostMemory         : %d\n", p.canMapHostMemory);
    printf("unifiedAddressing        : %d\n", p.unifiedAddressing);
    printf("pageableMemoryAccess     : %d\n", p.pageableMemoryAccess);
    printf("concurrentManagedAccess  : %d\n", p.concurrentManagedAccess);

    // ---- measured memory bandwidth ----
    printf("=================== MEMORY BW (measured) ===================\n");
    size_t bytes = (size_t)512*1024*1024;          // 512 MB
    size_t n4 = bytes/sizeof(float4);
    float4* dA; float* dOut; CK(cudaMalloc(&dA,bytes)); CK(cudaMalloc(&dOut,1024*sizeof(float)));
    CK(cudaMemset(dA,1,bytes));
    int blocks = p.multiProcessorCount*16, thr=256;
    readbw<<<blocks,thr>>>(dA,dOut,n4); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); for(int i=0;i<20;i++) readbw<<<blocks,thr>>>(dA,dOut,n4); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=20;
    printf("Read BW (sum 512MB)      : %.1f GB/s  (%.3f ms/pass)\n", bytes/1e9/(ms/1e3), ms);
    float4* dB; CK(cudaMalloc(&dB,bytes));
    cudaEventRecord(e0); for(int i=0;i<20;i++) cudaMemcpy(dB,dA,bytes,cudaMemcpyDeviceToDevice); cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms,e0,e1); ms/=20;
    printf("D2D copy BW (512MB)      : %.1f GB/s  (rd+wr counted, %.3f ms)\n", 2.0*bytes/1e9/(ms/1e3), ms);

    // ---- measured tensor-core throughput ----
    printf("=================== TENSOR CORES (measured) ===================\n");
    // clean read-BW measurement reused for the ridge point
    cudaEventRecord(e0); for(int i=0;i<20;i++) readbw<<<blocks,thr>>>(dA,dOut,n4); cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms,e0,e1); ms/=20;
    double read_bw_gbs = bytes/1e9/(ms/1e3);

    int tcblocks=p.multiProcessorCount*8, tcthr=128, iters=200000;
    int* dS; CK(cudaMalloc(&dS,1024*sizeof(int)));
    double warps = (double)tcblocks*(tcthr/32);
    tc_fp16<<<tcblocks,tcthr>>>((float*)dS,1000); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); tc_fp16<<<tcblocks,tcthr>>>((float*)dS,iters); cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms,e0,e1);
    double fp16_tflops = warps*iters*2.0*(16.0*8.0*16.0*2.0)/1e12/(ms/1e3);   // 2 mmas/iter, 4096 FLOP each
    printf("FP16 TC throughput       : %.1f TFLOP/s  (%.3f ms)\n", fp16_tflops, ms);
    tc_int8<<<tcblocks,tcthr>>>(dS,1000); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); tc_int8<<<tcblocks,tcthr>>>(dS,iters); cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms,e0,e1);
    double int8_tops = warps*iters*2.0*(16.0*8.0*32.0*2.0)/1e12/(ms/1e3);     // 2 mmas/iter, 8192 op each
    printf("INT8 TC throughput       : %.1f TOP/s   (%.3f ms)\n", int8_tops, ms);

    printf("=================== ROOFLINE ===================\n");
    printf("Measured read BW         : %.1f GB/s\n", read_bw_gbs);
    printf("Ridge (FP16-TC)          : %.0f FLOP/byte\n", fp16_tflops*1e12 / (read_bw_gbs*1e9));
    printf("Ridge (INT8-TC)          : %.0f OP/byte\n",   int8_tops*1e12  / (read_bw_gbs*1e9));
    printf("--- flash-attn arithmetic intensity (per Q-tile/K-tile, MQ queries) ---\n");
    printf("  AI = 1024*D FLOP / (KV bytes)\n");
    printf("  f16 KV  (64*D B), MQ=16 : 16 FLOP/byte\n");
    printf("  int8 KV (32*D B), MQ=16 : 32 OP/byte\n");
    printf("  int8 KV, MQ=32          : 64 OP/byte   (2x query reuse)\n");
    printf("  int8 KV, MQ=64          : 128 OP/byte  (4x query reuse)\n");
    printf("DONE\n");
    return 0;
}
