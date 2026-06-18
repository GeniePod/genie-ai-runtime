// pool.cpp — Scratch memory pool (bump allocator, zero malloc during inference)

#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace jllm {

bool ScratchPool::init(int64_t size_bytes) {
    // Scratch is only used by CUDA kernels and cudaMemcpy calls. A plain
    // device allocation avoids managed-memory bookkeeping and leaves more
    // room for device-resident weights on 8 GB Jetsons.
    cudaError_t err = cudaMalloc(&base_, size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[scratch] cudaMalloc(%ld MB) failed: %s\n",
                size_bytes / (1024*1024), cudaGetErrorString(err));
        return false;
    }
    capacity_ = size_bytes;
    offset_ = 0;
    fprintf(stderr, "[scratch] Allocated %ld MB scratch pool (device)\n",
            size_bytes / (1024*1024));
    return true;
}

void ScratchPool::destroy() {
    // cudaFree pairs with cudaMalloc.
    if (base_) { cudaFree(base_); base_ = nullptr; }
    capacity_ = 0;
    offset_ = 0;
}

void* ScratchPool::get(int64_t size) {
    // Align to 256 bytes (GPU coalescing + cache line friendly)
    size = (size + 255) & ~255LL;

    if (offset_ + size > capacity_) {
        fprintf(stderr, "[scratch] FATAL: pool exhausted (%ld / %ld bytes)\n",
                offset_ + size, capacity_);
        return nullptr;
    }

    void* ptr = (char*)base_ + offset_;
    offset_ += size;
    return ptr;
}

void ScratchPool::reset() {
    offset_ = 0;
}

void ScratchPool::rewind_to(int64_t saved_offset) {
    if (saved_offset < 0 || saved_offset > capacity_) {
        fprintf(stderr, "[scratch] WARN: rewind_to(%ld) out of range "
                "[0, %ld]; ignoring\n", saved_offset, capacity_);
        return;
    }
    offset_ = saved_offset;
}

}  // namespace jllm
