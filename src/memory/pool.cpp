// pool.cpp — Scratch memory pool (bump allocator, zero malloc during inference)

#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace jllm {

bool ScratchPool::init(int64_t size_bytes) {
    // Use cudaMallocManaged (Unified Memory) instead of cudaMallocHost.
    // cudaMallocHost on Tegra returns a host pointer that is NOT
    // automatically GPU-visible without explicit cudaHostAllocMapped +
    // cudaHostGetDevicePointer plumbing. Symptom of the bug we
    // diagnosed: kernel writes its `output` pointer (a cudaMallocHost'd
    // address) but the CPU later reads the ORIGINAL contents back —
    // kernel's writes landed on a device-side mapping the CPU never sees,
    // CPU memcpy reads the host-side mapping the kernel never touched.
    //
    // cudaMallocManaged returns a single pointer that is mapped on both
    // sides automatically. On Jetson with unified iGPU memory, migration
    // is essentially a no-op — same physical RAM, just consistent mapping
    // tables. Performance matches pinned host memory for this access
    // pattern (sequential reads, occasional CPU debug reads).
    cudaError_t err = cudaMallocManaged(&base_, size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[scratch] cudaMallocManaged(%ld MB) failed: %s\n",
                size_bytes / (1024*1024), cudaGetErrorString(err));
        return false;
    }
    capacity_ = size_bytes;
    offset_ = 0;
    fprintf(stderr, "[scratch] Allocated %ld MB scratch pool (managed)\n",
            size_bytes / (1024*1024));
    return true;
}

void ScratchPool::destroy() {
    // cudaFree pairs with cudaMallocManaged (NOT cudaFreeHost).
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
