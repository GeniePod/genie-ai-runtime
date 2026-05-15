// kv_cache.cpp — Pre-allocated KV cache with GPU/CPU tiering
//
// On Jetson unified memory:
//   cudaMalloc     → GPU-visible DRAM without managed-memory bookkeeping
//   malloc         → pageable DRAM, CPU-side overflow storage
//
// Strategy:
//   - "Fast pool" (cudaMalloc): recent tokens, GPU kernels read/write directly
//   - "Overflow pool" (malloc): old tokens, copied back only if overflow is used
//   - When fast pool full: evict oldest to overflow

#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace jllm {

bool KVCachePool::init(const Config& cfg) {
    cfg_ = cfg;

    int64_t fast_bytes = (int64_t)cfg.n_layers * entry_bytes() * cfg.max_context;
    int64_t overflow_bytes = (int64_t)cfg.n_layers * entry_bytes() * cfg.overflow_context;

    fprintf(stderr, "[kv_cache] Allocating fast pool: %ld MB (%d tokens)\n",
            fast_bytes / (1024*1024), cfg.max_context);

    // Fast pool: device allocation. The KV cache is consumed by CUDA kernels
    // and device-to-device copies; keeping it out of managed memory avoids
    // UM bookkeeping and is easier to fit after device-resident weights.
    cudaError_t err = cudaMalloc(&gpu_pool_, fast_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[kv_cache] FATAL: cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    err = cudaMemset(gpu_pool_, 0, fast_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[kv_cache] FATAL: cudaMemset failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(gpu_pool_);
        gpu_pool_ = nullptr;
        return false;
    }

    // Overflow pool: regular malloc (optional)
    if (cfg.overflow_context > 0 && overflow_bytes > 0) {
        fprintf(stderr, "[kv_cache] Allocating overflow pool: %ld MB (%d tokens)\n",
                overflow_bytes / (1024*1024), cfg.overflow_context);
        cpu_pool_ = malloc(overflow_bytes);
        if (!cpu_pool_) {
            fprintf(stderr, "[kv_cache] WARNING: overflow pool alloc failed, disabling\n");
            cfg_.overflow_context = 0;
        } else {
            memset(cpu_pool_, 0, overflow_bytes);
        }
    }

    used_tokens_ = 0;
    gpu_tokens_ = 0;
    return true;
}

void KVCachePool::destroy() {
    // cudaFree pairs with cudaMalloc.
    if (gpu_pool_) { cudaFree(gpu_pool_); gpu_pool_ = nullptr; }
    if (cpu_pool_) { free(cpu_pool_); cpu_pool_ = nullptr; }
    used_tokens_ = 0;
    gpu_tokens_ = 0;
}

void* KVCachePool::key_ptr(int layer, int pos) {
    int64_t eb = entry_bytes() / 2;  // key is half of entry (key + value)
    if (pos < cfg_.max_context) {
        int64_t layer_stride = (int64_t)cfg_.max_context * entry_bytes();
        return (char*)gpu_pool_ + layer * layer_stride + pos * eb;
    } else {
        int overflow_pos = pos - cfg_.max_context;
        int64_t layer_stride = (int64_t)cfg_.overflow_context * entry_bytes();
        return (char*)cpu_pool_ + layer * layer_stride + overflow_pos * eb;
    }
}

void* KVCachePool::val_ptr(int layer, int pos) {
    int64_t eb = entry_bytes() / 2;
    if (pos < cfg_.max_context) {
        int64_t layer_stride = (int64_t)cfg_.max_context * entry_bytes();
        int64_t key_offset = (int64_t)cfg_.max_context * eb;  // values stored after all keys
        return (char*)gpu_pool_ + layer * layer_stride + key_offset + pos * eb;
    } else {
        int overflow_pos = pos - cfg_.max_context;
        int64_t layer_stride = (int64_t)cfg_.overflow_context * entry_bytes();
        int64_t key_offset = (int64_t)cfg_.overflow_context * eb;
        return (char*)cpu_pool_ + layer * layer_stride + key_offset + overflow_pos * eb;
    }
}

int64_t KVCachePool::used_bytes() const {
    return (int64_t)used_tokens_ * entry_bytes() * cfg_.n_layers;
}

int64_t KVCachePool::capacity_bytes() const {
    return (int64_t)(cfg_.max_context + cfg_.overflow_context) * entry_bytes() * cfg_.n_layers;
}

void KVCachePool::evict(int n_tokens) {
    if (!cpu_pool_ || n_tokens <= 0) return;

    int64_t eb = entry_bytes();
    for (int l = 0; l < cfg_.n_layers; l++) {
        char* src = (char*)gpu_pool_ + l * (int64_t)cfg_.max_context * eb;
        char* dst = (char*)cpu_pool_ + l * (int64_t)cfg_.overflow_context * eb;

        // Copy oldest n_tokens to overflow.
        cudaMemcpy(dst, src, n_tokens * eb, cudaMemcpyDeviceToHost);

        // Shift remaining in fast pool
        int remaining = gpu_tokens_ - n_tokens;
        if (remaining > 0) {
            cudaMemcpy(src, src + n_tokens * eb, remaining * eb,
                       cudaMemcpyDeviceToDevice);
        }
    }

    gpu_tokens_ -= n_tokens;
    fprintf(stderr, "[kv_cache] Evicted %d tokens to overflow (gpu: %d, total: %d)\n",
            n_tokens, gpu_tokens_, used_tokens_);
}

void KVCachePool::clear() {
    used_tokens_ = 0;
    gpu_tokens_ = 0;
}

// Path F3b: per-layer gather of the populated portion into a packed
// host buffer. dst layout:
//
//   per layer l in [0, n_layers):
//     [ used_tokens × eb/2 key bytes  ]
//     [ used_tokens × eb/2 value bytes ]
//
// where eb = entry_bytes() = 2 × n_kv_heads × head_dim × kv_type_bytes.
// Within each layer slab in gpu_pool_, keys live at offset
// [layer × max_context × eb, ...) and values follow at offset + max_context × eb/2.
bool KVCachePool::gather_used_to_host(void* dst, int used_tokens) const {
    if (!gpu_pool_ || used_tokens <= 0) return false;
    if (used_tokens > cfg_.max_context) return false;

    const int64_t eb       = entry_bytes();
    const int64_t half_eb  = eb / 2;
    const int64_t layer_in_stride  = (int64_t)cfg_.max_context * eb;
    const int64_t layer_out_stride = (int64_t)used_tokens * eb;
    const int64_t key_used_bytes   = (int64_t)used_tokens * half_eb;

    uint8_t* d = static_cast<uint8_t*>(dst);
    for (int l = 0; l < cfg_.n_layers; l++) {
        const uint8_t* src_keys = static_cast<const uint8_t*>(gpu_pool_)
                                  + l * layer_in_stride;
        const uint8_t* src_vals = src_keys + (int64_t)cfg_.max_context * half_eb;

        uint8_t* dst_keys = d + l * layer_out_stride;
        uint8_t* dst_vals = dst_keys + key_used_bytes;

        cudaError_t err;
        err = cudaMemcpy(dst_keys, src_keys, key_used_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[kv_cache] gather: layer %d keys D2H: %s\n",
                    l, cudaGetErrorString(err));
            return false;
        }
        err = cudaMemcpy(dst_vals, src_vals, key_used_bytes,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[kv_cache] gather: layer %d vals D2H: %s\n",
                    l, cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool KVCachePool::scatter_from_host(const void* src, int used_tokens,
                                    bool zero_remaining)
{
    if (!gpu_pool_ || used_tokens <= 0) return false;
    if (used_tokens > cfg_.max_context) return false;

    const int64_t eb               = entry_bytes();
    const int64_t half_eb          = eb / 2;
    const int64_t layer_out_stride = (int64_t)cfg_.max_context * eb;
    const int64_t layer_in_stride  = (int64_t)used_tokens * eb;
    const int64_t key_used_bytes   = (int64_t)used_tokens * half_eb;
    const int64_t key_unused_bytes = (int64_t)(cfg_.max_context - used_tokens) * half_eb;

    const uint8_t* s = static_cast<const uint8_t*>(src);
    for (int l = 0; l < cfg_.n_layers; l++) {
        uint8_t* dst_keys = static_cast<uint8_t*>(gpu_pool_) + l * layer_out_stride;
        uint8_t* dst_vals = dst_keys + (int64_t)cfg_.max_context * half_eb;

        const uint8_t* src_keys = s + l * layer_in_stride;
        const uint8_t* src_vals = src_keys + key_used_bytes;

        cudaError_t err;
        err = cudaMemcpy(dst_keys, src_keys, key_used_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[kv_cache] scatter: layer %d keys H2D: %s\n",
                    l, cudaGetErrorString(err));
            return false;
        }
        err = cudaMemcpy(dst_vals, src_vals, key_used_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[kv_cache] scatter: layer %d vals H2D: %s\n",
                    l, cudaGetErrorString(err));
            return false;
        }
        if (zero_remaining && key_unused_bytes > 0) {
            err = cudaMemset(dst_keys + key_used_bytes, 0, key_unused_bytes);
            if (err != cudaSuccess) return false;
            err = cudaMemset(dst_vals + key_used_bytes, 0, key_unused_bytes);
            if (err != cudaSuccess) return false;
        }
    }
    used_tokens_ = used_tokens;
    gpu_tokens_  = used_tokens;
    return true;
}

}  // namespace jllm
