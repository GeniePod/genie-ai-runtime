// jllm_memory.h — Memory-first design for Jetson unified memory
//
// Core principle: on Jetson, CPU and GPU share the SAME 8 GB LPDDR5.
// Every byte allocated for the model is a byte NOT available for KV cache,
// OS, camera, or anything else. We track everything.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

namespace jllm {

// ── Memory budget — knows exactly where every MB goes ────────────────────

struct MemoryBudget {
    int64_t total_mb;          // physical DRAM (typically 7633 MB after carveouts)
    int64_t os_mb;             // kernel + services (measured at boot)
    int64_t cma_mb;            // CMA reservation (from /proc/meminfo)
    int64_t cuda_ctx_mb;       // CUDA context overhead (~200-300 MB)
    int64_t model_mb;          // loaded model weights
    int64_t kv_cache_mb;       // current KV cache usage
    int64_t scratch_mb;        // activation / intermediate buffers
    int64_t safety_mb;         // headroom to prevent OOM (default 256 MB)

    int64_t used_mb() const {
        return os_mb + cma_mb + cuda_ctx_mb + model_mb + kv_cache_mb + scratch_mb;
    }

    int64_t free_mb() const {
        return total_mb - used_mb() - safety_mb;
    }

    bool can_allocate(int64_t request_mb) const {
        return free_mb() >= request_mb;
    }

    // Maximum KV cache size given current allocations
    int64_t max_kv_mb() const {
        return total_mb - os_mb - cma_mb - cuda_ctx_mb - model_mb - scratch_mb - safety_mb;
    }

    // Calculate max context tokens for a specific model config
    int max_context(int n_layers, int n_kv_heads, int head_dim, int kv_bytes) const {
        int64_t bytes_per_token = 2LL * n_layers * n_kv_heads * head_dim * kv_bytes;
        if (bytes_per_token == 0) return 0;
        return static_cast<int>((max_kv_mb() * 1024 * 1024) / bytes_per_token);
    }

    void print() const {
        fprintf(stderr,
            "╔══════════════════════════════════╗\n"
            "║   JLLM Memory Budget             ║\n"
            "╠══════════════════════════════════╣\n"
            "║ Total DRAM:     %5ld MB         ║\n"
            "║ OS + kernel:   -%5ld MB         ║\n"
            "║ CMA reserved:  -%5ld MB         ║\n"
            "║ CUDA context:  -%5ld MB         ║\n"
            "║ Model weights: -%5ld MB         ║\n"
            "║ KV cache:      -%5ld MB         ║\n"
            "║ Scratch:       -%5ld MB         ║\n"
            "║ Safety margin: -%5ld MB         ║\n"
            "╠══════════════════════════════════╣\n"
            "║ FREE:           %5ld MB         ║\n"
            "╚══════════════════════════════════╝\n",
            total_mb, os_mb, cma_mb, cuda_ctx_mb,
            model_mb, kv_cache_mb, scratch_mb, safety_mb, free_mb());
    }
};

// Read actual system state from /proc
MemoryBudget probe_system_memory();

// ── OOM Guard — prevents crash by stopping generation early ──────────────

class OOMGuard {
public:
    explicit OOMGuard(int64_t safety_mb = 256) : safety_mb_(safety_mb) {}

    // Check before every KV cache extension
    bool can_extend(int64_t additional_bytes) const;

    // Get real free memory from kernel (not cached)
    int64_t real_free_mb() const;

    // Emergency: free everything possible
    void emergency_free();

private:
    int64_t safety_mb_;
};

// ── KV Cache Pool — pre-allocated, no runtime malloc ─────────────────────
//
// Key insight: on Jetson unified memory, we allocate KV cache with
// cudaMallocHost (pinned). This gives:
//   - GPU can read it directly (zero-copy, no cudaMemcpy)
//   - CPU can read it directly (for debugging, export)
//   - No page fault overhead (pinned = not swappable)
//
// For longer context: overflow to unpinned CPU memory (slower GPU access
// via page faults, but doesn't OOM).

class KVCachePool {
public:
    struct Config {
        int n_layers;
        int n_kv_heads;
        int head_dim;
        int max_context;       // max tokens in GPU-fast pool
        int overflow_context;  // extra tokens in CPU-slower pool (0 = disabled)
        int kv_type_bytes;     // 1 = INT8, 2 = FP16
    };

    bool init(const Config& cfg);
    void destroy();

    // Get pointer for layer l, token position t
    // Returns GPU-accessible pointer (fast for recent, slower for overflow)
    void* key_ptr(int layer, int pos);
    void* val_ptr(int layer, int pos);

    int64_t used_bytes() const;
    int64_t capacity_bytes() const;
    int     used_tokens() const { return used_tokens_; }
    int     max_tokens() const { return cfg_.max_context + cfg_.overflow_context; }
    bool    is_fast_position(int pos) const { return pos < cfg_.max_context; }

    // Evict oldest tokens from fast pool to overflow (when fast pool full)
    void evict(int n_tokens);

    // Reset for new conversation
    void clear();

private:
    Config cfg_ = {};
    void*  gpu_pool_ = nullptr;   // cudaMallocManaged — CPU/GPU visible
    void*  cpu_pool_ = nullptr;   // malloc — overflow, slower GPU access
    int    used_tokens_ = 0;
    int    gpu_tokens_ = 0;

    int64_t entry_bytes() const {
        return 2LL * cfg_.n_kv_heads * cfg_.head_dim * cfg_.kv_type_bytes;
    }
    int64_t layer_stride() const {
        return entry_bytes() * (cfg_.max_context + cfg_.overflow_context);
    }
};

// ── Scratch allocator — one-time alloc, reused by decode buffers ─────────

class ScratchPool {
public:
    bool init(int64_t size_bytes);
    void destroy();

    void* get(int64_t size);  // bump pointer, reset each forward pass
    void  reset();            // call at start of each decode step

    // Save current allocation cursor and rewind back to it later. Used by
    // batched prefill to reset transient per-layer scratch while keeping
    // the prompt-activation buffer (allocated below the mark) intact.
    int64_t mark() const { return offset_; }
    void    rewind_to(int64_t saved_offset);

    int64_t used() const { return offset_; }
    int64_t capacity() const { return capacity_; }

private:
    void*   base_ = nullptr;
    int64_t capacity_ = 0;
    int64_t offset_ = 0;
};

}  // namespace jllm
