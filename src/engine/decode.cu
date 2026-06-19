// decode.cpp — Transformer forward pass + generation loop
//
// Memory-first: zero allocation during decode. All buffers from scratch pool.
// Thermal-aware: checks temperature and memory before every token.
// CUDA graph: captures decode kernels after first iteration, replays for rest.
//
// BUG FIXES applied:
//   #2 — Residual connection properly chained (residual1 for attn, residual2 for FFN)
//   #3 — Embedding memcpy uses cudaMemcpyDefault (works for unified + discrete)
//   #4 — Added #include <sys/mman.h>
//   #5 — CUDA graph capture now runs actual transformer_layer kernels
//   #7 — FP32 logit output via host-side conversion after FP16 GEMV

#include "jllm_engine.h"
#include "../persistence/kv_cache_file.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <unistd.h>
#include <sys/mman.h>   // BUG #4 fix

namespace jllm {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<float, std::milli>;

static bool debug_kernels_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_DEBUG_KERNELS");
        return v && strcmp(v, "0") != 0;
    }();
    return enabled;
}

static bool profile_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_PROFILE");
        return v && strcmp(v, "0") != 0;
    }();
    return enabled;
}

static bool fast_embedding_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_FAST_EMBD");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

// ── Vector add kernel (for residual connections) ─────────────────────────
// Vectorized half2: each thread adds 2 elements with __hadd2, avoiding
// float round-trip.  n is always even (power-of-2 hidden dims), so the
// single-element tail path is a safety net, not a normal code path.
__global__ void vec_add_kernel(half* __restrict__ out,
                                const half* __restrict__ a,
                                const half* __restrict__ b, int n) {
    const int i2 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i2 + 1 < n) {
        *reinterpret_cast<half2*>(&out[i2]) =
            __hadd2(*reinterpret_cast<const half2*>(&a[i2]),
                    *reinterpret_cast<const half2*>(&b[i2]));
    } else if (i2 < n) {
        out[i2] = __hadd(a[i2], b[i2]);
    }
}

static void vec_add(half* out, const half* a, const half* b, int n, cudaStream_t s) {
    const int block = 128;
    const int n2    = (n + 1) >> 1;
    const int grid  = (n2 + block - 1) / block;
    vec_add_kernel<<<grid, block, 0, s>>>(out, a, b, n);
}

// ── FP16 to FP32 conversion kernel ──────────────────────────────────────
// BUG #7 fix: convert logits to FP32 on GPU before D2H copy

__global__ void fp16_to_fp32_kernel(float* __restrict__ out,
                                     const half* __restrict__ in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __half2float(in[i]);
}

static void fp16_to_fp32(float* out, const half* in, int n, cudaStream_t s) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fp16_to_fp32_kernel<<<grid, block, 0, s>>>(out, in, n);
}

// ── Dequantize one embedding row (CPU-side, one row is small) ───────────
// Handles Q4_K (type 12), F32 (type 0), F16 (type 1)

struct embd_block_q4_K {
    uint16_t d_raw;       // FP16 d as raw bits (super-block scale)
    uint16_t dmin_raw;    // FP16 dmin as raw bits (super-block min)
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(embd_block_q4_K) == 144, "");

// CPU-safe FP16→float conversion (no CUDA intrinsics)
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float result;
    if (exp == 0) {
        result = ldexpf((float)mant, -24);  // subnormal
    } else if (exp == 31) {
        result = mant ? NAN : INFINITY;
    } else {
        result = ldexpf((float)(mant + 1024), (int)exp - 25);
    }
    return sign ? -result : result;
}

static void get_scale_min_k4_cpu(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void dequant_q4k_row(float* out, const void* data, int token_id, int hidden_dim) {
    // Each row = hidden_dim elements = hidden_dim/256 Q4_K blocks
    int blocks_per_row = hidden_dim / 256;
    int block_bytes = 144;
    const uint8_t* row_data = (const uint8_t*)data + (int64_t)token_id * blocks_per_row * block_bytes;

    for (int b = 0; b < blocks_per_row; b++) {
        const embd_block_q4_K* blk = (const embd_block_q4_K*)(row_data + b * block_bytes);

        float dall = fp16_to_float(blk->d_raw);
        float dmin = fp16_to_float(blk->dmin_raw);

        for (int il = 0; il < 4; il++) {
            int is = 2 * il;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4_cpu(is + 0, blk->scales, sc1, m1);
            get_scale_min_k4_cpu(is + 1, blk->scales, sc2, m2);

            float d1 = dall * sc1;
            float dm1 = dmin * m1;
            float d2 = dall * sc2;
            float dm2 = dmin * m2;

            const uint8_t* q = blk->qs + 32 * il;
            int out_base = b * 256 + 64 * il;

            for (int l = 0; l < 32; l++) {
                out[out_base + l]      = d1 * (q[l] & 0xF) - dm1;
                out[out_base + l + 32] = d2 * (q[l] >> 4)  - dm2;
            }
        }
    }
}

// Q6_K block layout (matches llama.cpp ggml-quants.h):
//   uint8_t ql[128]     // low 4 bits of 256 6-bit quants
//   uint8_t qh[64]      // high 2 bits of 256 6-bit quants
//   int8_t  scales[16]  // 16 sub-block scales (signed 8-bit)
//   uint16_t d          // FP16 super-block scale
// Total: 128 + 64 + 16 + 2 = 210 bytes per 256-element block.
//
// Many GGUF emitters quantize token_embd.weight to Q6_K even in Q4_K_M
// builds because the embedding accuracy disproportionately affects
// quality. Qwen3-4B-Q4_K_M is one such build.
struct embd_block_q6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d_raw;
};
static_assert(sizeof(embd_block_q6_K) == 210, "Q6_K block must be 210 bytes");

static void dequant_q6k_row(float* out, const void* data, int token_id, int hidden_dim) {
    int blocks_per_row = hidden_dim / 256;
    int block_bytes = 210;
    const uint8_t* row_data = (const uint8_t*)data + (int64_t)token_id * blocks_per_row * block_bytes;

    // Mirrors llama.cpp's `dequantize_row_q6_K` exactly. Each 256-element
    // super-block is processed in two halves of 128 elements; within each
    // half, 32 lanes assemble four 6-bit quants from ql (low nibble) +
    // qh (high 2 bits) and scale by per-16-element sub-block scales.
    for (int b = 0; b < blocks_per_row; b++) {
        const embd_block_q6_K* blk = (const embd_block_q6_K*)(row_data + b * block_bytes);
        const float d = fp16_to_float(blk->d_raw);

        float* y = out + b * 256;
        const uint8_t* ql = blk->ql;
        const uint8_t* qh = blk->qh;
        const int8_t*  sc = blk->scales;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                int q1 = (int)(ql[l]      & 0xF) | (((int)(qh[l] >> 0) & 3) << 4);
                int q2 = (int)(ql[l + 32] & 0xF) | (((int)(qh[l] >> 2) & 3) << 4);
                int q3 = (int)(ql[l]      >> 4)  | (((int)(qh[l] >> 4) & 3) << 4);
                int q4 = (int)(ql[l + 32] >> 4)  | (((int)(qh[l] >> 6) & 3) << 4);
                // 6-bit values are stored unsigned 0..63 but interpreted as
                // signed -32..31. Subtract 32 to recover the signed value.
                y[l +  0] = d * (float)sc[is + 0] * (float)(q1 - 32);
                y[l + 32] = d * (float)sc[is + 2] * (float)(q2 - 32);
                y[l + 64] = d * (float)sc[is + 4] * (float)(q3 - 32);
                y[l + 96] = d * (float)sc[is + 6] * (float)(q4 - 32);
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// Q5_K block layout (matches llama.cpp): d, dmin (fp16), scales[12] (6-bit
// packed, same as Q4_K), qh[32] (one high bit per value), qs[128] (low 4 bits).
// Gemma's token_embd / per_layer_token_embd come Q5_K in the Q4_K_M build.
struct embd_block_q5_K {
    uint16_t d_raw;
    uint16_t dmin_raw;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(embd_block_q5_K) == 176, "");

static void dequant_q5k_row(float* out, const void* data, int token_id, int hidden_dim) {
    int blocks_per_row = hidden_dim / 256;
    int block_bytes = 176;
    const uint8_t* row_data = (const uint8_t*)data + (int64_t)token_id * blocks_per_row * block_bytes;

    for (int b = 0; b < blocks_per_row; b++) {
        const embd_block_q5_K* blk = (const embd_block_q5_K*)(row_data + b * block_bytes);
        float dall = fp16_to_float(blk->d_raw);
        float dmin = fp16_to_float(blk->dmin_raw);
        const uint8_t* qh = blk->qh;

        for (int il = 0; il < 4; il++) {
            int is = 2 * il;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4_cpu(is + 0, blk->scales, sc1, m1);
            get_scale_min_k4_cpu(is + 1, blk->scales, sc2, m2);
            float d1 = dall * sc1, dm1 = dmin * m1;
            float d2 = dall * sc2, dm2 = dmin * m2;
            const uint8_t* q = blk->qs + 32 * il;
            uint8_t u1 = 1 << (2 * il);
            uint8_t u2 = 2 << (2 * il);
            int out_base = b * 256 + 64 * il;
            for (int l = 0; l < 32; l++) {
                int hi1 = (qh[l] & u1) ? 16 : 0;
                int hi2 = (qh[l] & u2) ? 16 : 0;
                out[out_base + l]      = d1 * ((q[l] & 0xF) + hi1) - dm1;
                out[out_base + l + 32] = d2 * ((q[l] >> 4)  + hi2) - dm2;
            }
        }
    }
}

static void dequant_embedding(half* dst, const void* embd_data, int token_id,
                               int hidden_dim, int embd_type, cudaStream_t stream) {
    if (fast_embedding_enabled() && (embd_type == 12 || embd_type == 14) &&
        dequant_embedding_row(dst, embd_data, embd_type, token_id, hidden_dim, stream)) {
        static bool first_q4 = true;
        static bool first_q6 = true;
        bool& first = (embd_type == 12) ? first_q4 : first_q6;
        if (first && debug_kernels_enabled()) {
            cudaStreamSynchronize(stream);
            half h_dbg[8];
            cudaMemcpy(h_dbg, dst, sizeof(h_dbg), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[embd] Q%d_K GPU dequant token %d, first 8 values: ",
                    embd_type == 12 ? 4 : 6, token_id);
            for (int i = 0; i < 8 && i < hidden_dim; i++)
                fprintf(stderr, "%.4f ", __half2float(h_dbg[i]));
            fprintf(stderr, "\n");
        }
        first = false;
        return;
    }

    // Sized for the largest row we dequant: Gemma's per_layer_token_embd is
    // n_layers * ple_dim (e.g. 35*256 = 8960) wide, larger than any hidden_dim.
    float h_row[16384];
    half  h_fp16[16384];

    if (embd_type == 12) {
        // Q4_K (GGML_TYPE_Q4_K)
        dequant_q4k_row(h_row, embd_data, token_id, hidden_dim);
        static bool first = true;
        if (first && debug_kernels_enabled()) {
            fprintf(stderr, "[embd] Q4_K dequant token %d, first 8 values: ", token_id);
            for (int i = 0; i < 8 && i < hidden_dim; i++)
                fprintf(stderr, "%.4f ", h_row[i]);
            fprintf(stderr, "\n");
        }
        first = false;
        for (int i = 0; i < hidden_dim; i++)
            h_fp16[i] = __float2half(h_row[i]);
    } else if (embd_type == 14) {
        // Q6_K (GGML_TYPE_Q6_K) — common for token_embd.weight in Q4_K_M
        // builds because embedding accuracy disproportionately affects
        // overall model quality, so the embedding tensor gets one notch
        // higher precision than the rest of the model.
        dequant_q6k_row(h_row, embd_data, token_id, hidden_dim);
        static bool first = true;
        if (first && debug_kernels_enabled()) {
            fprintf(stderr, "[embd] Q6_K dequant token %d, first 8 values: ", token_id);
            for (int i = 0; i < 8 && i < hidden_dim; i++)
                fprintf(stderr, "%.4f ", h_row[i]);
            fprintf(stderr, "\n");
        }
        first = false;
        for (int i = 0; i < hidden_dim; i++)
            h_fp16[i] = __float2half(h_row[i]);
    } else if (embd_type == 13) {
        // Q5_K — Gemma's token_embd / per_layer_token_embd in Q4_K_M builds.
        dequant_q5k_row(h_row, embd_data, token_id, hidden_dim);
        for (int i = 0; i < hidden_dim; i++)
            h_fp16[i] = __float2half(h_row[i]);
    } else if (embd_type == 0) {
        // F32
        const float* src = (const float*)embd_data + (int64_t)token_id * hidden_dim;
        for (int i = 0; i < hidden_dim; i++)
            h_fp16[i] = __float2half(src[i]);
    } else if (embd_type == 1) {
        // F16: direct copy
        const half* src = (const half*)embd_data + (int64_t)token_id * hidden_dim;
        memcpy(h_fp16, src, hidden_dim * sizeof(half));
    } else {
        // Unsupported quant type — emit a diagnostic so this isn't silent.
        // Falling back to reading raw bytes as FP16 (the previous behavior)
        // would produce garbage and looks like a kernel bug downstream.
        fprintf(stderr,
                "[embd] FATAL: unsupported embedding quant type %d "
                "(supported: 0=F32, 1=F16, 12=Q4_K, 14=Q6_K). "
                "Cannot proceed — model output will be garbage.\n",
                embd_type);
        // Zero out the destination so at least downstream NaNs don't
        // propagate from random memory.
        memset(h_fp16, 0, hidden_dim * sizeof(half));
    }

    (void)stream;
    cudaMemcpy(dst, h_fp16, hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
}

// ═════════════════════════════════════════════════════════════════════════

Engine::Engine() {}
Engine::~Engine() { unload(); }

void Engine::unload() {
    rope_clear_tables();
    if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
    if (decode_graph_)      { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }
    if (d_pos_) { cudaFree(d_pos_); d_pos_ = nullptr; }
    if (stream_)            { cudaStreamDestroy(stream_); stream_ = nullptr; }
    if (host_logits_) {
        cudaFreeHost(host_logits_);
        host_logits_ = nullptr;
        host_logits_capacity_ = 0;
    }
    for (void* p : device_weight_copies_) cudaFree(p);
    device_weight_copies_.clear();
    kv_cache_.destroy();
    scratch_.destroy();
    if (mapped_output_host_base_) {
        clear_mapped_weight_region(mapped_output_host_base_);
        cudaHostUnregister(mapped_output_host_base_);
        mapped_output_host_base_ = nullptr;
        mapped_output_host_bytes_ = 0;
    }
    if (weights_) {
        if (resolve_mapped_weight_device_ptr(weights_)) {
            clear_mapped_weight_region(weights_);
            cudaHostUnregister(weights_);
        } else {
            clear_mapped_weight_region(weights_);
        }
        munmap(weights_, weights_size_);  // now works — #include <sys/mman.h> added
        weights_ = nullptr;
    }
    if (model_weights_.layers) { delete[] model_weights_.layers; model_weights_.layers = nullptr; }
    loaded_ = false;
    graph_captured_ = false;
}

static bool copy_weight_to_device(const void* src, size_t bytes, void** dst) {
    if (!src || bytes == 0) return false;
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[model] cudaMalloc weight copy (%zu bytes) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[model] cudaMemcpy weight copy (%zu bytes) failed: %s\n",
                bytes, cudaGetErrorString(err));
        cudaFree(p);
        return false;
    }
    *dst = p;
    return true;
}

static bool materialize_norm_weights(ModelWeights& mw, int hidden_dim,
                                     int head_dim, int ple_input_dim,
                                     std::vector<void*>& owned) {
    const size_t norm_bytes = (size_t)hidden_dim * sizeof(float);
    const size_t qk_norm_bytes = (size_t)head_dim * sizeof(float);
    const size_t ple_norm_bytes = (size_t)ple_input_dim * sizeof(float);

    // Gemma 4 adds sandwich (post-attn / post-ffn) and PLE norms. Like the
    // base RMSNorms, these are read directly by fused_rmsnorm_residual, so the
    // raw mmap host pointer traps; copy each to device the same way. Helper
    // stages one optional norm and reassigns the pointer in place.
    auto stage = [&](const void*& w, size_t bytes) -> bool {
        if (!w) return true;            // tensor absent (non-Gemma) — leave null
        void* d = nullptr;
        if (!copy_weight_to_device(w, bytes, &d)) return false;
        owned.push_back(d);
        w = d;
        return true;
    };

    for (int l = 0; l < mw.n_layers; l++) {
        auto& lw = mw.layers[l];
        if (lw.rms_type != 0) {
            fprintf(stderr, "[model] unsupported FP16 RMSNorm materialization in layer %d\n", l);
            return false;
        }

        void* attn = nullptr;
        void* ffn = nullptr;
        if (!copy_weight_to_device(lw.rms_attn, norm_bytes, &attn)) return false;
        if (!copy_weight_to_device(lw.rms_ffn, norm_bytes, &ffn)) {
            cudaFree(attn);
            return false;
        }
        owned.push_back(attn);
        owned.push_back(ffn);
        lw.rms_attn = attn;
        lw.rms_ffn = ffn;

        // Gemma 4 sandwich + PLE per-layer norms (all [hidden_dim] F32).
        if (!stage(lw.post_attn_norm, norm_bytes)) return false;
        if (!stage(lw.post_ffn_norm, norm_bytes)) return false;
        if (!stage(lw.ple_norm, norm_bytes)) return false;

        if (lw.q_norm && lw.k_norm) {
            if (lw.qk_norm_type != 0) {
                fprintf(stderr, "[model] unsupported FP16 Q/K RMSNorm materialization in layer %d\n", l);
                return false;
            }
            void* qn = nullptr;
            void* kn = nullptr;
            if (!copy_weight_to_device(lw.q_norm, qk_norm_bytes, &qn)) return false;
            if (!copy_weight_to_device(lw.k_norm, qk_norm_bytes, &kn)) {
                cudaFree(qn);
                return false;
            }
            owned.push_back(qn);
            owned.push_back(kn);
            lw.q_norm = qn;
            lw.k_norm = kn;
        }
    }

    void* out_norm = nullptr;
    if (!copy_weight_to_device(mw.output_norm, norm_bytes, &out_norm)) return false;
    owned.push_back(out_norm);
    mw.output_norm = out_norm;

    // Gemma 4 global PLE projection norm ([ple_input_dim] F32).
    if (!stage(mw.ple_proj_norm, ple_norm_bytes)) return false;

    fprintf(stderr, "[model] Materialized up to %d RMSNorm tensors on device\n",
            mw.n_layers * 4 + 1);
    return true;
}

static bool device_output_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_DEVICE_OUTPUT");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

static bool mapped_weight_device_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_MAPPED_WEIGHTS");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

static bool release_host_weight_pages_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_RELEASE_HOST_WEIGHTS");
        return !mapped_weight_device_enabled() && (!v || strcmp(v, "0") != 0);
    }();
    return enabled;
}

static bool map_host_weight_range_for_gpu(const void* ptr, size_t bytes,
                                          void** host_base_out,
                                          size_t* host_bytes_out) {
    if (!ptr || bytes == 0 || !host_base_out || !host_bytes_out) return false;

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return false;

    const uintptr_t begin = (uintptr_t)ptr & ~((uintptr_t)page_size - 1);
    const uintptr_t end = ((uintptr_t)ptr + bytes + (uintptr_t)page_size - 1) &
                          ~((uintptr_t)page_size - 1);
    if (end <= begin) return false;

    void* host_base = (void*)begin;
    const size_t host_bytes = (size_t)(end - begin);

    (void)cudaSetDeviceFlags(cudaDeviceMapHost);
    (void)cudaGetLastError();

    cudaError_t err = cudaHostRegister(
        host_base, host_bytes,
        cudaHostRegisterPortable | cudaHostRegisterReadOnly | cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[model] CUDA mapped output/embedding registration failed "
                "(%zu MB): %s\n",
                host_bytes / (1024 * 1024), cudaGetErrorString(err));
        return false;
    }

    void* device_base = nullptr;
    err = cudaHostGetDevicePointer(&device_base, host_base, 0);
    if (err != cudaSuccess || !device_base) {
        fprintf(stderr,
                "[model] CUDA mapped output/embedding device pointer failed: %s\n",
                cudaGetErrorString(err));
        cudaHostUnregister(host_base);
        return false;
    }

    register_mapped_weight_region(host_base, device_base, (int64_t)host_bytes);
    *host_base_out = host_base;
    *host_bytes_out = host_bytes;
    fprintf(stderr,
            "[model] CUDA mapped output/embedding weights at %p -> %p (%zu MB)\n",
            host_base, device_base, host_bytes / (1024 * 1024));
    return true;
}

static void release_host_weight_pages(const void* ptr, size_t bytes) {
    if (!release_host_weight_pages_enabled() || !ptr || bytes == 0) return;

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;

    const uintptr_t begin = (uintptr_t)ptr & ~((uintptr_t)page_size - 1);
    const uintptr_t end = ((uintptr_t)ptr + bytes + (uintptr_t)page_size - 1) &
                          ~((uintptr_t)page_size - 1);
    if (end > begin) {
        madvise((void*)begin, end - begin, MADV_DONTNEED);
    }
}

static bool fast_gemv_enabled_for_device_weights() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_FAST_GEMV");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

static int device_layer_limit() {
    static const int limit = [] {
        const char* v = getenv("JLLM_DEVICE_LAYERS");
        if (!v) {
            return mapped_weight_device_enabled() ? 0 : -1;
        }
        int n = atoi(v);
        return n < 0 ? -1 : n;
    }();
    return limit;
}

static int kv_overflow_context(int ctx) {
    const char* v = getenv("JLLM_KV_OVERFLOW");
    if (!v || strcmp(v, "0") == 0) return 0;
    if (strcmp(v, "1") == 0) return ctx / 4;

    int requested = atoi(v);
    if (requested < 0) requested = 0;
    return requested;
}

static size_t kquant_tensor_bytes(int ggml_type, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || cols % 256 != 0) return 0;

    size_t block_bytes = 0;
    switch (ggml_type) {
        case 12: block_bytes = 144; break;  // Q4_K
        case 13: block_bytes = 176; break;  // Q5_K
        case 14: block_bytes = 210; break;  // Q6_K
        default: return 0;
    }

    return (size_t)rows * (size_t)(cols / 256) * block_bytes;
}

static size_t align_up_size(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

static size_t materialize_output_weight(ModelWeights& mw, const ModelConfig& cfg,
                                        const MemoryBudget& budget,
                                        std::vector<void*>& owned,
                                        void** mapped_host_base = nullptr,
                                        size_t* mapped_host_bytes = nullptr) {
    if (!device_output_enabled() || !fast_gemv_enabled_for_device_weights() ||
        !mw.output) {
        return 0;
    }

    const size_t bytes = kquant_tensor_bytes(mw.output_type,
                                             cfg.vocab_size,
                                             cfg.hidden_dim);
    if (bytes == 0) return 0;

    const void* output_src = mw.output;
    const bool tied_output = (output_src == mw.tok_embd);

    if (!mapped_weight_device_enabled() && tied_output &&
        mapped_host_base && mapped_host_bytes) {
        const char* force_copy = getenv("JLLM_DEVICE_OUTPUT_COPY");
        if (!force_copy || strcmp(force_copy, "1") != 0) {
            if (map_host_weight_range_for_gpu(output_src, bytes,
                                              mapped_host_base,
                                              mapped_host_bytes)) {
                fprintf(stderr,
                        "[model] Using CUDA-mapped tied output/embedding "
                        "(%zu MB)\n",
                        bytes / (1024 * 1024));
                return bytes;
            }
        }
    }

    const int64_t mb = (int64_t)((bytes + 1024 * 1024 - 1) / (1024 * 1024));
    if (budget.free_mb() < mb + 128) {
        if (!mapped_weight_device_enabled() &&
            map_host_weight_range_for_gpu(mw.output, bytes,
                                          mapped_host_base,
                                          mapped_host_bytes)) {
            fprintf(stderr,
                    "[model] Using CUDA-mapped tied output/embedding "
                    "(%ld MB requested, %ld MB free budget)\n",
                    mb, budget.free_mb());
            return bytes;
        }
        fprintf(stderr,
                "[model] device output copy skipped (%ld MB requested, %ld MB free budget)\n",
                mb, budget.free_mb());
        return 0;
    }

    void* output_device = nullptr;
    cudaError_t err = cudaMalloc(&output_device, bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        if (!mapped_weight_device_enabled() &&
            map_host_weight_range_for_gpu(output_src, bytes,
                                          mapped_host_base,
                                          mapped_host_bytes)) {
            fprintf(stderr,
                    "[model] Device output copy unavailable; using CUDA-mapped "
                    "tied output/embedding (%zu MB)\n",
                    bytes / (1024 * 1024));
            return bytes;
        }
        fprintf(stderr,
                "[model] device output copy skipped (%zu MB): %s\n",
                bytes / (1024 * 1024), cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(output_device, output_src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[model] device output copy failed (%zu MB): %s\n",
                bytes / (1024 * 1024), cudaGetErrorString(err));
        cudaFree(output_device);
        return 0;
    }

    owned.push_back(output_device);
    mw.output = output_device;
    if (tied_output && fast_embedding_enabled()) {
        mw.tok_embd = output_device;
    }
    if (!tied_output || fast_embedding_enabled()) {
        release_host_weight_pages(output_src, bytes);
    }
    fprintf(stderr,
            "[model] Materialized output projection on device (%zu MB)\n",
            bytes / (1024 * 1024));
    return bytes;
}

static size_t layer_quant_weight_bytes(const LayerWeights& lw,
                                       const ModelConfig& cfg) {
    const int H = cfg.hidden_dim;
    const int Q = cfg.n_heads * cfg.head_dim;
    const int KV = cfg.n_kv_heads * cfg.head_dim;
    const int I = cfg.intermediate_dim;

    size_t bytes = 0;
    bytes += kquant_tensor_bytes(lw.type_wq, Q, H);
    bytes += kquant_tensor_bytes(lw.type_wk, KV, H);
    bytes += kquant_tensor_bytes(lw.type_wv, KV, H);
    bytes += kquant_tensor_bytes(lw.type_wo, H, Q);
    bytes += kquant_tensor_bytes(lw.type_w_gate, I, H);
    bytes += kquant_tensor_bytes(lw.type_w_up, I, H);
    bytes += kquant_tensor_bytes(lw.type_w_down, H, I);
    return bytes;
}

static bool copy_layer_quant_weights_to_device(int layer,
                                               LayerWeights& lw,
                                               const ModelConfig& cfg,
                                               std::vector<void*>& owned) {
    const int H = cfg.hidden_dim;
    const int Q = cfg.n_heads * cfg.head_dim;
    const int KV = cfg.n_kv_heads * cfg.head_dim;
    const int I = cfg.intermediate_dim;

    struct Spec {
        const void* src;
        const void** dst;
        int type;
        int rows;
        int cols;
    };

    Spec specs[] = {
        {lw.wq, &lw.wq, lw.type_wq, Q, H},
        {lw.wk, &lw.wk, lw.type_wk, KV, H},
        {lw.wv, &lw.wv, lw.type_wv, KV, H},
        {lw.wo, &lw.wo, lw.type_wo, H, Q},
        {lw.w_gate, &lw.w_gate, lw.type_w_gate, I, H},
        {lw.w_up, &lw.w_up, lw.type_w_up, I, H},
        {lw.w_down, &lw.w_down, lw.type_w_down, H, I},
    };

    constexpr size_t ARENA_ALIGN = 256;
    constexpr size_t N = sizeof(specs) / sizeof(specs[0]);
    size_t offsets[N] = {};
    size_t sizes[N] = {};
    void* copied[N] = {};
    size_t arena_bytes = 0;

    for (size_t i = 0; i < N; i++) {
        const size_t bytes = kquant_tensor_bytes(specs[i].type,
                                                 specs[i].rows,
                                                 specs[i].cols);
        sizes[i] = bytes;
        if (bytes == 0 || !specs[i].src) {
            return false;
        }
        offsets[i] = arena_bytes;
        arena_bytes += align_up_size(bytes, ARENA_ALIGN);
    }

    void* arena = nullptr;
    fprintf(stderr, "[model] Copying layer %d weights to device arena (%zu MB)\n",
            layer, arena_bytes / (1024 * 1024));
    cudaError_t err = cudaMalloc(&arena, arena_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[model] cudaMalloc layer %d weight arena (%zu MB) failed: %s\n",
                layer, arena_bytes / (1024 * 1024), cudaGetErrorString(err));
        return false;
    }

    uint8_t* arena_u8 = (uint8_t*)arena;
    for (size_t i = 0; i < N; i++) {
        copied[i] = arena_u8 + offsets[i];
        err = cudaMemcpy(copied[i], specs[i].src, sizes[i],
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "[model] cudaMemcpy layer weight copy (%zu bytes) failed: %s\n",
                    sizes[i], cudaGetErrorString(err));
            cudaFree(arena);
            return false;
        }
    }

    for (size_t i = 0; i < N; i++) {
        *specs[i].dst = copied[i];
        release_host_weight_pages(specs[i].src, sizes[i]);
    }
    owned.push_back(arena);
    return true;
}

static size_t materialize_layer_weights(ModelWeights& mw,
                                        const ModelConfig& cfg,
                                        const MemoryBudget& budget,
                                        std::vector<void*>& owned) {
    if (!fast_gemv_enabled_for_device_weights()) return 0;

    const int requested_layers = device_layer_limit();
    if (requested_layers == 0) return 0;

    const int layer_cap = requested_layers < 0
        ? mw.n_layers
        : std::min(requested_layers, mw.n_layers);
    if (!mapped_weight_device_enabled() && layer_cap < mw.n_layers) {
        fprintf(stderr,
                "[model] FATAL: JLLM_MAPPED_WEIGHTS=0 requires all %d layers "
                "to be device-resident; requested only %d\n",
                mw.n_layers, layer_cap);
        return (size_t)-1;
    }
    const int64_t reserve_mb = 128;
    int64_t free_mb = budget.free_mb();
    size_t total_bytes = 0;
    int copied_layers = 0;

    for (int l = 0; l < layer_cap; l++) {
        auto& lw = mw.layers[l];
        const size_t bytes = layer_quant_weight_bytes(lw, cfg);
        if (bytes == 0) break;

        const int64_t mb =
            (int64_t)((bytes + 1024 * 1024 - 1) / (1024 * 1024));
        if (free_mb < mb + reserve_mb) break;

        if (!copy_layer_quant_weights_to_device(l, lw, cfg, owned)) break;

        free_mb -= mb;
        total_bytes += bytes;
        copied_layers++;
    }

    if (copied_layers > 0) {
        fprintf(stderr,
                "[model] Materialized %d transformer layers on device (%zu MB)\n",
                copied_layers, total_bytes / (1024 * 1024));
        if (release_host_weight_pages_enabled()) {
            fprintf(stderr,
                    "[model] Released copied GGUF weight pages from host cache "
                    "(JLLM_RELEASE_HOST_WEIGHTS=1)\n");
        }
    }
    if (!mapped_weight_device_enabled() && copied_layers < mw.n_layers) {
        fprintf(stderr,
                "[model] FATAL: only materialized %d / %d transformer layers "
                "with JLLM_MAPPED_WEIGHTS=0\n",
                copied_layers, mw.n_layers);
        return (size_t)-1;
    }
    return total_bytes;
}

bool Engine::load(const std::string& gguf_path, const GenParams& params) {
    gen_params_ = params;
    budget_ = probe_system_memory();
    gguf_path_ = gguf_path;

    // Path F: cheap content fingerprint for KV cache validation. Failure
    // here is non-fatal — we just don't write/match KV cache files.
    if (!compute_model_fingerprint(gguf_path, model_fingerprint_)) {
        std::memset(model_fingerprint_, 0, sizeof(model_fingerprint_));
    }

    config_ = load_gguf_config(gguf_path);
    fprintf(stderr, "[engine] %s: %d layers, %d heads (%d KV), dim=%d, head_dim=%d, vocab=%d, rms_eps=%g, rope=%s\n",
            config_.name.c_str(), config_.n_layers, config_.n_heads,
            config_.n_kv_heads, config_.hidden_dim, config_.head_dim,
            config_.vocab_size,
            config_.rms_eps, config_.rope_neox ? "neox" : "normal");

    // Gemma 4 (#88): experimental dense forward + PLE (#92/#93). Sliding-window
    // masking is not applied yet, so contexts beyond sliding_window (512) will
    // diverge; short prompts are unaffected.
    if (config_.arch == Arch::Gemma4) {
        fprintf(stderr,
                "[engine] Gemma 4: %d layers, head_dim sliding=%d/global=%d, "
                "kv_shared=%d, ple=%d, softcap=%g, sliding_window=%d.\n",
                config_.n_layers, config_.sliding_head_dim, config_.global_head_dim,
                config_.n_kv_shared_layers, config_.ple_input_dim,
                config_.spec.final_logit_softcap, config_.sliding_window);
        // Gemma's V activations have large-dynamic-range outliers (massive
        // activations ~10x RMS). Per-row INT8 KV quantization collapses the
        // small-but-significant V components and wrecks attention, so force
        // FP16 KV for Gemma regardless of the requested kv_int8 default.
        if (gen_params_.kv_int8) {
            fprintf(stderr, "[engine] Gemma 4: forcing FP16 KV cache "
                            "(INT8 KV too lossy for Gemma V distribution)\n");
            gen_params_.kv_int8 = false;
        }
    }

    if (!load_and_map_weights(gguf_path, &weights_, &weights_size_,
                              &model_weights_, config_)) {
        fprintf(stderr, "[engine] Failed to load/map weights\n");
        return false;
    }
    if (!materialize_norm_weights(model_weights_, config_.hidden_dim,
                                  config_.head_dim, config_.ple_input_dim,
                                  device_weight_copies_)) {
        fprintf(stderr, "[engine] Failed to materialize norm weights\n");
        return false;
    }
    // Gemma 4 applies a weightless per-head RMSNorm to V before the KV store.
    // Build a [head_dim] device vector of ones to serve as the unit weight.
    if (config_.arch == Arch::Gemma4) {
        std::vector<float> ones(config_.head_dim, 1.0f);
        void* d_ones = nullptr;
        if (!copy_weight_to_device(ones.data(),
                                   ones.size() * sizeof(float), &d_ones)) {
            fprintf(stderr, "[engine] Failed to alloc Gemma V-norm ones\n");
            return false;
        }
        device_weight_copies_.push_back(d_ones);
        gemma_vnorm_ones_ = (float*)d_ones;
    }
    budget_.model_mb = mapped_weight_device_enabled()
        ? weights_size_ / (1024 * 1024)
        : 0;

    int kv_bytes = gen_params_.kv_int8 ? 1 : 2;
    // Gemma 4 shares KV across the trailing n_kv_shared_layers layers: only the
    // first (n_layers - n_kv_shared_layers) layers hold their own K/V, and the
    // shared layers read those slots (transformer_layer_gemma4's kv_layer is
    // always < this count). Sizing the pool to the producing-layer count frees
    // ~57% of KV memory on Gemma 4 E2B (15 of 35 layers) and lets the context
    // budget stretch correspondingly. Other arches keep all n_layers.
    const int kv_pool_layers =
        (config_.arch == Arch::Gemma4 && config_.n_kv_shared_layers > 0)
            ? (config_.n_layers - config_.n_kv_shared_layers)
            : config_.n_layers;
    int auto_ctx = budget_.max_context(kv_pool_layers, config_.n_kv_heads, config_.head_dim, kv_bytes);

    // Cap the implicit auto-context to something that leaves real headroom on
    // 8 GB Jetsons. Qwen3-4B uses 128-dim K/V heads, so 8192 tokens allocates
    // ~1.44 GB of KV (including overflow) after a 2.3 GB mmap'd model and can
    // push the budget negative. Users who explicitly pass `-c <N>` bypass this
    // cap and are responsible for fitting in memory.
    const int DEFAULT_AUTO_CONTEXT_CAP = (config_.head_dim >= 128) ? 4096 : 8192;
    if (params.context_limit > 0) {
        // User explicitly asked for a context size — honor it, but still
        // clamp to the model's max_seq_len.
        int requested = std::min(params.context_limit, config_.max_seq_len);
        int ctx = std::min(requested, auto_ctx);
        fprintf(stderr,
                "[engine] Context: %d tokens (user-requested, memory cap %d, model max %d)\n",
                ctx, auto_ctx, config_.max_seq_len);
        auto_ctx = ctx; // for subsequent use below
    } else {
        // Auto-pick a safe default.
        int ctx = std::min({auto_ctx, config_.max_seq_len, DEFAULT_AUTO_CONTEXT_CAP});
        fprintf(stderr,
                "[engine] Context: %d tokens (auto-capped to %d; memory cap %d, model max %d). "
                "Override with -c <N>.\n",
                ctx, DEFAULT_AUTO_CONTEXT_CAP, auto_ctx, config_.max_seq_len);
        auto_ctx = ctx;
    }
    int ctx = auto_ctx; // keep `ctx` for the rest of the function

    size_t output_device_bytes = 0;
    size_t layer_device_bytes = 0;

    KVCachePool::Config kv_cfg = {};
    kv_cfg.n_layers = kv_pool_layers;
    kv_cfg.n_kv_heads = config_.n_kv_heads;
    kv_cfg.head_dim = config_.head_dim;
    kv_cfg.max_context = ctx;
    kv_cfg.overflow_context = kv_overflow_context(ctx);
    kv_cfg.kv_type_bytes = kv_bytes;
    if (!kv_cache_.init(kv_cfg)) return false;
    budget_.kv_cache_mb = kv_cache_.capacity_bytes() / (1024 * 1024);

    int64_t scratch_bytes = 0;
    const int q_dim = config_.n_heads * config_.head_dim;
    const int kv_dim = config_.n_kv_heads * config_.head_dim;
    scratch_bytes += config_.hidden_dim * sizeof(half) * 8;
    scratch_bytes += q_dim * sizeof(half) * 2;   // q, attention output
    scratch_bytes += kv_dim * sizeof(half) * 2;  // k, v
    scratch_bytes += config_.intermediate_dim * sizeof(half) * 4;
    scratch_bytes += config_.vocab_size * sizeof(float);
    scratch_bytes = std::max(scratch_bytes, (int64_t)64 * 1024 * 1024);
    // Size for batched prefill of up to BATCH_CAP tokens so common-length
    // prompts take the fast layer-major path instead of per-token fallback.
    // Per-token batched footprint mirrors transformer_prefill's buffers
    // (5H + 2Q + 2KV + 3I), plus Gemma's persistent per-token PLE-input table.
    {
        const int64_t BATCH_CAP = std::min<int64_t>(ctx > 0 ? ctx : 2048, 1024);
        int64_t per_tok = (int64_t)(5 * config_.hidden_dim + 2 * q_dim +
                                    2 * kv_dim + 3 * config_.intermediate_dim) *
                          sizeof(half);
        if (config_.arch == Arch::Gemma4) {
            // Persistent per-token PLE-input table + the per-layer batched PLE
            // transient (ple_in_layer + inp_gate out + gated + proj).
            per_tok += (int64_t)config_.n_layers * config_.ple_input_dim *
                       sizeof(half);
            per_tok += (int64_t)(3 * config_.ple_input_dim + config_.hidden_dim) *
                       sizeof(half);
        }
        const int64_t batched_scratch = BATCH_CAP * per_tok +
                                        (int64_t)config_.vocab_size *
                                            sizeof(float) +
                                        8 * 1024 * 1024;
        scratch_bytes = std::max(scratch_bytes, batched_scratch);
    }
    scratch_bytes = (scratch_bytes + 4095) & ~4095LL;

    if (!scratch_.init(scratch_bytes)) return false;
    budget_.scratch_mb = scratch_bytes / (1024 * 1024);

    if (!mapped_weight_device_enabled()) {
        // Full device-resident mode has the tightest memory envelope on 8 GB
        // Jetsons. Give mandatory runtime buffers first claim on CUDA memory,
        // then copy transformer layers and release copied GGUF pages as we go.
        layer_device_bytes =
            materialize_layer_weights(model_weights_, config_, budget_,
                                      device_weight_copies_);
        if (layer_device_bytes == (size_t)-1) {
            return false;
        }
        budget_.model_mb += layer_device_bytes / (1024 * 1024);

        if (weights_) {
            madvise(weights_, weights_size_, MADV_DONTNEED);
        }

        output_device_bytes =
            materialize_output_weight(model_weights_, config_, budget_,
                                      device_weight_copies_,
                                      &mapped_output_host_base_,
                                      &mapped_output_host_bytes_);
        if (output_device_bytes == 0) {
            fprintf(stderr,
                    "[model] FATAL: JLLM_MAPPED_WEIGHTS=0 requires the "
                    "embedding/output projection to be device-resident or "
                    "CUDA-mapped\n");
            return false;
        }
        budget_.model_mb += output_device_bytes / (1024 * 1024);
    } else {
        output_device_bytes =
            materialize_output_weight(model_weights_, config_, budget_,
                                      device_weight_copies_);
        budget_.model_mb += output_device_bytes / (1024 * 1024);

        layer_device_bytes =
            materialize_layer_weights(model_weights_, config_, budget_,
                                      device_weight_copies_);
        if (layer_device_bytes == (size_t)-1) {
            return false;
        }
        budget_.model_mb += layer_device_bytes / (1024 * 1024);
    }

    cudaStreamCreate(&stream_);
    tokenizer_.load_from_gguf(gguf_path);

    cudaError_t logits_err = cudaMallocHost((void**)&host_logits_,
                                            config_.vocab_size * sizeof(float));
    if (logits_err != cudaSuccess) {
        fprintf(stderr, "[engine] cudaMallocHost logits buffer failed: %s\n",
                cudaGetErrorString(logits_err));
        return false;
    }
    host_logits_capacity_ = config_.vocab_size;

    // 4-byte device buffer that holds the current decode position; rewritten
    // before each captured-graph replay and read by the *_dyn rope/attention
    // kernels. Allocated even when JLLM_DECODE_GRAPH is off — costs nothing.
    cudaError_t pos_err = cudaMalloc((void**)&d_pos_, sizeof(int));
    if (pos_err != cudaSuccess) {
        fprintf(stderr, "[engine] cudaMalloc d_pos_ (4 B) failed: %s\n",
                cudaGetErrorString(pos_err));
        return false;
    }

    // Precompute RoPE cos/sin tables. One table per unique (theta_base, head_dim)
    // pair. The rope_inplace* kernels automatically use the table instead of
    // recomputing powf + cosf + sinf per decode step.
    if (config_.spec.dual_rope) {
        // Gemma 4: two theta bases, two head dims (sliding and global).
        rope_precompute_table(config_.max_seq_len,
                              config_.sliding_head_dim, config_.rope_theta_swa);
        rope_precompute_table(config_.max_seq_len,
                              config_.global_head_dim,  config_.rope_theta);
    } else {
        rope_precompute_table(config_.max_seq_len, config_.head_dim, config_.rope_theta);
    }

    budget_.print();
    loaded_ = true;
    return true;
}

// ── Single transformer layer ─────────────────────────────────────────────
// BUG #2 FIX: proper residual chaining:
//   residual1 = x (input to layer)
//   attn_out  = attention(RMSNorm(x))
//   x2        = residual1 + attn_out          ← first residual add
//   ffn_out   = FFN(RMSNorm(x2))
//   x_out     = x2 + ffn_out                  ← second residual add

// ── Arch-recipe dispatch helpers (#89) ───────────────────────────────────
// The forward pass reads these instead of hardcoded constants so a new model
// family is a recipe, not a fork of the hot path. For the default (Llama/Qwen)
// recipe they reproduce the original behavior exactly (byte-identical).

// FFN activation: SwiGLU (Llama/Qwen) vs GeGLU (Gemma).
static inline void ffn_activation(const ArchSpec& spec, half* out,
                                  const half* gate, const half* up,
                                  int rows, int dim, cudaStream_t stream) {
    if (spec.ffn_activation == FfnActivation::GeluGeGLU) {
        fused_geglu(out, gate, up, rows, dim, stream);
    } else {
        fused_swiglu(out, gate, up, rows, dim, stream);
    }
}

// Attention softmax scale: an explicit recipe override (Gemma uses 1.0), else
// the default 1/sqrt(head_dim).
static inline float attention_scale(const ArchSpec& spec, int head_dim) {
    return spec.attn_scale > 0.0f ? spec.attn_scale
                                  : 1.0f / sqrtf((float)head_dim);
}

// Embedding scale: Gemma multiplies token embeddings by sqrt(hidden). `n_elems`
// is the total element count (n_tokens * hidden). No-op for the default recipe.
static inline void scale_embedding(const ArchSpec& spec, half* x, int n_elems,
                                   int hidden, cudaStream_t stream) {
    if (spec.scale_embeddings) {
        vec_scale(x, n_elems, sqrtf((float)hidden), stream);
    }
}

// Final-logit soft-cap (Gemma): x = c*tanh(x/c). No-op for the default recipe.
static inline void apply_logit_softcap(const ArchSpec& spec, float* logits,
                                       int n, cudaStream_t stream) {
    if (spec.final_logit_softcap > 0.0f) {
        logit_softcap(logits, n, spec.final_logit_softcap, stream);
    }
}

void Engine::transformer_layer(int layer, int pos, half* x) {
    const auto& lw = model_weights_.layers[layer];
    int H = config_.hidden_dim;
    int Q_DIM = config_.n_heads * config_.head_dim;
    int KV_DIM = config_.n_kv_heads * config_.head_dim;

    half* normed   = (half*)scratch_.get(H * sizeof(half));
    half* q_buf    = (half*)scratch_.get(Q_DIM * sizeof(half));
    half* k_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* v_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));

    // Debug: check x values before first norm
    static int layer_dbg = 0;
    if (debug_kernels_enabled() && layer_dbg < 1) {
        cudaStreamSynchronize(stream_);
        half h_x[8];
        cudaMemcpy(h_x, x, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[layer %d] x before norm: ", layer);
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_x[i]));
        fprintf(stderr, "\n");

        if (lw.rms_attn) {
            float w_check[4];
            bool nfp32 = (lw.rms_type == 0);
            if (nfp32) {
                cudaMemcpy(w_check, lw.rms_attn, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                half h_w[4];
                cudaMemcpy(h_w, lw.rms_attn, 4 * sizeof(half), cudaMemcpyDeviceToHost);
                for (int i = 0; i < 4; i++) w_check[i] = __half2float(h_w[i]);
            }
            fprintf(stderr, "[layer %d] norm weight (fp32=%d): %.4f %.4f %.4f %.4f\n",
                    layer, nfp32, w_check[0], w_check[1], w_check[2], w_check[3]);
        }
        layer_dbg++;
    }

    // 1. Pre-attention RMSNorm: normed = RMSNorm(x) * weight
    bool norm_fp32 = (lw.rms_type == 0);  // 0=F32, 1=F16
    fused_rmsnorm_residual(normed, x, nullptr, lw.rms_attn, 1, H, config_.rms_eps, norm_fp32, stream_);

    if (debug_kernels_enabled() && layer_dbg <= 1 && layer == 0) {
        cudaStreamSynchronize(stream_);
        half h_n[8];
        cudaMemcpy(h_n, normed, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[layer %d] normed after RMSNorm: ", layer);
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_n[i]));
        fprintf(stderr, "\n");
    }

    // 2. QKV projections. These share the same input vector, so dispatch as
    // one combined GEMV launch when the fast K-quant path is available.
    gemv_quant_triple(q_buf, lw.wq, lw.type_wq, Q_DIM,
                      k_buf, lw.wk, lw.type_wk, KV_DIM,
                      v_buf, lw.wv, lw.type_wv, KV_DIM,
                      normed, H, stream_);

    // 3a. Attention block (QK-norm, RoPE, KV store, attention, Wo, residual #1).
    int I = config_.intermediate_dim;
    half* x2        = (half*)scratch_.get(H * sizeof(half));
    transformer_layer_attn_block(layer, pos, x, q_buf, k_buf, v_buf,
                                 x2, /*qk_norm_already=*/false);

    // 3b. FFN front (norm + gate/up + swiglu) per-token.
    half* normed2    = (half*)scratch_.get(H * sizeof(half));
    half* gate_buf   = (half*)scratch_.get(I * sizeof(half));
    half* up_buf     = (half*)scratch_.get(I * sizeof(half));
    half* swiglu_out = (half*)scratch_.get(I * sizeof(half));

    fused_rmsnorm_residual(normed2, x2, nullptr, lw.rms_ffn, 1, H, config_.rms_eps, norm_fp32, stream_);
    gemv_quant_pair(gate_buf, lw.w_gate, lw.type_w_gate, I,
                    up_buf,   lw.w_up,   lw.type_w_up,   I,
                    normed2, H, stream_);
    ffn_activation(config_.spec, swiglu_out, gate_buf, up_buf, 1, I, stream_);

    // 3c. FFN exit: x = x2 + W_down(swiglu_out).
    transformer_layer_ffn_block(layer, x2, swiglu_out, x);
}

// Gemma 4 decoder layer (PLE-only architecture). Differs from the Llama/Qwen
// path: per-layer head dim (256 sliding / 512 global), per-layer RoPE base,
// a 4-norm "sandwich" (the post-attention and post-ffn norms apply to the
// sub-layer OUTPUT before the residual add), GeGLU + double-wide MLP, and a
// learned per-layer output scalar. Per-layer head dim is threaded through the
// projections/RoPE/attention; the KV cache is sized for the max head dim (512)
// so sliding layers use the first 256 of each slot. Sliding-window masking is
// a no-op for prompts <= sliding_window (512) and lands later; PLE is #93.
void Engine::transformer_layer_gemma4(int layer, int pos, half* x) {
    const auto& lw = model_weights_.layers[layer];
    const int H  = config_.hidden_dim;
    const int hd = lw.head_dim_l;                 // 256 sliding / 512 global
    const int Q_DIM  = config_.n_heads * hd;
    const int KV_DIM = config_.n_kv_heads * hd;
    const int I  = lw.intermediate_l;             // 6144 / 12288
    const bool norm_fp32 = (lw.rms_type == 0);

    half* normed   = (half*)scratch_.get(H * sizeof(half));
    half* q_buf    = (half*)scratch_.get(Q_DIM * sizeof(half));
    half* k_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* v_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* attn_out = (half*)scratch_.get(Q_DIM * sizeof(half));

    // 1. Input RMSNorm (attn_norm).
    fused_rmsnorm_residual(normed, x, nullptr, lw.rms_attn, 1, H,
                           config_.rms_eps, norm_fp32, stream_);

    // 2. QKV projections (per-layer head dim). Gemma 4 shares KV across the
    //    last n_kv_shared_layers layers: layers [kv_from_start, n_layers) do
    //    NOT compute their own K/V — they reuse the cache of the last
    //    KV-producing layer of the same attention type (sliding ->
    //    kv_from_start-2, global -> kv_from_start-1), matching llama.cpp's
    //    has_kv() mapping (n_layer_kv_from_start = n_layer - shared_kv_layers).
    const int kv_from_start = config_.n_layers - config_.n_kv_shared_layers;
    const bool has_own_kv = (layer < kv_from_start);
    const int kv_layer = has_own_kv
        ? layer
        : (lw.is_sliding ? kv_from_start - 2 : kv_from_start - 1);

    if (has_own_kv) {
        gemv_quant_triple(q_buf, lw.wq, lw.type_wq, Q_DIM,
                          k_buf, lw.wk, lw.type_wk, KV_DIM,
                          v_buf, lw.wv, lw.type_wv, KV_DIM,
                          normed, H, stream_);
    } else {
        gemv_quant(q_buf, lw.wq, lw.type_wq, normed, Q_DIM, H, stream_);
    }

    // 3a. Per-head Q RMSNorm (always). K-norm + weightless V RMSNorm
    //     (ggml_rms_norm; V is not RoPE'd) only when this layer owns its K/V.
    if (lw.q_norm) {
        fused_rmsnorm_residual(q_buf, q_buf, nullptr, lw.q_norm,
                               config_.n_heads, hd, config_.rms_eps,
                               lw.qk_norm_type == 0, stream_);
    }
    if (has_own_kv) {
        if (lw.k_norm) {
            fused_rmsnorm_residual(k_buf, k_buf, nullptr, lw.k_norm,
                                   config_.n_kv_heads, hd, config_.rms_eps,
                                   lw.qk_norm_type == 0, stream_);
        }
        if (gemma_vnorm_ones_) {
            fused_rmsnorm_residual(v_buf, v_buf, nullptr, gemma_vnorm_ones_,
                                   config_.n_kv_heads, hd, config_.rms_eps,
                                   /*weight_fp32=*/true, stream_);
        }
    }

    // 3b. RoPE (per-layer base, full rotary over hd) + KV store. Shared-KV
    //     layers rotate Q only — their K/V already live in kv_layer's cache.
    if (!has_own_kv) {
        rope_inplace(q_buf, nullptr, config_.n_heads, 0, hd, pos,
                     lw.rope_theta_l, config_.rope_neox, stream_);
    } else if (!gen_params_.kv_int8 && kv_cache_.is_fast_position(pos)) {
        rope_inplace_store_kv_fp16(q_buf, k_buf, v_buf,
                                   (half*)kv_cache_.key_ptr(layer, pos),
                                   (half*)kv_cache_.val_ptr(layer, pos),
                                   config_.n_heads, config_.n_kv_heads, hd, pos,
                                   lw.rope_theta_l, config_.rope_neox, stream_);
    } else {
        rope_inplace(q_buf, k_buf, config_.n_heads, config_.n_kv_heads, hd, pos,
                     lw.rope_theta_l, config_.rope_neox, stream_);
        if (gen_params_.kv_int8) {
            float* ks = kv_cache_.kv_scale_ptr(layer, pos, false);
            float* vs = kv_cache_.kv_scale_ptr(layer, pos, true);
            fp16_to_int8((int8_t*)kv_cache_.key_ptr(layer, pos), ks, k_buf,
                         config_.n_kv_heads, hd, stream_);
            fp16_to_int8((int8_t*)kv_cache_.val_ptr(layer, pos), vs, v_buf,
                         config_.n_kv_heads, hd, stream_);
        } else {
            cudaMemcpyAsync(kv_cache_.key_ptr(layer, pos), k_buf,
                            KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
            cudaMemcpyAsync(kv_cache_.val_ptr(layer, pos), v_buf,
                            KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
        }
    }

    // 3c. Flash attention (recipe scale = 1.0 for Gemma). Reads K/V from
    //     kv_layer (== layer for KV-owning layers, else the shared source).
    //     Sliding layers mask to the last sliding_window keys; global = full.
    float scale = attention_scale(config_.spec, hd);
    const int attn_window = lw.is_sliding ? config_.sliding_window : 0;
    float* k_scales = gen_params_.kv_int8 ? kv_cache_.kv_scale_ptr(kv_layer, 0, false) : nullptr;
    float* v_scales = gen_params_.kv_int8 ? kv_cache_.kv_scale_ptr(kv_layer, 0, true)  : nullptr;
    flash_attention_decode(attn_out, q_buf, kv_cache_.key_ptr(kv_layer, 0),
                           kv_cache_.val_ptr(kv_layer, 0), config_.n_heads,
                           config_.n_kv_heads, hd, config_.head_dim, pos + 1, scale,
                           gen_params_.kv_int8, k_scales, v_scales, attn_window, stream_);

    // 3d. Output projection -> post-attention (sandwich) norm -> residual.
    half* wo_out = (half*)scratch_.get(H * sizeof(half));
    half* x2     = (half*)scratch_.get(H * sizeof(half));
    gemv_quant(wo_out, lw.wo, lw.type_wo, attn_out, H, Q_DIM, stream_);
    fused_rmsnorm_residual(wo_out, wo_out, nullptr, lw.post_attn_norm, 1, H,
                           config_.rms_eps, /*weight_fp32=*/true, stream_);
    vec_add(x2, x, wo_out, H, stream_);

    // 4. FFN: pre-norm -> GeGLU (double-wide) -> down -> post-ffn (sandwich) norm -> residual.
    half* normed2 = (half*)scratch_.get(H * sizeof(half));
    half* gate    = (half*)scratch_.get(I * sizeof(half));
    half* up      = (half*)scratch_.get(I * sizeof(half));
    half* act     = (half*)scratch_.get(I * sizeof(half));
    half* ffn_out = (half*)scratch_.get(H * sizeof(half));
    fused_rmsnorm_residual(normed2, x2, nullptr, lw.rms_ffn, 1, H,
                           config_.rms_eps, norm_fp32, stream_);
    gemv_quant_pair(gate, lw.w_gate, lw.type_w_gate, I,
                    up,   lw.w_up,   lw.type_w_up,   I, normed2, H, stream_);
    fused_geglu(act, gate, up, 1, I, stream_);
    gemv_quant(ffn_out, lw.w_down, lw.type_w_down, act, H, I, stream_);
    fused_rmsnorm_residual(ffn_out, ffn_out, nullptr, lw.post_ffn_norm, 1, H,
                           config_.rms_eps, /*weight_fp32=*/true, stream_);
    vec_add(x, x2, ffn_out, H, stream_);

    // 5. Per-Layer Embeddings (PLE): inp_gate -> gelu(gate)*ple_input -> proj
    //    -> post-norm -> residual. ple_input is computed once per token before
    //    the layer loop (compute_gemma_ple_input); each layer reads its slice.
    if (lw.ple_gate && gemma_ple_input_) {
        const int D = config_.ple_input_dim;  // 256
        half* g_in = (half*)scratch_.get(D * sizeof(half));
        half* g    = (half*)scratch_.get(D * sizeof(half));
        half* p    = (half*)scratch_.get(H * sizeof(half));
        // inp_gate / proj are F32 (not K-quant), so use the dense GEMV.
        gemv_dense(g_in, lw.ple_gate, lw.type_ple_gate, x, D, H, stream_);
        // fused_geglu(out, gate, up) = gelu_tanh(gate) * up.
        fused_geglu(g, g_in, gemma_ple_input_ + (int64_t)layer * D, 1, D, stream_);
        gemv_dense(p, lw.ple_proj, lw.type_ple_proj, g, H, D, stream_);
        fused_rmsnorm_residual(p, p, nullptr, lw.ple_norm, 1, H,
                               config_.rms_eps, /*weight_fp32=*/true, stream_);
        vec_add(x, x, p, H, stream_);
    }

    // 6. Per-layer learned output scale.
    if (lw.layer_scale_val != 1.0f) {
        vec_scale(x, H, lw.layer_scale_val, stream_);
    }
}

// Gemma 4 PLE: per-token per-layer input = (token_identity + context) / sqrt(2),
// where token_identity = sqrt(ple_dim) * per_layer_token_embd[token] and
// context = per_layer_proj_norm( (per_layer_model_proj @ embed) / sqrt(hidden) ),
// both reshaped to [n_layers, ple_dim]. Computed once per token; each decoder
// layer multiplies its gelu-gated hidden by its [ple_dim] slice.
void Engine::compute_gemma_ple_input(const half* inputs_embeds, int token,
                                     half* out) {
    const auto& mw = model_weights_;
    if (!mw.ple_embd || !mw.ple_model_proj || !mw.ple_proj_norm) return;
    const int H = config_.hidden_dim;
    const int D = config_.ple_input_dim;        // 256
    const int L = config_.n_layers;             // 35
    const int total = L * D;                     // 8960

    // token_identity = per_layer_token_embd[token] * sqrt(D)  (dequant one row).
    half* tid = (half*)scratch_.get(total * sizeof(half));
    dequant_embedding(tid, mw.ple_embd, token, total, mw.ple_embd_type, stream_);
    vec_scale(tid, total, sqrtf((float)D), stream_);

    // context = per_layer_proj_norm( (model_proj @ embed) / sqrt(H) ), per D-row.
    // per_layer_model_proj is BF16 (not K-quant), so use the dense GEMV.
    gemv_dense(out, mw.ple_model_proj, mw.ple_model_proj_type, inputs_embeds,
               total, H, stream_);
    vec_scale(out, total, 1.0f / sqrtf((float)H), stream_);
    fused_rmsnorm_residual(out, out, nullptr, mw.ple_proj_norm, L, D,
                           config_.rms_eps, /*weight_fp32=*/true, stream_);

    // out = (context + token_identity) * (1/sqrt(2)).
    vec_add(out, out, tid, total, stream_);
    vec_scale(out, total, 0.70710678f, stream_);
}

void Engine::transformer_layer_attn_compute(int layer, int pos,
                                            half* q_buf, half* k_buf, half* v_buf,
                                            half* attn_out, bool qk_norm_already) {
    const auto& lw = model_weights_.layers[layer];
    int KV_DIM = config_.n_kv_heads * config_.head_dim;

    if (lw.q_norm && lw.k_norm && !qk_norm_already) {
        fused_rmsnorm_residual(q_buf, q_buf, nullptr, lw.q_norm,
                               config_.n_heads, config_.head_dim,
                               config_.rms_eps, lw.qk_norm_type == 0, stream_);
        fused_rmsnorm_residual(k_buf, k_buf, nullptr, lw.k_norm,
                               config_.n_kv_heads, config_.head_dim,
                               config_.rms_eps, lw.qk_norm_type == 0, stream_);
    }

    if (!gen_params_.kv_int8 && kv_cache_.is_fast_position(pos)) {
        rope_inplace_store_kv_fp16(q_buf, k_buf, v_buf,
                                  (half*)kv_cache_.key_ptr(layer, pos),
                                  (half*)kv_cache_.val_ptr(layer, pos),
                                  config_.n_heads, config_.n_kv_heads,
                                  config_.head_dim, pos, config_.rope_theta,
                                  config_.rope_neox, stream_);
    } else {
        rope_inplace(q_buf, k_buf, config_.n_heads, config_.n_kv_heads,
                     config_.head_dim, pos, config_.rope_theta,
                     config_.rope_neox, stream_);
    }

    if (gen_params_.kv_int8) {
        // Path I phase I2 (#62): convert FP16 → INT8 with one absmax
        // scale per kv_head (n_kv_heads rows × head_dim cols), and
        // capture those scales into the per-(layer, pos) slot in
        // KVCachePool. Attention dequant in I3 reads these scales.
        // Pre-I2 the engine passed nullptr for scales and rows=1,
        // which both lost the per-head info AND was undefined behavior
        // in the kernel.
        float* k_scales = kv_cache_.kv_scale_ptr(layer, pos, /*is_value=*/false);
        float* v_scales = kv_cache_.kv_scale_ptr(layer, pos, /*is_value=*/true);
        fp16_to_int8((int8_t*)kv_cache_.key_ptr(layer, pos), k_scales,
                     k_buf, config_.n_kv_heads, config_.head_dim, stream_);
        fp16_to_int8((int8_t*)kv_cache_.val_ptr(layer, pos), v_scales,
                     v_buf, config_.n_kv_heads, config_.head_dim, stream_);
    } else if (!kv_cache_.is_fast_position(pos)) {
        cudaMemcpyAsync(kv_cache_.key_ptr(layer, pos), k_buf,
                        KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
        cudaMemcpyAsync(kv_cache_.val_ptr(layer, pos), v_buf,
                        KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
    }

    float scale = attention_scale(config_.spec, config_.head_dim);
    // Path I3 (#62): per-(pos, kv_head) scale regions for the current
    // layer. nullptr in FP16 mode (matches the kernel's null-check).
    float* k_scales = gen_params_.kv_int8 ? kv_cache_.kv_scale_ptr(layer, 0, /*is_value=*/false) : nullptr;
    float* v_scales = gen_params_.kv_int8 ? kv_cache_.kv_scale_ptr(layer, 0, /*is_value=*/true)  : nullptr;
    flash_attention_decode(attn_out, q_buf, kv_cache_.key_ptr(layer, 0),
                          kv_cache_.val_ptr(layer, 0),
                          config_.n_heads, config_.n_kv_heads, config_.head_dim,
                          config_.head_dim, pos + 1, scale, gen_params_.kv_int8,
                          k_scales, v_scales, /*window=*/0, stream_);
}

void Engine::transformer_layer_attn_block(int layer, int pos, half* x_in,
                                          half* q_buf, half* k_buf, half* v_buf,
                                          half* x_attn_out, bool qk_norm_already) {
    const auto& lw = model_weights_.layers[layer];
    int H = config_.hidden_dim;
    int Q_DIM = config_.n_heads * config_.head_dim;

    half* attn_out  = (half*)scratch_.get(Q_DIM * sizeof(half));

    transformer_layer_attn_compute(layer, pos, q_buf, k_buf, v_buf,
                                   attn_out, qk_norm_already);
    gemv_quant_add(x_attn_out, lw.wo, lw.type_wo, attn_out, x_in,
                   H, Q_DIM, stream_);
}

void Engine::transformer_layer_ffn_block(int layer, half* x_attn, half* swiglu_in,
                                         half* x_out) {
    const auto& lw = model_weights_.layers[layer];
    int H = config_.hidden_dim;
    int I = config_.intermediate_dim;

    gemv_quant_add(x_out, lw.w_down, lw.type_w_down, swiglu_in, x_attn,
                   H, I, stream_);
}

// Graph-capture-friendly variant of transformer_layer. Same kernel
// sequence, same scratch-allocation order (so iter-1+ replay sees the
// same buffer addresses iter-0 used), but with two substitutions:
//   • rope_inplace_store_kv_fp16  → rope_inplace_store_kv_fp16_dyn
//   • flash_attention_decode      → flash_attention_decode_dyn
// Those kernels read the current decode position from `*d_pos_` at
// execution time, so a single captured graph stays valid for every
// decode step. INT8 KV and overflow positions are NOT supported in
// this path — the caller is expected to fall back to decode_step in
// those cases (gen_params_.kv_int8 || pos >= max_context).
void Engine::transformer_layer_graph(int layer, half* x) {
    const auto& lw = model_weights_.layers[layer];
    const int H = config_.hidden_dim;
    const int Q_DIM = config_.n_heads * config_.head_dim;
    const int KV_DIM = config_.n_kv_heads * config_.head_dim;
    const int I = config_.intermediate_dim;
    const bool norm_fp32 = (lw.rms_type == 0);

    half* normed   = (half*)scratch_.get(H * sizeof(half));
    half* q_buf    = (half*)scratch_.get(Q_DIM * sizeof(half));
    half* k_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* v_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));

    fused_rmsnorm_residual(normed, x, nullptr, lw.rms_attn, 1, H,
                           config_.rms_eps, norm_fp32, stream_);
    gemv_quant_triple(q_buf, lw.wq, lw.type_wq, Q_DIM,
                      k_buf, lw.wk, lw.type_wk, KV_DIM,
                      v_buf, lw.wv, lw.type_wv, KV_DIM,
                      normed, H, stream_);

    half* x2        = (half*)scratch_.get(H * sizeof(half));
    half* attn_out  = (half*)scratch_.get(Q_DIM * sizeof(half));
    half* attn_proj = (half*)scratch_.get(H * sizeof(half));

    if (lw.q_norm && lw.k_norm) {
        fused_rmsnorm_residual(q_buf, q_buf, nullptr, lw.q_norm,
                               config_.n_heads, config_.head_dim,
                               config_.rms_eps, lw.qk_norm_type == 0, stream_);
        fused_rmsnorm_residual(k_buf, k_buf, nullptr, lw.k_norm,
                               config_.n_kv_heads, config_.head_dim,
                               config_.rms_eps, lw.qk_norm_type == 0, stream_);
    }

    rope_inplace_store_kv_fp16_dyn(q_buf, k_buf, v_buf,
                                   (half*)kv_cache_.key_ptr(layer, 0),
                                   (half*)kv_cache_.val_ptr(layer, 0),
                                   config_.n_heads, config_.n_kv_heads,
                                   config_.head_dim, config_.head_dim,
                                   d_pos_, config_.rope_theta,
                                   config_.rope_neox, stream_);

    const float scale = 1.0f / sqrtf((float)config_.head_dim);
    flash_attention_decode_dyn(attn_out, q_buf,
                               kv_cache_.key_ptr(layer, 0),
                               kv_cache_.val_ptr(layer, 0),
                               config_.n_heads, config_.n_kv_heads,
                               config_.head_dim,
                               d_pos_, scale,
                               /*kv_int8=*/false, /*kv_scales=*/nullptr,
                               /*window=*/0, stream_);

    gemv_quant(attn_proj, lw.wo, lw.type_wo, attn_out, H, Q_DIM, stream_);
    vec_add(x2, x, attn_proj, H, stream_);

    half* normed2    = (half*)scratch_.get(H * sizeof(half));
    half* gate_buf   = (half*)scratch_.get(I * sizeof(half));
    half* up_buf     = (half*)scratch_.get(I * sizeof(half));
    half* swiglu_out = (half*)scratch_.get(I * sizeof(half));

    fused_rmsnorm_residual(normed2, x2, nullptr, lw.rms_ffn, 1, H,
                           config_.rms_eps, norm_fp32, stream_);
    gemv_quant_pair(gate_buf, lw.w_gate, lw.type_w_gate, I,
                    up_buf,   lw.w_up,   lw.type_w_up,   I,
                    normed2, H, stream_);
    fused_swiglu(swiglu_out, gate_buf, up_buf, 1, I, stream_);

    half* ffn_out = (half*)scratch_.get(H * sizeof(half));
    gemv_quant(ffn_out, lw.w_down, lw.type_w_down, swiglu_out, H, I, stream_);
    vec_add(x, x2, ffn_out, H, stream_);
}

// Graph-capture-friendly Gemma4 transformer layer. Mirrors
// transformer_layer_gemma4 in scratch-allocation order so replay sees the same
// buffer addresses. Uses rope_inplace_store_kv_fp16_dyn and
// flash_attention_decode_dyn so a single captured graph is valid for every
// decode step (pos is read from *d_pos_ inside those kernels).
//
// Requirements (enforced by build_cuda_graph before capture):
//   - n_kv_shared_layers == 0: all layers own K/V; no Q-only RoPE path needed.
//   - !gen_params_.kv_int8
void Engine::transformer_layer_graph_gemma4(int layer, half* x) {
    const auto& lw = model_weights_.layers[layer];
    const int H  = config_.hidden_dim;
    const int hd = lw.head_dim_l;                    // 256 sliding / 512 global
    const int Q_DIM  = config_.n_heads * hd;
    const int KV_DIM = config_.n_kv_heads * hd;
    const int I  = lw.intermediate_l;
    const bool norm_fp32 = (lw.rms_type == 0);

    half* normed   = (half*)scratch_.get(H * sizeof(half));
    half* q_buf    = (half*)scratch_.get(Q_DIM * sizeof(half));
    half* k_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* v_buf    = (half*)scratch_.get(KV_DIM * sizeof(half));
    half* attn_out = (half*)scratch_.get(Q_DIM * sizeof(half));

    fused_rmsnorm_residual(normed, x, nullptr, lw.rms_attn, 1, H,
                           config_.rms_eps, norm_fp32, stream_);

    // All layers own their own K/V (n_kv_shared_layers==0 enforced in build_cuda_graph).
    gemv_quant_triple(q_buf, lw.wq, lw.type_wq, Q_DIM,
                      k_buf, lw.wk, lw.type_wk, KV_DIM,
                      v_buf, lw.wv, lw.type_wv, KV_DIM,
                      normed, H, stream_);

    if (lw.q_norm) {
        fused_rmsnorm_residual(q_buf, q_buf, nullptr, lw.q_norm,
                               config_.n_heads, hd, config_.rms_eps,
                               lw.qk_norm_type == 0, stream_);
    }
    if (lw.k_norm) {
        fused_rmsnorm_residual(k_buf, k_buf, nullptr, lw.k_norm,
                               config_.n_kv_heads, hd, config_.rms_eps,
                               lw.qk_norm_type == 0, stream_);
    }
    if (gemma_vnorm_ones_) {
        fused_rmsnorm_residual(v_buf, v_buf, nullptr, gemma_vnorm_ones_,
                               config_.n_kv_heads, hd, config_.rms_eps,
                               /*weight_fp32=*/true, stream_);
    }

    // RoPE + KV store: position is read from *d_pos_ at execution time.
    // cache_head_dim = config_.head_dim (512) so sliding layers (hd=256)
    // write into the first 256 elements of each 512-element cache slot.
    rope_inplace_store_kv_fp16_dyn(q_buf, k_buf, v_buf,
                                   (half*)kv_cache_.key_ptr(layer, 0),
                                   (half*)kv_cache_.val_ptr(layer, 0),
                                   config_.n_heads, config_.n_kv_heads,
                                   hd, config_.head_dim,
                                   d_pos_, lw.rope_theta_l,
                                   config_.rope_neox, stream_);

    const float scale      = attention_scale(config_.spec, hd);
    const int   attn_window = lw.is_sliding ? config_.sliding_window : 0;
    flash_attention_decode_dyn(attn_out, q_buf,
                               kv_cache_.key_ptr(layer, 0),
                               kv_cache_.val_ptr(layer, 0),
                               config_.n_heads, config_.n_kv_heads,
                               hd, d_pos_, scale,
                               /*kv_int8=*/false, /*kv_scales=*/nullptr,
                               attn_window, stream_);

    half* wo_out = (half*)scratch_.get(H * sizeof(half));
    half* x2     = (half*)scratch_.get(H * sizeof(half));
    gemv_quant(wo_out, lw.wo, lw.type_wo, attn_out, H, Q_DIM, stream_);
    fused_rmsnorm_residual(wo_out, wo_out, nullptr, lw.post_attn_norm, 1, H,
                           config_.rms_eps, /*weight_fp32=*/true, stream_);
    vec_add(x2, x, wo_out, H, stream_);

    half* normed2 = (half*)scratch_.get(H * sizeof(half));
    half* gate    = (half*)scratch_.get(I * sizeof(half));
    half* up      = (half*)scratch_.get(I * sizeof(half));
    half* act     = (half*)scratch_.get(I * sizeof(half));
    half* ffn_out = (half*)scratch_.get(H * sizeof(half));
    fused_rmsnorm_residual(normed2, x2, nullptr, lw.rms_ffn, 1, H,
                           config_.rms_eps, norm_fp32, stream_);
    gemv_quant_pair(gate, lw.w_gate, lw.type_w_gate, I,
                    up,   lw.w_up,   lw.type_w_up,   I, normed2, H, stream_);
    fused_geglu(act, gate, up, 1, I, stream_);
    gemv_quant(ffn_out, lw.w_down, lw.type_w_down, act, H, I, stream_);
    fused_rmsnorm_residual(ffn_out, ffn_out, nullptr, lw.post_ffn_norm, 1, H,
                           config_.rms_eps, /*weight_fp32=*/true, stream_);
    vec_add(x, x2, ffn_out, H, stream_);

    if (lw.ple_gate && gemma_ple_input_) {
        const int D = config_.ple_input_dim;  // 256
        half* g_in = (half*)scratch_.get(D * sizeof(half));
        half* g    = (half*)scratch_.get(D * sizeof(half));
        half* p    = (half*)scratch_.get(H * sizeof(half));
        gemv_dense(g_in, lw.ple_gate, lw.type_ple_gate, x, D, H, stream_);
        // gemma_ple_input_ is updated each token (outside the graph).
        // The pointer is fixed; the kernel reads updated values at replay time.
        fused_geglu(g, g_in, gemma_ple_input_ + (int64_t)layer * D, 1, D, stream_);
        gemv_dense(p, lw.ple_proj, lw.type_ple_proj, g, H, D, stream_);
        fused_rmsnorm_residual(p, p, nullptr, lw.ple_norm, 1, H,
                               config_.rms_eps, /*weight_fp32=*/true, stream_);
        vec_add(x, x, p, H, stream_);
    }

    if (lw.layer_scale_val != 1.0f) {
        vec_scale(x, H, lw.layer_scale_val, stream_);
    }
}

// ── Batched prefill scaffolding (Path B, issue #12) ──────────────────────
//
// The initial implementation delegates to the proven single-token
// `transformer_layer` N times. That produces byte-identical output to the
// pre-existing per-token prefill loop and no speedup. The scaffolding's
// value is in giving future PRs a single high-level entry point where each
// inner kernel (QKV gemv → batched GEMM, RoPE → batched RoPE, attention →
// chunked prefill attention, etc.) can be swapped one at a time without
// touching `generate()` again.
//
// Layout invariant: `x_batch[i*H .. (i+1)*H)` is token i's hidden state,
// read AND written in place per layer (residual additions).
//
// KV ordering invariant: at the time we call transformer_layer for token i
// at layer l, layer l's K/V for positions [start_pos, start_pos+i) are
// already written. This holds because we iterate tokens in ascending
// order within each layer, and transformer_layer writes K/V[start_pos+i]
// before reading the attention range [0, start_pos+i].
void Engine::transformer_prefill(int layer, int start_pos, int n_tokens, half* x_batch) {
    const auto& lw = model_weights_.layers[layer];
    const int H = config_.hidden_dim;
    const int Q_DIM = config_.n_heads * config_.head_dim;
    const int KV_DIM = config_.n_kv_heads * config_.head_dim;
    const int I = config_.intermediate_dim;
    const int N = n_tokens;
    const bool norm_fp32 = (lw.rms_type == 0);

    const int64_t snapshot = scratch_.mark();

    // Persistent batched buffers for this layer:
    //   normed_batch    [N×H]    — input to QKV
    //   q/k/v_batch     [N×Q/KV] — QKV output, then QK-normed, then RoPE'd in place
    //   attn_out_batch  [N×Q]    — output of attention (head mix, pre-Wo)
    //   attn_proj_batch [N×H]    — Wo(attn_out)
    //   x_attn_batch    [N×H]    — x_batch + attn_proj_batch (residual #1)
    //   normed2_batch   [N×H]    — input to gate/up
    //   gate/up_batch   [N×I]    — gate/up output
    //   swiglu_batch    [N×I]    — input to W_down
    //   ffn_out_batch   [N×H]    — W_down(swiglu_batch)
    half* normed_batch    = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    half* q_batch         = (half*)scratch_.get((int64_t)N * Q_DIM  * sizeof(half));
    half* k_batch         = (half*)scratch_.get((int64_t)N * KV_DIM * sizeof(half));
    half* v_batch         = (half*)scratch_.get((int64_t)N * KV_DIM * sizeof(half));
    half* attn_out_batch  = (half*)scratch_.get((int64_t)N * Q_DIM  * sizeof(half));
    half* attn_proj_batch = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    half* x_attn_batch    = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    half* normed2_batch   = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    half* gate_batch      = (half*)scratch_.get((int64_t)N * I      * sizeof(half));
    half* up_batch        = (half*)scratch_.get((int64_t)N * I      * sizeof(half));
    half* swiglu_batch    = (half*)scratch_.get((int64_t)N * I      * sizeof(half));
    half* ffn_out_batch   = (half*)scratch_.get((int64_t)N * H      * sizeof(half));

    // ── Pre-attention RMSNorm + QKV + QK-norm (all batched) ────────────
    fused_rmsnorm_residual(normed_batch, x_batch, nullptr, lw.rms_attn,
                           N, H, config_.rms_eps, norm_fp32, stream_);
    gemm_quant_batched(q_batch, lw.wq, lw.type_wq, normed_batch, Q_DIM,  N, H, stream_);
    gemm_quant_batched(k_batch, lw.wk, lw.type_wk, normed_batch, KV_DIM, N, H, stream_);
    gemm_quant_batched(v_batch, lw.wv, lw.type_wv, normed_batch, KV_DIM, N, H, stream_);
    if (lw.q_norm && lw.k_norm) {
        fused_rmsnorm_residual(q_batch, q_batch, nullptr, lw.q_norm,
                               N * config_.n_heads, config_.head_dim,
                               config_.rms_eps, lw.qk_norm_type == 0, stream_);
        fused_rmsnorm_residual(k_batch, k_batch, nullptr, lw.k_norm,
                               N * config_.n_kv_heads, config_.head_dim,
                               config_.rms_eps, lw.qk_norm_type == 0, stream_);
    }

    // ── Per-token RoPE + KV store ─────────────────────────────────────
    // RoPE + KV store still run per-token because token i's K/V at layer
    // l must be written before token i+1's attention reads it. The
    // attention itself is now batched (one launch covers all N queries
    // via flash_attention_prefill_batched).
    for (int i = 0; i < N; i++) {
        const int pos = start_pos + i;
        half* q_t = q_batch + (int64_t)i * Q_DIM;
        half* k_t = k_batch + (int64_t)i * KV_DIM;
        half* v_t = v_batch + (int64_t)i * KV_DIM;
        if (!gen_params_.kv_int8 && kv_cache_.is_fast_position(pos)) {
            rope_inplace_store_kv_fp16(q_t, k_t, v_t,
                                      (half*)kv_cache_.key_ptr(layer, pos),
                                      (half*)kv_cache_.val_ptr(layer, pos),
                                      config_.n_heads, config_.n_kv_heads,
                                      config_.head_dim, pos, config_.rope_theta,
                                      config_.rope_neox, stream_);
        } else {
            rope_inplace(q_t, k_t, config_.n_heads, config_.n_kv_heads,
                         config_.head_dim, pos, config_.rope_theta,
                         config_.rope_neox, stream_);
            if (gen_params_.kv_int8) {
                // Path I phase I2: same wiring as the decode-step path.
                float* k_scales = kv_cache_.kv_scale_ptr(layer, pos, /*is_value=*/false);
                float* v_scales = kv_cache_.kv_scale_ptr(layer, pos, /*is_value=*/true);
                fp16_to_int8((int8_t*)kv_cache_.key_ptr(layer, pos), k_scales,
                             k_t, config_.n_kv_heads, config_.head_dim, stream_);
                fp16_to_int8((int8_t*)kv_cache_.val_ptr(layer, pos), v_scales,
                             v_t, config_.n_kv_heads, config_.head_dim, stream_);
            } else {
                cudaMemcpyAsync(kv_cache_.key_ptr(layer, pos), k_t,
                                KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
                cudaMemcpyAsync(kv_cache_.val_ptr(layer, pos), v_t,
                                KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
            }
        }
    }

    // ── Batched attention (NEW in PR #17): all N queries in one launch ──
    {
        const float scale = attention_scale(config_.spec, config_.head_dim);
        // Path I3 (#62): per-(pos, kv_head) scale regions for the layer.
        float* k_scales = gen_params_.kv_int8 ? kv_cache_.kv_scale_ptr(layer, 0, /*is_value=*/false) : nullptr;
        float* v_scales = gen_params_.kv_int8 ? kv_cache_.kv_scale_ptr(layer, 0, /*is_value=*/true)  : nullptr;
        flash_attention_prefill_batched(
            attn_out_batch, q_batch,
            kv_cache_.key_ptr(layer, 0), kv_cache_.val_ptr(layer, 0),
            config_.n_heads, config_.n_kv_heads, config_.head_dim,
            config_.head_dim, N, start_pos, scale,
            gen_params_.kv_int8, k_scales, v_scales, /*window=*/0, stream_);
    }

    // ── Batched Wo + batched residual #1 (NEW in PR #16) ──────────────
    // Wo is 5.9 MB / layer on Qwen3-4B (~8% of per-layer weight bandwidth).
    gemm_quant_batched(attn_proj_batch, lw.wo, lw.type_wo, attn_out_batch,
                       H, N, Q_DIM, stream_);
    vec_add(x_attn_batch, x_batch, attn_proj_batch, N * H, stream_);

    // ── FFN front: norm + gate/up + SwiGLU (all batched) ──────────────
    fused_rmsnorm_residual(normed2_batch, x_attn_batch, nullptr, lw.rms_ffn,
                           N, H, config_.rms_eps, norm_fp32, stream_);
    gemm_quant_batched(gate_batch, lw.w_gate, lw.type_w_gate, normed2_batch,
                       I, N, H, stream_);
    gemm_quant_batched(up_batch,   lw.w_up,   lw.type_w_up,   normed2_batch,
                       I, N, H, stream_);
    ffn_activation(config_.spec, swiglu_batch, gate_batch, up_batch, N, I, stream_);

    // ── Batched W_down + batched residual #2 (NEW in PR #16) ──────────
    // W_down is 25.8 MB / layer on Qwen3-4B (~35% of per-layer weight
    // bandwidth, second-biggest after gate/up).
    gemm_quant_batched(ffn_out_batch, lw.w_down, lw.type_w_down, swiglu_batch,
                       H, N, I, stream_);
    vec_add(x_batch, x_attn_batch, ffn_out_batch, N * H, stream_);

    scratch_.rewind_to(snapshot);
}

// Gemma 4 batched-prefill decoder layer. Mirrors transformer_prefill's
// layer-major batched structure, but with the Gemma decoder math validated in
// transformer_layer_gemma4: per-layer head dim, KV sharing (shared layers
// compute Q only and read the source layer's cache), weightless V-RMSNorm,
// GeGLU, the 4-norm sandwich, per-layer RoPE base + sliding window, PLE, and
// the learned layer-output scale. The big K-quant projections (QKV, Wo,
// gate/up, down) are batched across the N prompt tokens; only RoPE/KV-store
// and the small F32/BF16 PLE projections remain per-token.
void Engine::transformer_prefill_gemma4(int layer, int start_pos, int n_tokens,
                                        half* x_batch,
                                        const half* ple_input_batch) {
    const auto& lw = model_weights_.layers[layer];
    const int H  = config_.hidden_dim;
    const int hd = lw.head_dim_l;                  // 256 sliding / 512 global
    const int Q_DIM  = config_.n_heads * hd;
    const int KV_DIM = config_.n_kv_heads * hd;
    const int I  = lw.intermediate_l;              // 6144 / 12288
    const int N  = n_tokens;
    const int L  = config_.n_layers;
    const int D  = config_.ple_input_dim;          // 256
    const bool norm_fp32 = (lw.rms_type == 0);

    // KV sharing: layers [kv_from_start, n_layers) reuse the cache of the last
    // KV-producing layer of their attention type (see transformer_layer_gemma4).
    const int kv_from_start = config_.n_layers - config_.n_kv_shared_layers;
    const bool has_own_kv = (layer < kv_from_start);
    const int kv_layer = has_own_kv
        ? layer
        : (lw.is_sliding ? kv_from_start - 2 : kv_from_start - 1);

    const int64_t snapshot = scratch_.mark();

    half* normed_batch    = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    half* q_batch         = (half*)scratch_.get((int64_t)N * Q_DIM  * sizeof(half));
    half* k_batch         = (half*)scratch_.get((int64_t)N * KV_DIM * sizeof(half));
    half* v_batch         = (half*)scratch_.get((int64_t)N * KV_DIM * sizeof(half));
    half* attn_out_batch  = (half*)scratch_.get((int64_t)N * Q_DIM  * sizeof(half));
    half* attn_proj_batch = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    half* x_attn_batch    = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    half* normed2_batch   = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    half* gate_batch      = (half*)scratch_.get((int64_t)N * I      * sizeof(half));
    half* up_batch        = (half*)scratch_.get((int64_t)N * I      * sizeof(half));
    half* act_batch       = (half*)scratch_.get((int64_t)N * I      * sizeof(half));
    half* ffn_out_batch   = (half*)scratch_.get((int64_t)N * H      * sizeof(half));
    // Batched PLE buffers: this layer's per-token ple-input gathered into a
    // contiguous [N×D], then inp_gate output, gated value, and projection.
    half* ple_in_layer = (half*)scratch_.get((int64_t)N * D * sizeof(half));
    half* ple_gin_b    = (half*)scratch_.get((int64_t)N * D * sizeof(half));
    half* ple_g_b      = (half*)scratch_.get((int64_t)N * D * sizeof(half));
    half* ple_p_b      = (half*)scratch_.get((int64_t)N * H * sizeof(half));

    // 1. Input RMSNorm (batched).
    fused_rmsnorm_residual(normed_batch, x_batch, nullptr, lw.rms_attn,
                           N, H, config_.rms_eps, norm_fp32, stream_);

    // 2. QKV projections (batched). Shared-KV layers compute Q only.
    gemm_quant_batched(q_batch, lw.wq, lw.type_wq, normed_batch, Q_DIM, N, H, stream_);
    if (has_own_kv) {
        gemm_quant_batched(k_batch, lw.wk, lw.type_wk, normed_batch, KV_DIM, N, H, stream_);
        gemm_quant_batched(v_batch, lw.wv, lw.type_wv, normed_batch, KV_DIM, N, H, stream_);
    }

    // 3a. Q-norm always; K-norm + weightless V-norm only when owning K/V.
    if (lw.q_norm) {
        fused_rmsnorm_residual(q_batch, q_batch, nullptr, lw.q_norm,
                               N * config_.n_heads, hd, config_.rms_eps,
                               lw.qk_norm_type == 0, stream_);
    }
    if (has_own_kv) {
        if (lw.k_norm) {
            fused_rmsnorm_residual(k_batch, k_batch, nullptr, lw.k_norm,
                                   N * config_.n_kv_heads, hd, config_.rms_eps,
                                   lw.qk_norm_type == 0, stream_);
        }
        if (gemma_vnorm_ones_) {
            fused_rmsnorm_residual(v_batch, v_batch, nullptr, gemma_vnorm_ones_,
                                   N * config_.n_kv_heads, hd, config_.rms_eps,
                                   /*weight_fp32=*/true, stream_);
        }
    }

    // 3b. RoPE (per-layer base) + KV store. Batched over all N tokens in one
    //     launch when possible (FP16 KV + all-fast cache positions); per-token
    //     fallback otherwise (int8 KV / overflow / N<2). Shared-KV layers rotate
    //     Q only — their K/V already live in kv_layer's cache.
    bool roped_batched = false;
    if (!has_own_kv) {
        rope_inplace_batched(q_batch, nullptr, config_.n_heads, 0, hd,
                             start_pos, N, Q_DIM, 0, lw.rope_theta_l,
                             config_.rope_neox, stream_);
        roped_batched = true;
    } else if (!gen_params_.kv_int8 && N >= 2 &&
               kv_cache_.is_fast_position(start_pos + N - 1)) {
        half* kcb = (half*)kv_cache_.key_ptr(layer, start_pos);
        half* vcb = (half*)kv_cache_.val_ptr(layer, start_pos);
        const int kv_slot = (int)(((char*)kv_cache_.key_ptr(layer, start_pos + 1)
                                   - (char*)kcb) / sizeof(half));
        rope_inplace_store_kv_fp16_batched(q_batch, k_batch, v_batch, kcb, vcb,
            kv_slot, config_.n_heads, config_.n_kv_heads, hd, start_pos, N,
            Q_DIM, KV_DIM, lw.rope_theta_l, config_.rope_neox, stream_);
        roped_batched = true;
    }
    if (!roped_batched)
    for (int i = 0; i < N; i++) {
        const int pos = start_pos + i;
        half* q_t = q_batch + (int64_t)i * Q_DIM;
        if (!has_own_kv) {
            rope_inplace(q_t, nullptr, config_.n_heads, 0, hd, pos,
                         lw.rope_theta_l, config_.rope_neox, stream_);
            continue;
        }
        half* k_t = k_batch + (int64_t)i * KV_DIM;
        half* v_t = v_batch + (int64_t)i * KV_DIM;
        if (!gen_params_.kv_int8 && kv_cache_.is_fast_position(pos)) {
            rope_inplace_store_kv_fp16(q_t, k_t, v_t,
                                       (half*)kv_cache_.key_ptr(layer, pos),
                                       (half*)kv_cache_.val_ptr(layer, pos),
                                       config_.n_heads, config_.n_kv_heads, hd, pos,
                                       lw.rope_theta_l, config_.rope_neox, stream_);
        } else {
            rope_inplace(q_t, k_t, config_.n_heads, config_.n_kv_heads, hd, pos,
                         lw.rope_theta_l, config_.rope_neox, stream_);
            if (gen_params_.kv_int8) {
                float* ks = kv_cache_.kv_scale_ptr(layer, pos, false);
                float* vs = kv_cache_.kv_scale_ptr(layer, pos, true);
                fp16_to_int8((int8_t*)kv_cache_.key_ptr(layer, pos), ks, k_t,
                             config_.n_kv_heads, hd, stream_);
                fp16_to_int8((int8_t*)kv_cache_.val_ptr(layer, pos), vs, v_t,
                             config_.n_kv_heads, hd, stream_);
            } else {
                cudaMemcpyAsync(kv_cache_.key_ptr(layer, pos), k_t,
                                KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
                cudaMemcpyAsync(kv_cache_.val_ptr(layer, pos), v_t,
                                KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
            }
        }
    }

    // 3c. Batched attention (reads kv_layer; sliding layers mask the window).
    {
        const float scale = attention_scale(config_.spec, hd);
        const int window = lw.is_sliding ? config_.sliding_window : 0;
        float* k_scales = gen_params_.kv_int8 ? kv_cache_.kv_scale_ptr(kv_layer, 0, false) : nullptr;
        float* v_scales = gen_params_.kv_int8 ? kv_cache_.kv_scale_ptr(kv_layer, 0, true)  : nullptr;
        flash_attention_prefill_batched(
            attn_out_batch, q_batch,
            kv_cache_.key_ptr(kv_layer, 0), kv_cache_.val_ptr(kv_layer, 0),
            config_.n_heads, config_.n_kv_heads, hd, config_.head_dim,
            N, start_pos, scale,
            gen_params_.kv_int8, k_scales, v_scales, window, stream_);
    }

    // 3d. Wo -> post-attention (sandwich) norm -> residual (all batched).
    gemm_quant_batched(attn_proj_batch, lw.wo, lw.type_wo, attn_out_batch,
                       H, N, Q_DIM, stream_);
    fused_rmsnorm_residual(attn_proj_batch, attn_proj_batch, nullptr,
                           lw.post_attn_norm, N, H, config_.rms_eps,
                           /*weight_fp32=*/true, stream_);
    vec_add(x_attn_batch, x_batch, attn_proj_batch, N * H, stream_);

    // 4. FFN: norm -> GeGLU (double-wide) -> down -> post-ffn norm -> residual.
    fused_rmsnorm_residual(normed2_batch, x_attn_batch, nullptr, lw.rms_ffn,
                           N, H, config_.rms_eps, norm_fp32, stream_);
    gemm_quant_batched(gate_batch, lw.w_gate, lw.type_w_gate, normed2_batch, I, N, H, stream_);
    gemm_quant_batched(up_batch,   lw.w_up,   lw.type_w_up,   normed2_batch, I, N, H, stream_);
    fused_geglu(act_batch, gate_batch, up_batch, N, I, stream_);
    gemm_quant_batched(ffn_out_batch, lw.w_down, lw.type_w_down, act_batch, H, N, I, stream_);
    fused_rmsnorm_residual(ffn_out_batch, ffn_out_batch, nullptr, lw.post_ffn_norm,
                           N, H, config_.rms_eps, /*weight_fp32=*/true, stream_);
    // x_batch becomes pe_in = x_attn + post_ffn_norm(ffn); PLE then adds to it.
    vec_add(x_batch, x_attn_batch, ffn_out_batch, N * H, stream_);

    // 5. Per-Layer Embeddings (PLE), batched: inp_gate -> gelu*ple_input ->
    //    proj -> post-norm -> residual, over all N tokens at once.
    if (lw.ple_gate && ple_input_batch) {
        // Gather this layer's per-token ple-input into a contiguous [N×D].
        // ple_input_batch is [N × n_layers × D]; token t's layer slice is at
        // (t*L + layer)*D, strided by L*D between tokens.
        cudaMemcpy2DAsync(ple_in_layer, (size_t)D * sizeof(half),
                          ple_input_batch + (int64_t)layer * D,
                          (size_t)L * D * sizeof(half),
                          (size_t)D * sizeof(half), N,
                          cudaMemcpyDeviceToDevice, stream_);
        gemm_dense_batched(ple_gin_b, lw.ple_gate, lw.type_ple_gate, x_batch,
                           D, N, H, stream_);
        fused_geglu(ple_g_b, ple_gin_b, ple_in_layer, N, D, stream_);
        gemm_dense_batched(ple_p_b, lw.ple_proj, lw.type_ple_proj, ple_g_b,
                           H, N, D, stream_);
        fused_rmsnorm_residual(ple_p_b, ple_p_b, nullptr, lw.ple_norm, N, H,
                               config_.rms_eps, /*weight_fp32=*/true, stream_);
        vec_add(x_batch, x_batch, ple_p_b, N * H, stream_);
    }

    // 6. Per-layer learned output scale (batched).
    if (lw.layer_scale_val != 1.0f) {
        vec_scale(x_batch, N * H, lw.layer_scale_val, stream_);
    }

    scratch_.rewind_to(snapshot);
}

bool Engine::batched_prefill_enabled() const {
    // Default-on after the Path B series (#13-#17) landed. Output is
    // bit-identical to the per-token path on every tested model, and the
    // measured win on Qwen3-4B is 1.88× prefill / 47% TTFT reduction.
    // Set JLLM_BATCHED_PREFILL=0 to disable.
    static const bool enabled = [] {
        const char* v = getenv("JLLM_BATCHED_PREFILL");
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

// ── Full decode step ─────────────────────────────────────────────────────

int Engine::decode_step(int pos) {
    int H = config_.hidden_dim;
    const bool profile = profile_enabled();
    auto prof_t = Clock::now();
    float prof_emb_ms = 0.0f;
    float prof_layers_ms = 0.0f;
    float prof_norm_ms = 0.0f;
    float prof_logits_ms = 0.0f;
    float prof_sample_ms = 0.0f;

    half* x = (half*)scratch_.get(H * sizeof(half));

    // Embedding lookup — dequantize one row from the embedding table
    // token_embd is typically Q4_K (type 12) or Q6_K (type 14) in GGUF
    dequant_embedding(x, model_weights_.tok_embd, last_token_, H,
                      model_weights_.embd_type, stream_);
    scale_embedding(config_.spec, x, H, config_.hidden_dim, stream_);
    if (profile) {
        cudaStreamSynchronize(stream_);
        auto now = Clock::now();
        prof_emb_ms = Ms(now - prof_t).count();
        prof_t = now;
    }

    // Gemma 4: compute the Per-Layer-Embedding input once from the scaled
    // embedding (it depends on the token, so it can't be CUDA-graph captured).
    gemma_ple_input_ = nullptr;
    if (config_.arch == Arch::Gemma4 && config_.spec.per_layer_embeddings) {
        gemma_ple_input_ = (half*)scratch_.get(
            (int64_t)config_.n_layers * config_.ple_input_dim * sizeof(half));
        compute_gemma_ple_input(x, last_token_, gemma_ple_input_);
    }

    // All transformer layers
    for (int l = 0; l < config_.n_layers; l++) {
        if (config_.arch == Arch::Gemma4) transformer_layer_gemma4(l, pos, x);
        else transformer_layer(l, pos, x);
    }
    if (profile) {
        cudaStreamSynchronize(stream_);
        auto now = Clock::now();
        prof_layers_ms = Ms(now - prof_t).count();
        prof_t = now;
    }

    // Final RMSNorm
    half* normed = (half*)scratch_.get(H * sizeof(half));
    fused_rmsnorm_residual(normed, x, nullptr, model_weights_.output_norm, 1, H, config_.rms_eps, true, stream_);
    if (profile) {
        cudaStreamSynchronize(stream_);
        auto now = Clock::now();
        prof_norm_ms = Ms(now - prof_t).count();
        prof_t = now;
    }

    // Logits: write FP32 directly from the output projection. This avoids an
    // intermediate FP16 logits buffer plus a separate conversion kernel.
    float* logits_fp32 = (float*)scratch_.get(config_.vocab_size * sizeof(float));

    if (!logits_fp32 || !host_logits_ || host_logits_capacity_ < config_.vocab_size) {
        fprintf(stderr, "[decode] FATAL: failed to allocate logits buffer\n");
        return tokenizer_.eos_id;
    }

    gemv_quant_f32(logits_fp32, model_weights_.output, model_weights_.output_type,
                   normed, config_.vocab_size, H, stream_);
    apply_logit_softcap(config_.spec, logits_fp32, config_.vocab_size, stream_);

    // Copy FP32 logits to a pinned host buffer for CPU sampling.
    cudaError_t copy_err = cudaMemcpyAsync(host_logits_, logits_fp32,
                                           config_.vocab_size * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream_);
    if (copy_err == cudaSuccess) {
        copy_err = cudaStreamSynchronize(stream_);
    }
    if (copy_err != cudaSuccess) {
        fprintf(stderr, "[decode] FATAL: logits copy failed: %s\n",
                cudaGetErrorString(copy_err));
        return tokenizer_.eos_id;
    }
    if (profile) {
        auto now = Clock::now();
        prof_logits_ms = Ms(now - prof_t).count();
        prof_t = now;
    }

    // #86: mask logits to the grammar before sampling (no-op when inactive).
    if (grammar_active_) {
        grammar_apply(grammar_state_, grammar_token_bytes_, tokenizer_.eos_id,
                      host_logits_, config_.vocab_size);
    }

    // Sample
    int token = sample_token(host_logits_, config_.vocab_size, gen_params_,
                             recent_tokens_.data(), recent_tokens_.size());

    // #86: advance the grammar by the token we just committed to.
    if (grammar_active_) {
        grammar_accept_token(grammar_state_, grammar_token_bytes_,
                             tokenizer_.eos_id, token);
    }
    if (profile) {
        auto now = Clock::now();
        prof_sample_ms = Ms(now - prof_t).count();

        static int prof_count = 0;
        static double acc_emb = 0.0;
        static double acc_layers = 0.0;
        static double acc_norm = 0.0;
        static double acc_logits = 0.0;
        static double acc_sample = 0.0;

        prof_count++;
        acc_emb += prof_emb_ms;
        acc_layers += prof_layers_ms;
        acc_norm += prof_norm_ms;
        acc_logits += prof_logits_ms;
        acc_sample += prof_sample_ms;
        const double total = prof_emb_ms + prof_layers_ms + prof_norm_ms +
                             prof_logits_ms + prof_sample_ms;
        const double avg_total = (acc_emb + acc_layers + acc_norm +
                                  acc_logits + acc_sample) / prof_count;
        fprintf(stderr,
                "[profile] token=%d total=%.2f ms emb=%.2f layers=%.2f final_norm=%.2f logits=%.2f sample=%.2f "
                "avg_total=%.2f avg_layers=%.2f avg_logits=%.2f\n",
                prof_count, total, prof_emb_ms, prof_layers_ms, prof_norm_ms,
                prof_logits_ms, prof_sample_ms,
                avg_total, acc_layers / prof_count, acc_logits / prof_count);
    }

    recent_tokens_.push_back(token);
    if ((int)recent_tokens_.size() > 64) recent_tokens_.erase(recent_tokens_.begin());

    last_token_ = token;
    return token;
}

bool Engine::decode_graph_enabled() const {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_DECODE_GRAPH");
        // Default-on for non-Gemma4 models (Gemma4 is excluded in
        // build_cuda_graph). Opt out with JLLM_DECODE_GRAPH=0.
        return !v || strcmp(v, "0") != 0;
    }();
    return enabled;
}

// Capture the full decode-step GPU pipeline as a CUDA graph:
//   transformer_layer_graph × n_layers
//   final RMSNorm
//   output projection (gemv_quant_f32 → FP32 logits)
//   D2H cudaMemcpyAsync to the pinned host_logits_ buffer
//
// The captured kernels read the current decode position from `*d_pos_`
// (which the host updates before each replay) so a single graph stays
// valid for every step. Embedding lookup and sampling stay on the host
// side around each replay.
//
// Requirements (caller-checked before invoking):
//   - !gen_params_.kv_int8  (the dyn rope/KV-store kernel only emits FP16)
//   - pos < config_.max_seq_len AND pos within the KV fast pool, so all
//     captured KV-cache writes go to addresses derived from gpu_pool_
//     (no CPU overflow path).
void Engine::build_cuda_graph(int pos) {
    if (graph_captured_) return;
    if (!gen_params_.use_cuda_graph) return;
    if (!decode_graph_enabled()) return;
    // Gemma 4 with shared KV layers is not yet supported in the graph path
    // (Q-only RoPE for shared-KV layers needs a dyn variant we don't have).
    // Gemma4 E2B (n_kv_shared_layers==0) is fully supported.
    if (config_.arch == Arch::Gemma4 && config_.n_kv_shared_layers > 0) {
        fprintf(stderr, "[engine] decode-graph: Gemma4 shared-KV not supported in graph; staying on per-step path\n");
        return;
    }
    if (gen_params_.kv_int8) {
        fprintf(stderr, "[engine] decode-graph: INT8 KV not supported; staying on per-step path\n");
        return;
    }
    if (!kv_cache_.is_fast_position(pos)) {
        fprintf(stderr, "[engine] decode-graph: pos %d outside KV fast pool; staying on per-step path\n", pos);
        return;
    }

    fprintf(stderr, "[engine] Capturing decode CUDA graph%s...\n",
            config_.arch == Arch::Gemma4 ? " (Gemma4 — PLE updated per-step outside graph)" : "");

    int H = config_.hidden_dim;

    // Reset scratch so the captured allocation pattern matches what
    // decode_step_graph will produce on replay (it starts with one
    // scratch_.get(x) after scratch_.reset()).
    scratch_.reset();
    half* graph_x = (half*)scratch_.get(H * sizeof(half));

    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    for (int l = 0; l < config_.n_layers; l++) {
        if (config_.arch == Arch::Gemma4)
            transformer_layer_graph_gemma4(l, graph_x);
        else
            transformer_layer_graph(l, graph_x);
    }

    half* g_normed = (half*)scratch_.get(H * sizeof(half));
    fused_rmsnorm_residual(g_normed, graph_x, nullptr,
                          model_weights_.output_norm, 1, H, config_.rms_eps, true, stream_);

    float* g_logits_fp32 = (float*)scratch_.get(config_.vocab_size * sizeof(float));
    gemv_quant_f32(g_logits_fp32, model_weights_.output, model_weights_.output_type,
                   g_normed, config_.vocab_size, H, stream_);
    apply_logit_softcap(config_.spec, g_logits_fp32, config_.vocab_size, stream_);

    // D2H copy is inside the captured region so the host only has to
    // sync once per token. host_logits_ is pinned, allocated once on
    // load, so its address is stable across replays.
    cudaMemcpyAsync(host_logits_, g_logits_fp32,
                    config_.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    cudaError_t err = cudaStreamEndCapture(stream_, &decode_graph_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[engine] Graph capture failed: %s\n", cudaGetErrorString(err));
        decode_graph_ = nullptr;
        return;
    }

    err = cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[engine] Graph instantiation failed: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(decode_graph_);
        decode_graph_ = nullptr;
        return;
    }

    graph_captured_ = true;
    fprintf(stderr, "[engine] decode CUDA graph captured (%d layers + final norm + logits + D2H)\n",
            config_.n_layers);
}

int Engine::decode_step_graph(int pos) {
    int H = config_.hidden_dim;

    // Match the capture-time scratch layout exactly: reset, then one
    // scratch_.get(x) at offset 0. Every other captured pointer lives
    // at deterministic offsets above that and survives reset (the bump
    // allocator only zeroes the cursor, not the memory).
    scratch_.reset();
    half* x = (half*)scratch_.get(H * sizeof(half));

    dequant_embedding(x, model_weights_.tok_embd, last_token_, H,
                      model_weights_.embd_type, stream_);

    // Push the new pos to the device-side int that the captured
    // rope/attention dyn kernels read.
    cudaError_t err = cudaMemcpyAsync(d_pos_, &pos, sizeof(int),
                                      cudaMemcpyHostToDevice, stream_);

    // Gemma4: PLE is computed outside the layer loop into a fixed device
    // buffer that the captured graph reads by pointer.  Values change each
    // step but the pointer captured in the graph is stable.
    if (err == cudaSuccess && config_.arch == Arch::Gemma4 && gemma_ple_input_) {
        compute_gemma_ple_input(x, last_token_, gemma_ple_input_);
    }

    if (err == cudaSuccess) {
        err = cudaGraphLaunch(decode_graph_exec_, stream_);
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream_);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[engine] decode-graph replay failed: %s; falling back\n",
                cudaGetErrorString(err));
        return decode_step(pos);
    }

    int token = sample_token(host_logits_, config_.vocab_size, gen_params_,
                             recent_tokens_.data(), recent_tokens_.size());

    recent_tokens_.push_back(token);
    if ((int)recent_tokens_.size() > 64) recent_tokens_.erase(recent_tokens_.begin());
    last_token_ = token;
    return token;
}

// ── Memory and thermal check ─────────────────────────────────────────────

bool Engine::check_memory_and_thermal(int pos) {
    OOMGuard guard(256);
    int kv_bytes = gen_params_.kv_int8 ? 1 : 2;
    if (!guard.can_extend(config_.kv_per_token_bytes(kv_bytes))) {
        fprintf(stderr, "\n[oom_guard] Stopping at token %d — %ld MB free\n",
                pos, guard.real_free_mb());
        return false;
    }

    if (pos % 10 == 0) {
        auto ts = read_thermal();
        int backoff = thermal_backoff_us(ts);
        if (backoff > 0) {
            fprintf(stderr, "\n[thermal] %.1f°C — backing off %d ms\n",
                    ts.gpu_temp_c, backoff / 1000);
            usleep(backoff);
        }
    }
    return true;
}

// ── Main generation loop ─────────────────────────────────────────────────

// F4b: hydrate KV from a saved cache file if compatible. Returns the
// number of tokens at the start of `prompt_tokens` that are already
// resident in kv_cache_ after this call (= longest common prefix
// between cached and new tokens). 0 means no hydration; the caller
// must run a cold prefill.
int Engine::try_hydrate_kv(const std::vector<int>& prompt_tokens,
                           const std::string& conv_id)
{
    if (conv_id.empty() || !validate_conversation_id(conv_id)) return 0;
    if (prompt_tokens.empty()) return 0;

    const std::string dir  = default_kv_cache_dir();
    const std::string path = path_for_conv_id(dir, conv_id);

    KVCacheFileHeader hdr{};
    if (!peek_kv_header(path, &hdr)) return 0;

    const auto& kv_cfg = kv_cache_.config();

    // Compatibility checks. Any mismatch → drop to cold prefill.
    if (std::memcmp(hdr.model_hash, model_fingerprint_, sizeof(hdr.model_hash)) != 0) {
        fprintf(stderr, "[kv_cache] hydrate: model fingerprint mismatch "
                        "for %s — cache built against a different model\n",
                path.c_str());
        return 0;
    }
    if ((int)hdr.n_layers      != kv_cfg.n_layers      ||
        (int)hdr.n_kv_heads    != kv_cfg.n_kv_heads    ||
        (int)hdr.head_dim      != kv_cfg.head_dim      ||
        (int)hdr.max_context   != kv_cfg.max_context   ||
        (int)hdr.kv_type_bytes != kv_cfg.kv_type_bytes)
    {
        fprintf(stderr, "[kv_cache] hydrate: shape mismatch on %s "
                        "(layers/heads/head_dim/ctx/kv_bytes) — using cold "
                        "prefill\n", path.c_str());
        return 0;
    }
    if (hdr.used_tokens == 0 || hdr.used_tokens > (uint32_t)kv_cfg.max_context) {
        return 0;
    }

    // Read tokens + body. We need to allocate a host buffer for the
    // body before calling load_kv_from_file.
    std::vector<uint32_t> cached_tokens;
    std::vector<uint8_t>  staging((size_t)hdr.body_bytes);

    auto t_load0 = Clock::now();
    KVCacheFileHeader hdr2{};
    bool ok = load_kv_from_file(path, hdr2, &cached_tokens,
                                staging.data(), staging.size());
    if (!ok) {
        fprintf(stderr, "[kv_cache] hydrate: load failed for %s\n",
                path.c_str());
        return 0;
    }
    if (cached_tokens.size() != hdr.used_tokens) {
        fprintf(stderr, "[kv_cache] hydrate: tokens vector size %zu != "
                        "header used_tokens %u\n",
                cached_tokens.size(), hdr.used_tokens);
        return 0;
    }

    // Longest common prefix between cached and new prompt tokens.
    const int max_match = std::min((int)cached_tokens.size(),
                                   (int)prompt_tokens.size());
    int matched = 0;
    while (matched < max_match &&
           cached_tokens[matched] == (uint32_t)prompt_tokens[matched]) {
        matched++;
    }
    if (matched == 0) {
        fprintf(stderr, "[kv_cache] hydrate: no common prefix vs cached "
                        "(%u tokens) — cold prefill\n", hdr.used_tokens);
        return 0;
    }

    // Compact the saved per-layer packed body (each layer is
    // hdr.used_tokens × eb) into the matched-only layout that
    // scatter_from_host expects (each layer is `matched` × eb).
    const int64_t eb       = 2LL * kv_cfg.n_kv_heads * kv_cfg.head_dim
                                * kv_cfg.kv_type_bytes;
    const int64_t half_eb  = eb / 2;
    const int64_t src_layer_stride = (int64_t)hdr.used_tokens * eb;
    const int64_t dst_layer_stride = (int64_t)matched * eb;
    const int64_t matched_keys_bytes = (int64_t)matched * half_eb;

    std::vector<uint8_t> compact((size_t)kv_cfg.n_layers * dst_layer_stride);
    for (int l = 0; l < kv_cfg.n_layers; l++) {
        const uint8_t* src_keys = staging.data() + l * src_layer_stride;
        const uint8_t* src_vals = src_keys + (int64_t)hdr.used_tokens * half_eb;
        uint8_t* dst_keys = compact.data() + l * dst_layer_stride;
        uint8_t* dst_vals = dst_keys + matched_keys_bytes;
        std::memcpy(dst_keys, src_keys, (size_t)matched_keys_bytes);
        std::memcpy(dst_vals, src_vals, (size_t)matched_keys_bytes);
    }

    if (!kv_cache_.scatter_from_host(compact.data(), matched,
                                     /*zero_remaining=*/true)) {
        fprintf(stderr, "[kv_cache] hydrate: scatter_from_host failed\n");
        return 0;
    }

    auto t_load1 = Clock::now();
    float load_ms = Ms(t_load1 - t_load0).count();
    fprintf(stderr,
            "[kv_cache] Hydrated %d / %u cached tokens from %s "
            "(prompt %zu, %.0f ms) — skipping prefill for matched prefix\n",
            matched, hdr.used_tokens, path.c_str(),
            prompt_tokens.size(), load_ms);
    return matched;
}

bool validate_conversation_id(const std::string& id) {
    if (id.empty()) return true;
    if (id.size() > 64) return false;
    for (char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                         c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

// #86: parse the request's GBNF grammar and arm constrained decoding. On any
// parse failure we log and leave grammar_active_ false so generation proceeds
// unconstrained rather than failing the request. The token-bytes cache (token
// id -> literal output bytes) is built once; the vocabulary is fixed after
// load(), so it is reused across requests.
void Engine::prepare_grammar(const GenParams& params) {
    grammar_active_ = false;
    if (params.grammar.empty()) return;

    grammar_ = grammar_parse(params.grammar, params.grammar_root);
    if (!grammar_.ok) {
        fprintf(stderr, "[engine] WARN: grammar parse failed — decoding unconstrained\n");
        return;
    }
    if ((int)grammar_token_bytes_.size() != config_.vocab_size) {
        grammar_token_bytes_.assign(config_.vocab_size, std::string());
        for (int i = 0; i < config_.vocab_size; i++)
            grammar_token_bytes_[i] = tokenizer_.decode(i);
    }
    grammar_state_init(grammar_state_, grammar_);
    grammar_active_ = true;
    fprintf(stderr, "[engine] grammar-constrained decoding active (%zu rules, root=%s)\n",
            grammar_.rules.size(), params.grammar_root.c_str());
}

GenStats Engine::generate(const std::string& prompt, const GenParams& params,
                          TokenCallback token_cb) {
    GenStats stats = {};
    stop_flag_ = false;
    gen_params_ = params;
    // Keep KV precision consistent with how the pool was sized in load():
    // Gemma forces FP16 KV (INT8 too lossy for its V distribution).
    if (config_.arch == Arch::Gemma4) gen_params_.kv_int8 = false;
    recent_tokens_.clear();

    // #86: set up grammar-constrained decoding for this request (no-op when
    // params.grammar is empty). Must run before prefill so the first-token
    // sample (Path A) is already masked.
    prepare_grammar(params);

    // Path F (#45): plumbing only in F2 — log conversation_id if set; the
    // F3 hooks (save on turn end, hydrate on turn start) consume it.
    // validate_conversation_id() ran at the CLI / HTTP boundary; we
    // re-check defensively here in case a future call site forgets.
    if (!params.conversation_id.empty()) {
        if (!validate_conversation_id(params.conversation_id)) {
            fprintf(stderr,
                    "[engine] WARNING: rejecting invalid conversation_id "
                    "(must match [A-Za-z0-9_-]{1,64}); single-shot mode\n");
        } else {
            fprintf(stderr,
                    "[engine] conversation_id=%s (Path F plumbing — "
                    "persistence not yet wired in F2)\n",
                    params.conversation_id.c_str());
        }
    }

    // t_request: wall-clock at the moment generate() was called. The
    // reference point for time-to-first-token, which captures the full
    // user-visible latency from prompt-submitted to first-token-delivered
    // (tokenization + prefill + first decode step).
    auto t_request = Clock::now();

    const int kv_token_limit = kv_cache_.max_tokens();
    const size_t max_prompt_bytes = std::max<size_t>(
        4096, (size_t)kv_token_limit * 8);
    if (prompt.size() > max_prompt_bytes) {
        throw std::length_error(
            "prompt size " + std::to_string(prompt.size()) +
            " bytes exceeds runtime pre-tokenization budget " +
            std::to_string(max_prompt_bytes) +
            " bytes for context capacity " +
            std::to_string(kv_token_limit) +
            " tokens; reduce the prompt/history or start the server with a larger -c value");
    }

    auto prompt_tokens = tokenizer_.encode(prompt);
    stats.prompt_tokens = prompt_tokens.size();
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(tokenizer_.bos_id);
    }

    if ((int)prompt_tokens.size() >= kv_token_limit) {
        throw std::length_error(
            "prompt length " + std::to_string(prompt_tokens.size()) +
            " tokens exceeds runtime context capacity " +
            std::to_string(kv_token_limit) +
            " tokens; reduce the prompt/history or start the server with a larger -c value");
    }

    // Path F4a (#45): track the tokens whose K/V state is committed to
    // the cache, in order, so we can persist them alongside the KV bytes
    // and F4b can find the longest common prefix between a cached
    // conversation and a new prompt. Starts with prompt_tokens (those
    // are written by prefill); grows by one in the decode loop right
    // before each decode_step writes the next position. The LAST
    // sampled token of a turn lives in last_token_ but never makes it
    // to KV (the loop exits before another decode_step commits it), so
    // we deliberately don't push it.
    std::vector<uint32_t> kv_tokens;
    kv_tokens.reserve(prompt_tokens.size() + params.max_tokens);
    for (int t : prompt_tokens) kv_tokens.push_back((uint32_t)t);

    // In-memory prefix cache: the KV buffers still hold the previous
    // request's tokens (requests are serialized by the server mutex and
    // nothing zeroes the KV between turns). Reuse the longest common prefix
    // with the resident tokens and skip prefill for it — this is what makes
    // the shared system prompt free across unrelated requests, with zero disk
    // I/O. RoPE/positions stay consistent because the shared prefix sits at
    // the same absolute positions. Capped at size()-1 so we always prefill
    // >=1 token (the M>0 path; M==0 has no Path A residual). Causal attention
    // only reads [0, pos], so the stale KV beyond the new sequence is never
    // read — exactly as in the cold path. Disable with JLLM_NO_PREFIX_CACHE=1.
    int prefill_start = 0;
    if (getenv("JLLM_NO_PREFIX_CACHE") == nullptr) {
        const int cap = std::min((int)resident_tokens_.size(),
                                 (int)prompt_tokens.size() - 1);
        int matched = 0;
        while (matched < cap &&
               resident_tokens_[matched] == (uint32_t)prompt_tokens[matched]) {
            matched++;
        }
        if (matched > 0) {
            prefill_start = matched;
            fprintf(stderr,
                    "[kv_cache] in-memory prefix reuse: %d / %zu prompt tokens "
                    "(resident %zu) — skipping prefill for matched prefix\n",
                    matched, prompt_tokens.size(), resident_tokens_.size());
        }
    }

    // Path F4b (#45): fall back to the persisted conversation_id cache only
    // when the in-memory prefix missed (cold start, or a divergent first
    // token). Returns the longest common prefix it could hydrate; 0 = cold.
    if (prefill_start == 0) {
        prefill_start = try_hydrate_kv(prompt_tokens, params.conversation_id);
    }
    // #83: reused-token count for the jetson.cache stats — covers both the
    // in-memory prefix reuse above and the conversation_id file hydrate.
    stats.kv_cache_reused_tokens = prefill_start;
    if (debug_kernels_enabled()) {
        fprintf(stderr, "[tokenizer] prompt tokens:");
        for (int i = 0; i < (int)prompt_tokens.size() && i < 16; i++) {
            std::string piece = tokenizer_.decode(prompt_tokens[i]);
            fprintf(stderr, " %d='%s'", prompt_tokens[i], piece.c_str());
        }
        if (prompt_tokens.size() > 16) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }

    // Prefill — process each prompt token through all 36 transformer layers
    // and store its K/V in the cache. The last iteration also retains the
    // final hidden state `last_prefill_x` so we can sample the first
    // generated token without a redundant forward pass (see Path A
    // optimization comment below).
    auto t0 = Clock::now();
    half* last_prefill_x = nullptr;
    int H = config_.hidden_dim;
    const int N = (int)prompt_tokens.size();

    // x_batch + transformer_prefill's persistent per-layer batched buffers
    // (normed, q/k/v, attn_out, attn_proj, x_attn, normed2, gate/up/swiglu,
    // ffn_out — see the comment there for the full list) all live in
    // scratch. Approximate total: N × (5H + 2Q + 2KV + 3I) × 2 bytes.
    const int Q_DIM_est  = config_.n_heads    * config_.head_dim;
    const int KV_DIM_est = config_.n_kv_heads * config_.head_dim;
    const int64_t batched_per_token_bytes =
        (int64_t)(5 * H + 2 * Q_DIM_est + 2 * KV_DIM_est + 3 * config_.intermediate_dim) * sizeof(half);
    constexpr int64_t kBatchedScratchMargin = 4 * 1024 * 1024;
    const int64_t x_batch_bytes = (int64_t)N * H * sizeof(half);
    const int64_t batched_total = x_batch_bytes + (int64_t)N * batched_per_token_bytes;
    const bool batched_fits =
        (scratch_.capacity() - batched_total) >= kBatchedScratchMargin;

    // Gemma 4 batched prefill needs the same per-token transient plus a
    // persistent per-token PLE-input table [n_layers × ple_input_dim] and the
    // per-layer batched PLE transient (3*ple_dim + hidden per token).
    const int64_t gemma_ple_batch_bytes =
        (int64_t)N * (config_.n_layers * config_.ple_input_dim +
                      3 * config_.ple_input_dim + config_.hidden_dim) *
        sizeof(half);
    const int64_t gemma_batched_total = batched_total + gemma_ple_batch_bytes;
    const bool gemma_batched_fits =
        (scratch_.capacity() - gemma_batched_total) >= kBatchedScratchMargin;

    // Path F4b: prefill works on tokens [prefill_start, N). Positions
    // [0, prefill_start) are already in kv_cache_ via try_hydrate_kv.
    // M is the number of tokens actually prefilled this turn; M=0 means
    // the cache already covered the full prompt — handled as a special
    // case below (no Path A residual, fall through to fallback decode).
    const int M = N - prefill_start;
    stats.prefill_tokens = M;

    if (M > 0 && config_.arch != Arch::Gemma4 && batched_prefill_enabled() && batched_fits) {
        // Path B (issue #12): layer-major prefill. Allocate one
        // [M × H] activation buffer (M = tokens to prefill this turn),
        // dequantize each new prompt embedding into it, then loop
        // layers-outer / tokens-inner. transformer_prefill writes
        // KV[prefill_start + i] for i in [0, M) and reads attention
        // history [0, prefill_start + i] — picks up the hydrated
        // prefix transparently.
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[engine] Path B: batched prefill enabled (JLLM_BATCHED_PREFILL=1)\n");
            logged = true;
        }
        scratch_.reset();
        const int64_t x_batch_bytes_m = (int64_t)M * H * sizeof(half);
        half* x_batch = (half*)scratch_.get(x_batch_bytes_m);
        for (int i = 0; i < M; i++) {
            dequant_embedding(x_batch + (int64_t)i * H,
                              model_weights_.tok_embd,
                              prompt_tokens[prefill_start + i], H,
                              model_weights_.embd_type, stream_);
        }
        scale_embedding(config_.spec, x_batch, M * H, config_.hidden_dim, stream_);
        for (int l = 0; l < config_.n_layers; l++) {
            transformer_prefill(l, prefill_start, M, x_batch);
        }
        last_token_ = prompt_tokens.back();
        last_prefill_x = x_batch + (int64_t)(M - 1) * H;
    } else if (M > 0 && config_.arch == Arch::Gemma4 &&
               config_.spec.per_layer_embeddings &&
               batched_prefill_enabled()) {
        // Path B for Gemma 4: layer-major batched prefill, CHUNKED. Prompts
        // longer than the batched scratch (BATCH_CAP) used to fall back to the
        // ~5x slower per-token prefill; instead process the prompt in chunks
        // that fit the scratch, each chunk batched layer-major. The KV cache
        // accumulates across chunks (chunk c attends to all prior KV), so the
        // result is identical — only faster for long prompts.
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[engine] Path B: Gemma 4 batched prefill enabled (chunked)\n");
            logged = true;
        }
        const int L = config_.n_layers;
        const int Dple = config_.ple_input_dim;
        const int64_t gemma_per_tok =
            (int64_t)H * sizeof(half) + batched_per_token_bytes +
            (int64_t)(L * Dple + 3 * Dple + config_.hidden_dim) * sizeof(half);
        int chunk_cap = (int)((scratch_.capacity() - kBatchedScratchMargin) / gemma_per_tok);
        if (chunk_cap < 1) chunk_cap = 1;
        for (int c0 = prefill_start; c0 < N; c0 += chunk_cap) {
            const int chunk = std::min(chunk_cap, N - c0);
            scratch_.reset();
            half* x_batch = (half*)scratch_.get((int64_t)chunk * H * sizeof(half));
            for (int i = 0; i < chunk; i++) {
                dequant_embedding(x_batch + (int64_t)i * H, model_weights_.tok_embd,
                                  prompt_tokens[c0 + i], H,
                                  model_weights_.embd_type, stream_);
            }
            scale_embedding(config_.spec, x_batch, chunk * H, config_.hidden_dim, stream_);
            half* ple_input_batch =
                (half*)scratch_.get((int64_t)chunk * L * Dple * sizeof(half));
            // Per-Layer-Embedding input, BATCHED over the chunk: the model_proj
            // is a [L*Dple x H] @ [H] gemv per token; doing it per-token (one
            // gemv_dense each) dominated prefill. Batch the GEMM + norm + scales;
            // keep the cheap token_identity dequant+add per-token (small reused
            // scratch, so the arena sizing is unchanged).
            {
                const auto& mw = model_weights_;
                if (mw.ple_embd && mw.ple_model_proj && mw.ple_proj_norm) {
                    const int total  = L * Dple;
                    const int ntotal = chunk * total;
                    // context = ple_proj_norm( (model_proj @ embed)/sqrt(H) ), batched.
                    gemm_dense_batched(ple_input_batch, mw.ple_model_proj,
                                       mw.ple_model_proj_type, x_batch,
                                       total, chunk, H, stream_);
                    vec_scale(ple_input_batch, ntotal, 1.0f / sqrtf((float)H), stream_);
                    fused_rmsnorm_residual(ple_input_batch, ple_input_batch, nullptr,
                                           mw.ple_proj_norm, chunk * L, Dple,
                                           config_.rms_eps, /*weight_fp32=*/true, stream_);
                    // + token_identity (per-token dequant) then *1/sqrt(2), per token.
                    for (int i = 0; i < chunk; i++) {
                        const int64_t snap = scratch_.mark();
                        half* o = ple_input_batch + (int64_t)i * total;
                        half* tid = (half*)scratch_.get(total * sizeof(half));
                        dequant_embedding(tid, mw.ple_embd, prompt_tokens[c0 + i],
                                          total, mw.ple_embd_type, stream_);
                        vec_scale(tid, total, sqrtf((float)Dple), stream_);
                        vec_add(o, o, tid, total, stream_);
                        scratch_.rewind_to(snap);
                    }
                    vec_scale(ple_input_batch, ntotal, 0.70710678f, stream_);
                }
            }
            for (int l = 0; l < config_.n_layers; l++) {
                transformer_prefill_gemma4(l, c0, chunk, x_batch, ple_input_batch);
            }
            if (c0 + chunk == N) last_prefill_x = x_batch + (int64_t)(chunk - 1) * H;
        }
        last_token_ = prompt_tokens.back();
        (void)gemma_batched_fits;
    } else if (M > 0) {
        if (batched_prefill_enabled() && !batched_fits) {
            static bool warned = false;
            if (!warned) {
                fprintf(stderr,
                        "[engine] Path B: prompt too long for batched prefill "
                        "(N=%d, would need %ld bytes of scratch); falling back "
                        "to per-token prefill\n", N, (long)batched_total);
                warned = true;
            }
        }
        for (int i = prefill_start; i < N; i++) {
            scratch_.reset();
            last_token_ = prompt_tokens[i];
            half* x = (half*)scratch_.get(H * sizeof(half));
            dequant_embedding(x, model_weights_.tok_embd, last_token_, H,
                              model_weights_.embd_type, stream_);
            scale_embedding(config_.spec, x, H, config_.hidden_dim, stream_);
            gemma_ple_input_ = nullptr;
            if (config_.arch == Arch::Gemma4 && config_.spec.per_layer_embeddings) {
                gemma_ple_input_ = (half*)scratch_.get(
                    (int64_t)config_.n_layers * config_.ple_input_dim * sizeof(half));
                compute_gemma_ple_input(x, last_token_, gemma_ple_input_);
            }
            for (int l = 0; l < config_.n_layers; l++) {
                if (config_.arch == Arch::Gemma4) transformer_layer_gemma4(l, i, x);
                else transformer_layer(l, i, x);
            }
            // Hang onto the last token's residual; the scratch pool has NOT
            // been reset between iterations within prefill, so this pointer
            // remains valid through the post-prefill sampling step below.
            if (i == N - 1) {
                last_prefill_x = x;
            }
        }
    } else {
        // M == 0: hydrate covered the entire prompt. Nothing to prefill.
        // Set last_token_ so the decode loop's fallback path picks up
        // properly — Path A's last_prefill_x is null in this case, which
        // forces the loop into the fallback that re-runs decode_step at
        // pos = N - 1 to recover logits for the final token.
        last_token_ = prompt_tokens.back();
        last_prefill_x = nullptr;
    }
    cudaStreamSynchronize(stream_);
    auto t1 = Clock::now();
    stats.prompt_ms = Ms(t1 - t0).count();
    if (stats.prefill_tokens > 0 && stats.prompt_ms > 0)
        stats.prompt_tok_per_sec = stats.prefill_tokens / (stats.prompt_ms / 1000.0f);
    fprintf(stderr,
            "[engine] Prefill: %d new / %d prompt tokens in %.0f ms "
            "(%.1f tok/s, kv_reused=%d)\n",
            stats.prefill_tokens, stats.prompt_tokens, stats.prompt_ms,
            stats.prompt_tok_per_sec, stats.kv_cache_reused_tokens);

    // Decode timer starts here so the Path A first-token sampling below
    // and the subsequent decode-loop iterations are both attributed to
    // decode_ms. Otherwise decode_tok_per_sec would over-report (token
    // count includes Path A's first token, but if t2 starts after Path A
    // the elapsed time excludes it).
    auto t2 = Clock::now();

    // ── Path A: sample the first generated token directly from prefill ──
    //
    // The previous flow re-ran the entire transformer stack for position
    // N-1 inside the first decode_step() iteration just to recover logits
    // from a hidden state we already had. That redundant forward pass
    // was ~120-150 ms on Qwen3-4B and dominated TTFT after the prefill
    // bandwidth cost itself.
    //
    // Now we run output_norm + logits gemv + sample on `last_prefill_x`
    // directly, then enter the decode loop already holding the first
    // sampled token. The decode loop starts at i=1 / pos=N — no double-
    // store in the KV cache, no redundant transformer-layer pass.
    int first_token = -1;
    bool first_is_eos = false;
    if (!prompt_tokens.empty() && last_prefill_x != nullptr) {
        half* normed = (half*)scratch_.get(H * sizeof(half));
        float* logits_fp32 = (float*)scratch_.get(config_.vocab_size * sizeof(float));
        if (normed && logits_fp32 && host_logits_ &&
            host_logits_capacity_ >= config_.vocab_size) {
            fused_rmsnorm_residual(normed, last_prefill_x, nullptr,
                                   model_weights_.output_norm,
                                   1, H, config_.rms_eps, true, stream_);
            gemv_quant_f32(logits_fp32, model_weights_.output,
                           model_weights_.output_type,
                           normed, config_.vocab_size, H, stream_);
            apply_logit_softcap(config_.spec, logits_fp32, config_.vocab_size, stream_);
            cudaError_t copy_err =
                cudaMemcpyAsync(host_logits_, logits_fp32,
                                config_.vocab_size * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_);
            if (copy_err == cudaSuccess) {
                copy_err = cudaStreamSynchronize(stream_);
            }
            if (copy_err == cudaSuccess) {
                // #86: grammar-mask the first token too (Path A).
                if (grammar_active_) {
                    grammar_apply(grammar_state_, grammar_token_bytes_,
                                  tokenizer_.eos_id, host_logits_,
                                  config_.vocab_size);
                }
                first_token = sample_token(host_logits_, config_.vocab_size,
                                           gen_params_,
                                           recent_tokens_.data(),
                                           recent_tokens_.size());
                if (grammar_active_) {
                    grammar_accept_token(grammar_state_, grammar_token_bytes_,
                                         tokenizer_.eos_id, first_token);
                }
            } else {
                fprintf(stderr,
                        "[engine] WARN: first-token logits copy failed (%s); "
                        "falling back to full decode_step for first token\n",
                        cudaGetErrorString(copy_err));
            }
        }
    }

    // Decode loop (t2 already taken above so it covers Path A too).
    int64_t peak_mem = 0;
    float peak_temp = 0;
    // Path A: the first generated token (if any) was already sampled
    // above. Position advances to N (one past the prompt) — we did NOT
    // double-store position N-1 in the KV cache, unlike the previous
    // recompute-the-last-prompt-position flow.
    int pos = (int)prompt_tokens.size();
    int loop_start = 0;
    if (first_token >= 0) {
        // Deliver the first token, stamp TTFT, advance loop state.
        stats.ttft_ms = Ms(Clock::now() - t_request).count();
        first_is_eos = (first_token == tokenizer_.eos_id);
        if (token_cb) {
            std::string text = tokenizer_.decode(first_token);
            token_cb(text.c_str(), first_is_eos);
        }
        last_token_ = first_token;
        recent_tokens_.push_back(first_token);
        if ((int)recent_tokens_.size() > 64) recent_tokens_.erase(recent_tokens_.begin());
        stats.completion_tokens++;
        // Skip the body of decode-loop iteration 0; it's already been done.
        loop_start = 1;
    } else if (!prompt_tokens.empty()) {
        // Fallback if Path A failed: start from pos=N-1 like the
        // original code so decode_step's first call recomputes the
        // last prompt position to get logits. Behavior is identical
        // to the pre-Path-A code path.
        pos = std::max(0, (int)prompt_tokens.size() - 1);
    }

    for (int i = loop_start; i < params.max_tokens && !stop_flag_ && !first_is_eos; i++) {
        if (pos >= kv_token_limit) {
            fprintf(stderr, "\n[engine] Stopping at context limit (%d tokens)\n",
                    kv_token_limit);
            break;
        }
        if (!check_memory_and_thermal(pos)) {
            stats.oom_stops++;
            break;
        }

        // Path C Phase 2 (#21): once we've successfully completed one
        // decode step on the per-token path, capture a CUDA graph for
        // every subsequent step. Replay covers ~448 launches per token
        // → one cudaGraphLaunch, saves ~10 ms of host-side overhead
        // per token. Falls back to per-step on any error and stays off
        // when JLLM_DECODE_GRAPH is unset.
        const bool use_graph = decode_graph_enabled()
                            && graph_captured_
                            && !gen_params_.kv_int8
                            && kv_cache_.is_fast_position(pos);

        scratch_.reset();
        // F4a token tracking: the decode step is about to write KV[pos] =
        // last_token_, then sample a new token. If `pos` extends the
        // committed prefix, record what's being written. (Fallback path's
        // first iteration overwrites KV[N-1] in place — `pos == N-1` and
        // kv_tokens.size() == N already, so the condition skips and we
        // don't double-count.) Runs for both the graph and per-step paths.
        if ((int)kv_tokens.size() == pos) {
            kv_tokens.push_back((uint32_t)last_token_);
        }
        int token;
        if (use_graph) {
            token = decode_step_graph(pos);
        } else {
            token = decode_step(pos);
            if (decode_graph_enabled() && !graph_captured_
                && !gen_params_.kv_int8 && kv_cache_.is_fast_position(pos)) {
                // After the first per-token step succeeds, capture the
                // graph for all future steps in this generate() call.
                build_cuda_graph(pos);
            }
        }
        pos++;

        bool is_eos = (token == tokenizer_.eos_id);

        // Stamp TTFT on the first sampled token. Only fires when Path A
        // did NOT already stamp it (i.e. fallback path), gated on
        // ttft_ms == 0.
        if (stats.ttft_ms == 0.0f) {
            stats.ttft_ms = Ms(Clock::now() - t_request).count();
        }

        if (token_cb) {
            std::string text = tokenizer_.decode(token);
            token_cb(text.c_str(), is_eos);
        }

        stats.completion_tokens++;
        if (is_eos) break;

        if (i % 10 == 0) {
            OOMGuard g(0);
            int64_t used = budget_.total_mb - g.real_free_mb();
            peak_mem = std::max(peak_mem, used);
            peak_temp = std::max(peak_temp, read_thermal().gpu_temp_c);
        }
    }

    auto t3 = Clock::now();
    stats.decode_ms = Ms(t3 - t2).count();
    if (stats.completion_tokens > 0)
        stats.decode_tok_per_sec = stats.completion_tokens / (stats.decode_ms / 1000.0f);
    stats.peak_memory_mb = peak_mem;
    stats.peak_thermal_c = peak_temp;

    fprintf(stderr, "[engine] Decode: %d tokens in %.0f ms (%.1f tok/s)\n",
            stats.completion_tokens, stats.decode_ms, stats.decode_tok_per_sec);
    if (stats.ttft_ms > 0) {
        fprintf(stderr, "[engine] TTFT (first token): %.0f ms\n", stats.ttft_ms);
    }

    // Path F (#45) F3: save the GPU-resident KV state at turn end if
    // the caller tagged the request with a valid conversation_id.
    // F4 will add the corresponding hydrate-on-start hook so the next
    // turn skips prefill for the matching prefix.
    //
    // Skip rules:
    //   - conv_id empty / invalid → single-shot mode, no save
    //   - total tokens < MIN_TOKENS_TO_SAVE → trivial request, not
    //     worth the eMMC write
    //   - overflow pool was used → fast pool no longer represents the
    //     full state; v1 deliberately can't recover those conversations
    if (!params.conversation_id.empty() &&
        validate_conversation_id(params.conversation_id))
    {
        constexpr int MIN_TOKENS_TO_SAVE = 4;
        // F4a: committed_tokens is the number of KV positions actually
        // written. Equals prompt_tokens + completion_tokens - 1 in the
        // typical Path A case (the last sampled token lives in
        // last_token_ but isn't in KV) and matches the kv_tokens vector
        // we've been tracking through the decode loop.
        const int committed_tokens = (int)kv_tokens.size();
        const auto& kv_cfg = kv_cache_.config();
        const bool overflow_used = committed_tokens > kv_cfg.max_context;

        if (committed_tokens < MIN_TOKENS_TO_SAVE) {
            fprintf(stderr, "[kv_cache] skip save (only %d tokens, "
                            "min=%d)\n",
                    committed_tokens, MIN_TOKENS_TO_SAVE);
        } else if (overflow_used) {
            fprintf(stderr, "[kv_cache] skip save (overflow pool used, "
                            "v1 fast-pool-only)\n");
        } else if (params.kv_int8) {
            // alpha.12 Path I + Path F interop guard: the on-disk format
            // (v2, #50) doesn't yet carry the per-head INT8 scales that
            // attention needs at hydrate time. Saving INT8 bytes without
            // the scales would produce silently-garbled output on the
            // next turn. Skip save until Path F format v3 lands the
            // scales region.
            static bool warned = false;
            if (!warned) {
                fprintf(stderr,
                        "[kv_cache] skip save (INT8 KV active — Path F "
                        "format v2 doesn't yet carry per-head scales; "
                        "format v3 tracked separately. Use --fp16-kv if "
                        "you need persistent KV today.)\n");
                warned = true;
            }
        } else {
            const std::string dir  = default_kv_cache_dir();
            const std::string path = path_for_conv_id(dir, params.conversation_id);

            if (!ensure_dir(dir)) {
                fprintf(stderr, "[kv_cache] skip save (dir unavailable)\n");
            } else {
                const int64_t packed_bytes =
                    kv_cache_.packed_used_bytes(committed_tokens);

                KVCacheFileHeader hdr{};
                hdr.magic         = KVCACHE_MAGIC;
                hdr.version       = KVCACHE_VERSION;
                hdr.n_layers      = (uint32_t)kv_cfg.n_layers;
                hdr.n_kv_heads    = (uint32_t)kv_cfg.n_kv_heads;
                hdr.head_dim      = (uint32_t)kv_cfg.head_dim;
                hdr.kv_type_bytes = (uint32_t)kv_cfg.kv_type_bytes;
                hdr.max_context   = (uint32_t)kv_cfg.max_context;
                hdr.used_tokens   = (uint32_t)committed_tokens;
                hdr.body_bytes    = (uint64_t)packed_bytes;
                std::memcpy(hdr.model_hash, model_fingerprint_,
                            sizeof(hdr.model_hash));

                auto t_save0 = Clock::now();
                std::vector<uint8_t> staging((size_t)packed_bytes);
                bool ok = kv_cache_.gather_used_to_host(staging.data(),
                                                       committed_tokens);
                if (ok) {
                    ok = save_kv_to_file(path, hdr,
                                         kv_tokens.data(),
                                         (size_t)committed_tokens,
                                         staging.data(),
                                         (size_t)packed_bytes);
                }
                auto t_save1 = Clock::now();
                float save_ms = Ms(t_save1 - t_save0).count();
                if (ok) {
                    const size_t tokens_bytes =
                        (size_t)committed_tokens * sizeof(uint32_t);
                    fprintf(stderr,
                            "[kv_cache] Saved %lu KV bytes + %zu tokens "
                            "(%zu B) to %s (%d tokens committed, %.0f ms)\n",
                            (unsigned long)hdr.body_bytes,
                            (size_t)committed_tokens, tokens_bytes,
                            path.c_str(), committed_tokens, save_ms);

                    // Path F5: enforce per-process cache budget + clean
                    // up any stale .tmp left by a previous crashed run.
                    // Defaults: 1 GB budget, 60-s stale-tmp threshold.
                    // Set JLLM_KV_CACHE_MAX_MB=0 to disable eviction;
                    // set JLLM_KV_CACHE_STALE_TMP_S=0 to keep .tmp files.
                    cleanup_stale_tmp_files(dir,
                                            default_kv_cache_stale_tmp_s());
                    enforce_kv_cache_budget(
                        dir,
                        default_kv_cache_max_mb() * (int64_t)(1024 * 1024),
                        path);
                } else {
                    fprintf(stderr,
                            "[kv_cache] Save failed for %s (see prior "
                            "error)\n", path.c_str());
                }
            }
        }
    }

    // Remember the tokens now resident in the KV buffers so the next request
    // can reuse the common prefix. Skip if the KV overflowed the fast pool
    // (the contiguous-prefix invariant the in-memory reuse relies on no longer
    // holds) — clearing forces a clean cold prefill next time.
    if ((int)kv_tokens.size() <= kv_cache_.config().max_context) {
        resident_tokens_ = std::move(kv_tokens);
    } else {
        resident_tokens_.clear();
    }

    return stats;
}

void Engine::stop() { stop_flag_ = true; }
LiveStats Engine::stats() const { return read_live_stats(); }

}  // namespace jllm
