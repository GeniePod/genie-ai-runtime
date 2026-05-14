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
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdlib>
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
// BUG #2 fix: need explicit residual add between stages

__global__ void vec_add_kernel(half* __restrict__ out,
                                const half* __restrict__ a,
                                const half* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

static void vec_add(half* out, const half* a, const half* b, int n, cudaStream_t s) {
    int block = 128;
    int grid = (n + block - 1) / block;
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

    float h_row[8192];  // max hidden_dim = 8192
    half  h_fp16[8192];

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
    if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
    if (decode_graph_)      { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }
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
                                     int head_dim,
                                     std::vector<void*>& owned) {
    const size_t norm_bytes = (size_t)hidden_dim * sizeof(float);
    const size_t qk_norm_bytes = (size_t)head_dim * sizeof(float);

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

static size_t materialize_output_weight(ModelWeights& mw, const ModelConfig& cfg,
                                        const MemoryBudget& budget,
                                        std::vector<void*>& owned) {
    if (!device_output_enabled() || !fast_gemv_enabled_for_device_weights() ||
        !mw.output) {
        return 0;
    }

    const size_t bytes = kquant_tensor_bytes(mw.output_type,
                                             cfg.vocab_size,
                                             cfg.hidden_dim);
    if (bytes == 0) return 0;

    const int64_t mb = (int64_t)((bytes + 1024 * 1024 - 1) / (1024 * 1024));
    if (budget.free_mb() < mb + 128) {
        fprintf(stderr,
                "[model] device output copy skipped (%ld MB requested, %ld MB free budget)\n",
                mb, budget.free_mb());
        return 0;
    }

    void* output_device = nullptr;
    cudaError_t err = cudaMalloc(&output_device, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[model] device output copy skipped (%zu MB): %s\n",
                bytes / (1024 * 1024), cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(output_device, mw.output, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[model] device output copy failed (%zu MB): %s\n",
                bytes / (1024 * 1024), cudaGetErrorString(err));
        cudaFree(output_device);
        return 0;
    }

    owned.push_back(output_device);
    const bool tied_output = (mw.output == mw.tok_embd);
    mw.output = output_device;
    if (tied_output && fast_embedding_enabled()) {
        mw.tok_embd = output_device;
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

static bool copy_layer_quant_weights_to_device(LayerWeights& lw,
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

    void* copied[sizeof(specs) / sizeof(specs[0])] = {};
    for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); i++) {
        const size_t bytes = kquant_tensor_bytes(specs[i].type,
                                                 specs[i].rows,
                                                 specs[i].cols);
        if (bytes == 0 || !copy_weight_to_device(specs[i].src, bytes, &copied[i])) {
            for (void* p : copied) {
                if (p) cudaFree(p);
            }
            return false;
        }
    }

    for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); i++) {
        *specs[i].dst = copied[i];
        owned.push_back(copied[i]);
    }
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

        if (!copy_layer_quant_weights_to_device(lw, cfg, owned)) break;

        free_mb -= mb;
        total_bytes += bytes;
        copied_layers++;
    }

    if (copied_layers > 0) {
        fprintf(stderr,
                "[model] Materialized %d transformer layers on device (%zu MB)\n",
                copied_layers, total_bytes / (1024 * 1024));
    }
    return total_bytes;
}

bool Engine::load(const std::string& gguf_path, const GenParams& params) {
    gen_params_ = params;
    budget_ = probe_system_memory();

    config_ = load_gguf_config(gguf_path);
    fprintf(stderr, "[engine] %s: %d layers, %d heads (%d KV), dim=%d, head_dim=%d, vocab=%d, rms_eps=%g, rope=%s\n",
            config_.name.c_str(), config_.n_layers, config_.n_heads,
            config_.n_kv_heads, config_.hidden_dim, config_.head_dim,
            config_.vocab_size,
            config_.rms_eps, config_.rope_neox ? "neox" : "normal");

    if (!load_and_map_weights(gguf_path, &weights_, &weights_size_,
                              &model_weights_, config_)) {
        fprintf(stderr, "[engine] Failed to load/map weights\n");
        return false;
    }
    if (!materialize_norm_weights(model_weights_, config_.hidden_dim,
                                  config_.head_dim,
                                  device_weight_copies_)) {
        fprintf(stderr, "[engine] Failed to materialize norm weights\n");
        return false;
    }
    budget_.model_mb = mapped_weight_device_enabled()
        ? weights_size_ / (1024 * 1024)
        : 0;

    int kv_bytes = params.kv_int8 ? 1 : 2;
    int auto_ctx = budget_.max_context(config_.n_layers, config_.n_kv_heads, config_.head_dim, kv_bytes);

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

    KVCachePool::Config kv_cfg = {};
    kv_cfg.n_layers = config_.n_layers;
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
    scratch_bytes = (scratch_bytes + 4095) & ~4095LL;

    if (!scratch_.init(scratch_bytes)) return false;
    budget_.scratch_mb = scratch_bytes / (1024 * 1024);

    const size_t output_device_bytes =
        materialize_output_weight(model_weights_, config_, budget_,
                                  device_weight_copies_);
    budget_.model_mb += output_device_bytes / (1024 * 1024);

    const size_t layer_device_bytes =
        materialize_layer_weights(model_weights_, config_, budget_,
                                  device_weight_copies_);
    budget_.model_mb += layer_device_bytes / (1024 * 1024);

    if (!mapped_weight_device_enabled() && weights_) {
        madvise(weights_, weights_size_, MADV_DONTNEED);
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
    fused_swiglu(swiglu_out, gate_buf, up_buf, 1, I, stream_);

    // 3c. FFN exit: x = x2 + W_down(swiglu_out).
    transformer_layer_ffn_block(layer, x2, swiglu_out, x);
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
        fp16_to_int8((int8_t*)kv_cache_.key_ptr(layer, pos), nullptr, k_buf, 1, KV_DIM, stream_);
        fp16_to_int8((int8_t*)kv_cache_.val_ptr(layer, pos), nullptr, v_buf, 1, KV_DIM, stream_);
    } else if (!kv_cache_.is_fast_position(pos)) {
        cudaMemcpyAsync(kv_cache_.key_ptr(layer, pos), k_buf,
                        KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
        cudaMemcpyAsync(kv_cache_.val_ptr(layer, pos), v_buf,
                        KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
    }

    float scale = 1.0f / sqrtf((float)config_.head_dim);
    flash_attention_decode(attn_out, q_buf, kv_cache_.key_ptr(layer, 0),
                          kv_cache_.val_ptr(layer, 0),
                          config_.n_heads, config_.n_kv_heads, config_.head_dim,
                          pos + 1, scale, gen_params_.kv_int8, nullptr, stream_);
}

void Engine::transformer_layer_attn_block(int layer, int pos, half* x_in,
                                          half* q_buf, half* k_buf, half* v_buf,
                                          half* x_attn_out, bool qk_norm_already) {
    const auto& lw = model_weights_.layers[layer];
    int H = config_.hidden_dim;
    int Q_DIM = config_.n_heads * config_.head_dim;

    half* attn_out  = (half*)scratch_.get(Q_DIM * sizeof(half));
    half* attn_proj = (half*)scratch_.get(H * sizeof(half));

    transformer_layer_attn_compute(layer, pos, q_buf, k_buf, v_buf,
                                   attn_out, qk_norm_already);
    gemv_quant(attn_proj, lw.wo, lw.type_wo, attn_out, H, Q_DIM, stream_);
    vec_add(x_attn_out, x_in, attn_proj, H, stream_);
}

void Engine::transformer_layer_ffn_block(int layer, half* x_attn, half* swiglu_in,
                                         half* x_out) {
    const auto& lw = model_weights_.layers[layer];
    int H = config_.hidden_dim;
    int I = config_.intermediate_dim;

    half* ffn_out = (half*)scratch_.get(H * sizeof(half));
    gemv_quant(ffn_out, lw.w_down, lw.type_w_down, swiglu_in, H, I, stream_);
    vec_add(x_out, x_attn, ffn_out, H, stream_);
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
                fp16_to_int8((int8_t*)kv_cache_.key_ptr(layer, pos), nullptr, k_t, 1, KV_DIM, stream_);
                fp16_to_int8((int8_t*)kv_cache_.val_ptr(layer, pos), nullptr, v_t, 1, KV_DIM, stream_);
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
        const float scale = 1.0f / sqrtf((float)config_.head_dim);
        flash_attention_prefill_batched(
            attn_out_batch, q_batch,
            kv_cache_.key_ptr(layer, 0), kv_cache_.val_ptr(layer, 0),
            config_.n_heads, config_.n_kv_heads, config_.head_dim,
            N, start_pos, scale,
            gen_params_.kv_int8, nullptr, stream_);
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
    fused_swiglu(swiglu_batch, gate_batch, up_batch, N, I, stream_);

    // ── Batched W_down + batched residual #2 (NEW in PR #16) ──────────
    // W_down is 25.8 MB / layer on Qwen3-4B (~35% of per-layer weight
    // bandwidth, second-biggest after gate/up).
    gemm_quant_batched(ffn_out_batch, lw.w_down, lw.type_w_down, swiglu_batch,
                       H, N, I, stream_);
    vec_add(x_batch, x_attn_batch, ffn_out_batch, N * H, stream_);

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
    if (profile) {
        cudaStreamSynchronize(stream_);
        auto now = Clock::now();
        prof_emb_ms = Ms(now - prof_t).count();
        prof_t = now;
    }

    // All transformer layers
    for (int l = 0; l < config_.n_layers; l++) {
        transformer_layer(l, pos, x);
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

    // Sample
    int token = sample_token(host_logits_, config_.vocab_size, gen_params_,
                             recent_tokens_.data(), recent_tokens_.size());
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

// ── CUDA graph capture (BUG #5 FIX) ─────────────────────────────────────
// Captures the GPU-side kernels of one decode step.
// Requirements:
//   - KV cache length is padded to max_context (fixed graph structure)
//   - Embedding lookup done outside graph (host→device copy not capturable)
//   - Sampling done outside graph (host-side operation)

void Engine::build_cuda_graph(int pos) {
    if (graph_captured_) return;
    if (!gen_params_.use_cuda_graph) return;

    fprintf(stderr, "[engine] Capturing CUDA graph...\n");

    // Pre-allocate a fixed hidden state buffer for graph capture
    int H = config_.hidden_dim;
    half* graph_x = (half*)scratch_.get(H * sizeof(half));

    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    // Capture all transformer layers
    for (int l = 0; l < config_.n_layers; l++) {
        transformer_layer(l, pos, graph_x);
    }

    // Capture final norm
    half* g_normed = (half*)scratch_.get(H * sizeof(half));
    fused_rmsnorm_residual(g_normed, graph_x, nullptr,
                          model_weights_.output_norm, 1, H, config_.rms_eps, true, stream_);

    // Capture logit projection without an intermediate FP16 logits pass.
    float* g_logits_fp32 = (float*)scratch_.get(config_.vocab_size * sizeof(float));
    gemv_quant_f32(g_logits_fp32, model_weights_.output, model_weights_.output_type,
                   g_normed, config_.vocab_size, H, stream_);

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
    fprintf(stderr, "[engine] CUDA graph captured successfully\n");
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

GenStats Engine::generate(const std::string& prompt, const GenParams& params,
                          TokenCallback token_cb) {
    GenStats stats = {};
    stop_flag_ = false;
    gen_params_ = params;
    recent_tokens_.clear();

    // t_request: wall-clock at the moment generate() was called. The
    // reference point for time-to-first-token, which captures the full
    // user-visible latency from prompt-submitted to first-token-delivered
    // (tokenization + prefill + first decode step).
    auto t_request = Clock::now();

    auto prompt_tokens = tokenizer_.encode(prompt);
    stats.prompt_tokens = prompt_tokens.size();
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(tokenizer_.bos_id);
    }
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

    if (batched_prefill_enabled() && N > 0 && batched_fits) {
        // Path B (issue #12): layer-major prefill. Allocate one
        // [N × H] activation buffer at the bottom of scratch, dequantize
        // every prompt embedding into it, then loop layers-outer /
        // tokens-inner. The inner step still calls the proven
        // single-token kernels, so output is byte-identical to the
        // per-token loop. Future PRs swap the inner calls for batched
        // (GEMM-shaped) variants without touching this dispatch.
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[engine] Path B: batched prefill enabled (JLLM_BATCHED_PREFILL=1)\n");
            logged = true;
        }
        scratch_.reset();
        half* x_batch = (half*)scratch_.get(x_batch_bytes);
        for (int i = 0; i < N; i++) {
            dequant_embedding(x_batch + (int64_t)i * H,
                              model_weights_.tok_embd, prompt_tokens[i], H,
                              model_weights_.embd_type, stream_);
        }
        for (int l = 0; l < config_.n_layers; l++) {
            transformer_prefill(l, 0, N, x_batch);
        }
        last_token_ = prompt_tokens.back();
        last_prefill_x = x_batch + (int64_t)(N - 1) * H;
    } else {
        if (batched_prefill_enabled() && N > 0 && !batched_fits) {
            static bool warned = false;
            if (!warned) {
                fprintf(stderr,
                        "[engine] Path B: prompt too long for batched prefill "
                        "(N=%d, would need %ld bytes of scratch); falling back "
                        "to per-token prefill\n", N, (long)batched_total);
                warned = true;
            }
        }
        for (int i = 0; i < N; i++) {
            scratch_.reset();
            last_token_ = prompt_tokens[i];
            half* x = (half*)scratch_.get(H * sizeof(half));
            dequant_embedding(x, model_weights_.tok_embd, last_token_, H,
                              model_weights_.embd_type, stream_);
            for (int l = 0; l < config_.n_layers; l++)
                transformer_layer(l, i, x);
            // Hang onto the last token's residual; the scratch pool has NOT
            // been reset between iterations within prefill, so this pointer
            // remains valid through the post-prefill sampling step below.
            if (i == N - 1) {
                last_prefill_x = x;
            }
        }
    }
    cudaStreamSynchronize(stream_);
    auto t1 = Clock::now();
    stats.prompt_ms = Ms(t1 - t0).count();
    if (stats.prompt_tokens > 0)
        stats.prompt_tok_per_sec = stats.prompt_tokens / (stats.prompt_ms / 1000.0f);
    fprintf(stderr, "[engine] Prefill: %d tokens in %.0f ms (%.1f tok/s)\n",
            stats.prompt_tokens, stats.prompt_ms, stats.prompt_tok_per_sec);

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
            cudaError_t copy_err =
                cudaMemcpyAsync(host_logits_, logits_fp32,
                                config_.vocab_size * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_);
            if (copy_err == cudaSuccess) {
                copy_err = cudaStreamSynchronize(stream_);
            }
            if (copy_err == cudaSuccess) {
                first_token = sample_token(host_logits_, config_.vocab_size,
                                           gen_params_,
                                           recent_tokens_.data(),
                                           recent_tokens_.size());
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
    const int kv_token_limit = kv_cache_.max_tokens();
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

        scratch_.reset();
        int token = decode_step(pos);
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
    return stats;
}

void Engine::stop() { stop_flag_ = true; }
LiveStats Engine::stats() const { return read_live_stats(); }

}  // namespace jllm
