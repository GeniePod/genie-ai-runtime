// jllm_engine.h — LLM inference engine (Jetson-native)

#pragma once

#include "jllm_memory.h"
#include "jllm_jetson.h"
#include "jllm_kernels.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <utility>

namespace jllm {

// ── Model config (read from GGUF header) ─────────────────────────────────

struct ModelConfig {
    std::string name;
    int         n_layers       = 0;
    int         n_heads        = 0;
    int         n_kv_heads     = 0;
    int         head_dim       = 0;
    int         hidden_dim     = 0;
    int         intermediate_dim = 0;
    int         vocab_size     = 0;
    int         max_seq_len    = 0;
    float       rope_theta     = 0;
    float       rms_eps        = 1e-5f;
    bool        rope_neox      = false;
    int         quant_type     = 0;

    int gqa_group_size() const { return n_kv_heads > 0 ? n_heads / n_kv_heads : 1; }
    int64_t weight_bytes() const;
    int64_t kv_per_token_bytes(int kv_type_bytes) const {
        return 2LL * n_layers * n_kv_heads * head_dim * kv_type_bytes;
    }
};

// ── Generation parameters ────────────────────────────────────────────────

struct GenParams {
    int   max_tokens    = 256;
    float temperature   = 0.7f;
    int   top_k         = 40;
    float top_p         = 0.9f;
    float repeat_penalty = 1.1f;
    int   context_limit = 0;
    bool  use_cuda_graph = true;
    bool  kv_int8       = false;
};

using TokenCallback = std::function<void(const char* text, bool is_eos)>;

struct GenStats {
    int     prompt_tokens      = 0;
    int     completion_tokens  = 0;
    float   prompt_ms          = 0;
    float   decode_ms          = 0;
    // Time-to-first-token: end-to-end wall clock from the start of
    // generate() (prompt available) to the moment the first sampled
    // token is delivered to the user via token_cb. Includes
    // tokenization, prefill, and the first decode step. The most
    // user-visible latency number for interactive workloads.
    float   ttft_ms            = 0;
    float   prompt_tok_per_sec = 0;
    float   decode_tok_per_sec = 0;
    int64_t peak_memory_mb     = 0;
    float   peak_thermal_c     = 0;
    int     oom_stops          = 0;
    int     thermal_pauses     = 0;
};

// ── Layer weights (pointers into mmap'd GGUF blob) ───────────────────────

struct LayerWeights {
    const void* wq       = nullptr;
    const void* wk       = nullptr;
    const void* wv       = nullptr;
    const void* wo       = nullptr;
    const void* w_gate   = nullptr;
    const void* w_up     = nullptr;
    const void* w_down   = nullptr;
    const void* rms_attn = nullptr;  // FP32 or FP16 depending on GGUF
    const void* rms_ffn  = nullptr;  // FP32 or FP16 depending on GGUF
    const void* q_norm   = nullptr;  // Qwen3 per-head Q RMSNorm [head_dim]
    const void* k_norm   = nullptr;  // Qwen3 per-head K RMSNorm [head_dim]
    int type_wq     = 12;            // GGML tensor type for quantized matvec
    int type_wk     = 12;
    int type_wv     = 12;
    int type_wo     = 12;
    int type_w_gate = 12;
    int type_w_up   = 12;
    int type_w_down = 12;
    int rms_type = 0;                // 0=F32, 1=F16
    int qk_norm_type = 0;            // 0=F32, 1=F16
    const half* sq       = nullptr;
    const half* sk       = nullptr;
    const half* sv       = nullptr;
    const half* so       = nullptr;
    const half* s_gate   = nullptr;
    const half* s_up     = nullptr;
    const half* s_down   = nullptr;
};

struct ModelWeights {
    const void*     tok_embd    = nullptr;   // F32 or F16
    const void*     output_norm = nullptr;  // F32 or F16
    int             embd_type   = 0;        // 0=F32, 1=F16
    const void*     output      = nullptr;
    int             output_type = 12;       // GGML tensor type for output projection
    const half*     s_output    = nullptr;
    LayerWeights*   layers      = nullptr;
    int             n_layers    = 0;
};

// ── Tokenizer ────────────────────────────────────────────────────────────

struct Tokenizer {
    std::vector<std::string> vocab;
    int bos_id = 1;
    int eos_id = 2;

    bool load_from_gguf(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& ids) const;

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> bpe_ranks_;
    std::vector<std::pair<std::string, int>> special_tokens_;
    std::vector<int> sorted_by_len_;
    int max_token_len_ = 0;
    bool byte_encode_ = true;
    bool add_bos_ = true;
    bool add_eos_ = false;
};

// ── Sampling ─────────────────────────────────────────────────────────────

int sample_token(float* logits, int vocab_size, const GenParams& params,
                 const int* recent_tokens = nullptr, int n_recent = 0);

// ── GGUF loaders ─────────────────────────────────────────────────────────

ModelConfig   load_gguf_config(const std::string& path);
bool          load_gguf_weights(const std::string& path, void** weights, int64_t* size);
bool          load_and_map_weights(const std::string& path, void** blob, int64_t* blob_size,
                                   ModelWeights* mw, const ModelConfig& cfg);
ModelWeights  map_weights(const void* blob, int64_t blob_size, const ModelConfig& cfg);

// ── Engine ───────────────────────────────────────────────────────────────

class Engine {
public:
    Engine();
    ~Engine();

    bool load(const std::string& gguf_path, const GenParams& params);
    void unload();
    GenStats generate(const std::string& prompt, const GenParams& params,
                      TokenCallback token_cb = nullptr);
    void stop();

    bool          is_loaded() const { return loaded_; }
    ModelConfig   config() const { return config_; }
    MemoryBudget  memory() const { return budget_; }
    LiveStats     stats() const;

private:
    bool loaded_ = false;
    bool stop_flag_ = false;

    ModelConfig   config_;
    GenParams     gen_params_;
    MemoryBudget  budget_;
    KVCachePool   kv_cache_;
    ScratchPool   scratch_;

    void*         weights_ = nullptr;
    int64_t       weights_size_ = 0;
    ModelWeights  model_weights_ = {};
    std::vector<void*> device_weight_copies_;
    Tokenizer     tokenizer_;

    int           last_token_ = 0;
    std::vector<int> recent_tokens_;

    cudaStream_t    stream_ = nullptr;
    cudaGraph_t     decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    bool            graph_captured_ = false;
    float*          host_logits_ = nullptr;
    int             host_logits_capacity_ = 0;

    void transformer_layer(int layer, int pos, half* x);
    // Single-token attention pre-Wo: QK-norm (if not already done by the
    // caller), RoPE, KV store, attention. Writes attention output (head
    // mix) to `attn_out` [Q_DIM]. No projection, no residual — caller is
    // expected to do Wo + first residual, batched or not.
    //
    // This split was introduced so transformer_prefill can do per-token
    // attention compute (the RoPE/KV-store order has to stay sequential
    // until chunked-prefill attention lands) followed by a single batched
    // Wo and a single batched residual.
    void transformer_layer_attn_compute(int layer, int pos,
                                        half* q_buf, half* k_buf, half* v_buf,
                                        half* attn_out, bool qk_norm_already);
    // Single-token attention block: attn_compute + Wo + first residual.
    // Thin wrapper used by the per-token transformer_layer path.
    //
    // `q_buf`, `k_buf`, `v_buf` are the token's projections. `x_in` is the
    // layer input (residual #1's left operand); `x_attn_out` receives
    // x_in + Wo(attn). `qk_norm_already` is true when the caller batched
    // QK-norm.
    void transformer_layer_attn_block(int layer, int pos,
                                      half* x_in,
                                      half* q_buf, half* k_buf, half* v_buf,
                                      half* x_attn_out, bool qk_norm_already);
    // Single-token FFN exit: x_out = x_attn + W_down(swiglu_in).
    //
    // `swiglu_in` carries the post-SwiGLU intermediate-dim activations,
    // produced per-token in transformer_layer or batched in
    // transformer_prefill.
    void transformer_layer_ffn_block(int layer,
                                     half* x_attn, half* swiglu_in,
                                     half* x_out);
    // Process a contiguous batch of `n_tokens` prompt tokens through one
    // transformer layer in a single high-level call. Scaffolding for the
    // Path B batched-prefill effort (#12) — the initial implementation
    // delegates to transformer_layer N times, so it produces byte-equal
    // output but no speedup. Subsequent PRs will replace the inner
    // kernel calls with batched (GEMM-shaped) variants and the speedup
    // arrives without changing this entry point.
    //
    // Inputs/outputs live in `x_batch` which is laid out as
    // [n_tokens × hidden_dim] (row-major: token 0's hidden state in
    // rows 0..hidden_dim-1, etc.). x_batch is read AND written in
    // place per layer (residual additions).
    void transformer_prefill(int layer, int start_pos, int n_tokens, half* x_batch);
    int  decode_step(int pos);
    void build_cuda_graph(int pos);
    bool check_memory_and_thermal(int pos);
    // Has the JLLM_BATCHED_PREFILL env var been set to a non-zero value?
    // Cached on first read.
    bool batched_prefill_enabled() const;
};

}  // namespace jllm
