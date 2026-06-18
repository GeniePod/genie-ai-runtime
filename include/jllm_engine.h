// jllm_engine.h — LLM inference engine (Jetson-native)

#pragma once

#include "jllm_memory.h"
#include "jllm_jetson.h"
#include "jllm_kernels.h"
#include "jllm_grammar.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <utility>

namespace jllm {

// ── Architecture recipe ──────────────────────────────────────────────────
// The forward pass was originally hardcoded to the Qwen3 / Llama-3 shape.
// `Arch` + `ArchSpec` make the model-family-specific behavior data-driven, so a
// new architecture (Gemma 4, ...) is a *recipe* rather than a fork of the hot
// path. The default recipe reproduces the original Qwen/Llama behavior exactly;
// Gemma-specific knobs are off by default. See docs/gemma4-plan.md and issue #89.

enum class Arch {
    LlamaQwen,   // Llama-3 / Qwen2 / Qwen3 family — the original hardcoded path
    Gemma4,      // Gemma 4 (PLE, dual head dim, dual RoPE, GeGLU, 4 norms, ...)
    Unknown,
};

enum class FfnActivation {
    SiluSwiGLU,  // silu(gate) * up      — Llama / Qwen
    GeluGeGLU,   // gelu_tanh(gate) * up — Gemma
};

// Per-model-family forward recipe, read by the engine instead of hardcoded
// constants.
struct ArchSpec {
    Arch          arch                 = Arch::LlamaQwen;
    FfnActivation ffn_activation       = FfnActivation::SiluSwiGLU;
    bool          scale_embeddings     = false;  // token embeds *= sqrt(hidden_dim)
    float         attn_scale           = 0.0f;   // 0 => default 1/sqrt(head_dim)
    float         final_logit_softcap  = 0.0f;   // 0 => off; else c*tanh(logit/c)
    // Norm layout. LlamaQwen: input + pre-ffn (post-attention) norms only.
    // Gemma 4 adds a post-attention norm (attn output, pre-residual) and a
    // post-ffn norm (ffn output, pre-residual).
    bool          post_attn_norm       = false;
    bool          post_ffn_norm        = false;
    // Per-attention-type behavior (Gemma 4: sliding/local vs full/global layers).
    bool          per_layer_head_dim   = false;  // sliding 256 / full 512
    bool          dual_rope            = false;  // local vs global theta + rotary frac
    bool          sliding_window       = false;  // alternating local/global mask
    int           kv_shared_layers     = 0;      // trailing layers reusing K/V (0 => none)
    bool          per_layer_embeddings = false;  // PLE auxiliary pathway (Gemma 4 E2B)
};

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

    // Architecture family + forward recipe (issue #89). `arch_name` is the raw
    // GGUF general.architecture string; `spec` data-drives the forward pass.
    std::string arch_name;
    Arch        arch           = Arch::LlamaQwen;
    ArchSpec    spec;

    // Gemma 4 numeric hparams (valid only when arch == Arch::Gemma4). Head dim
    // differs by attention type: sliding (local) layers use `head_dim`; full
    // (global) layers use `global_head_dim`.
    int         global_head_dim        = 0;   // gemma4.attention.key_length (full)
    int         sliding_head_dim       = 0;   // gemma4.attention.key_length_swa (local)
    int         sliding_window         = 0;   // gemma4.attention.sliding_window
    int         sliding_window_pattern = 0;   // gemma4: 6 => 5 local : 1 global
    int         n_kv_shared_layers     = 0;   // gemma4.attention.shared_kv_layers
    int         ple_input_dim          = 0;   // gemma4.embedding_length_per_layer_input
    float       rope_theta_swa         = 0.0f;// gemma4.rope.freq_base_swa (local)
    // gemma4 per-layer tables (parsed from GGUF arrays).
    std::vector<int> ff_per_layer;            // feed_forward_length per layer (6144 / 12288)
    std::vector<int> layer_is_sliding;        // 1 = sliding (local), 0 = full (global)

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
    // alpha.12: INT8 KV is the default. Saves ~50 % of KV pool memory
    // (e.g. 144 MB → 74 MB for Qwen3-4B at 1024 ctx) with FP16-ULP-
    // bounded quality drift validated in Path I (#62). Set false via
    // CLI --fp16-kv or HTTP body field to opt back into full FP16 KV.
    bool  kv_int8       = true;

    // Path F (#45): persistent KV cache identifier. Empty = single-shot,
    // no cross-turn state. Non-empty = the engine will (in F3+) try to
    // load <conversation_id>.bin at request start and write it back at
    // turn end. Plumbed-but-ignored in F2; engine logs it for now.
    //
    // Validated by validate_conversation_id(): only [A-Za-z0-9_-] and
    // ≤ 64 chars accepted. Invalid IDs are dropped at the CLI / HTTP
    // boundary, never reach the engine.
    std::string conversation_id;

    // #86: GBNF grammar for constrained decoding. Empty = unconstrained
    // (default; zero overhead). When non-empty and it parses, the sampler
    // masks every token that can't continue a valid string under the
    // grammar, guaranteeing structurally-valid output (e.g. tool-call JSON).
    std::string grammar;
    std::string grammar_root = "root";  // entry rule name
};

// Returns true if `id` is empty or matches /^[A-Za-z0-9_-]{1,64}$/.
// Empty is treated as "feature disabled", not an error. Locked down
// before F3 starts using the value to build file paths.
bool validate_conversation_id(const std::string& id);

using TokenCallback = std::function<void(const char* text, bool is_eos)>;

struct GenStats {
    int     prompt_tokens      = 0;
    int     prefill_tokens     = 0;
    int     completion_tokens  = 0;
    int     kv_cache_reused_tokens = 0;
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
    const void* q_norm   = nullptr;  // Qwen3/Gemma per-head Q RMSNorm [head_dim]
    const void* k_norm   = nullptr;  // Qwen3/Gemma per-head K RMSNorm [head_dim]
    // Gemma 4 extras (PLE-only architecture; nullptr for other models).
    const void* post_attn_norm = nullptr;  // post_attention_norm.weight (sandwich)
    const void* post_ffn_norm  = nullptr;  // post_ffw_norm.weight (sandwich)
    const void* ple_gate       = nullptr;  // inp_gate.weight  (PLE input gate)
    const void* ple_proj       = nullptr;  // proj.weight      (PLE projection)
    const void* ple_norm       = nullptr;  // post_norm.weight (PLE post-norm)
    const void* layer_scale    = nullptr;  // layer_output_scale.weight
    float layer_scale_val = 1.0f;          // scalar value (read at load; 1.0 = no-op)
    int type_ple_gate = 12;
    int type_ple_proj = 12;
    // Per-layer geometry (Gemma 4: sliding/local vs full/global layers differ).
    int   head_dim_l     = 0;      // head dim (256 sliding / 512 global)
    int   intermediate_l = 0;      // ffn width (6144 / 12288)
    bool  is_sliding     = false;  // sliding (local) vs full (global) attention
    float rope_theta_l   = 0.0f;   // RoPE base (1e4 sliding / 1e6 global)
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
    // Gemma 4 Per-Layer Embeddings (PLE) — model-level tensors (nullptr otherwise).
    const void*     ple_embd          = nullptr;  // per_layer_token_embd.weight
    const void*     ple_model_proj    = nullptr;  // per_layer_model_proj.weight
    const void*     ple_proj_norm     = nullptr;  // per_layer_proj_norm.weight
    int             ple_embd_type       = 0;
    int             ple_model_proj_type = 12;
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
    // SentencePiece/unigram encode (Gemma, Llama-SPM): Viterbi over piece
    // scores with byte fallback. Used when is_spm_; the BPE path is unchanged.
    void encode_spm(const std::string& text, std::vector<int>& out) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> bpe_ranks_;
    std::vector<std::pair<std::string, int>> special_tokens_;
    std::vector<int> sorted_by_len_;
    std::vector<float> token_scores_;  // unigram piece scores (SPM)
    int max_token_len_ = 0;
    bool byte_encode_ = true;
    bool add_bos_ = true;
    bool add_eos_ = false;
    bool is_spm_ = false;              // SentencePiece/unigram tokenizer
    bool add_space_prefix_ = true;     // SPM: prepend ▁ to the segment
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

    // Path F4b (#45): if a saved KV cache exists for the given
    // conversation_id and is compatible (matching model fingerprint,
    // same n_layers / kv_heads / head_dim / max_context / kv_type_bytes,
    // same KV quant mode), load it, find the longest common prefix
    // between the cached tokens and the new prompt, scatter that
    // matched portion into kv_cache_, and return the matched prefix
    // length so the caller can skip prefill for [0, hydrated_len).
    // Returns 0 (cold prefill) on any mismatch or missing file.
    int try_hydrate_kv(const std::vector<int>& prompt_tokens,
                       const std::string& conv_id);

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

    // Path F (#45): GGUF path + cheap fingerprint, set at load() time.
    // F3 writes the fingerprint into each KV cache file header; F4
    // will refuse to hydrate from a cache file whose hash doesn't
    // match the currently-loaded model.
    std::string   gguf_path_;
    char          model_fingerprint_[32] = {};

    void*         weights_ = nullptr;
    int64_t       weights_size_ = 0;
    void*         mapped_output_host_base_ = nullptr;
    size_t        mapped_output_host_bytes_ = 0;
    ModelWeights  model_weights_ = {};
    std::vector<void*> device_weight_copies_;
    Tokenizer     tokenizer_;

    int           last_token_ = 0;
    std::vector<int> recent_tokens_;

    // #86: grammar-constrained decoding state. grammar_active_ is set per
    // generate() when GenParams::grammar parses. grammar_token_bytes_ maps
    // token id -> literal output bytes; built once from the tokenizer (vocab
    // is fixed after load) and reused across requests.
    Grammar                  grammar_;
    GrammarState             grammar_state_;
    bool                     grammar_active_ = false;
    std::vector<std::string> grammar_token_bytes_;
    void prepare_grammar(const GenParams& params);

    // In-memory prefix cache: the token sequence whose K/V is currently
    // resident in kv_cache_ (the previous request's prompt + generated
    // tokens). Requests are serialized by the server mutex and nothing zeroes
    // the KV between turns, so a new request can skip prefill for the longest
    // common prefix with these tokens — reusing the shared system prompt
    // across unrelated requests with no disk I/O. Empty = cold (no reuse).
    std::vector<uint32_t> resident_tokens_;

    cudaStream_t    stream_ = nullptr;
    cudaGraph_t     decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    bool            graph_captured_ = false;
    float*          host_logits_ = nullptr;
    int             host_logits_capacity_ = 0;
    // Device-side int written by the host before each decode-graph
    // replay; read by rope_inplace_store_kv_fp16_dyn and
    // flash_attention_decode_dyn so a single captured graph works for
    // every step. Allocated on engine load, freed on unload.
    int*            d_pos_ = nullptr;

    void transformer_layer(int layer, int pos, half* x);
    // Gemma 4 decoder layer: per-layer head dim (256/512), dual RoPE, 4-norm
    // sandwich + layer_output_scale, GeGLU double-wide MLP, and Per-Layer
    // Embeddings (#93). See transformer_layer_gemma4 in decode.cu.
    void transformer_layer_gemma4(int layer, int pos, half* x);
    // Gemma 4 PLE: compute per-token per-layer-input [n_layers * ple_dim] from
    // the (scaled) token embedding + token id, into `out`. Set before the layer
    // loop; each gemma4 layer reads its slice via gemma_ple_input_.
    void compute_gemma_ple_input(const half* inputs_embeds, int token, half* out);
    half* gemma_ple_input_ = nullptr;
    // Gemma 4 normalizes V with a *weightless* per-head RMSNorm before the KV
    // store (ggml_rms_norm in llama.cpp). fused_rmsnorm_residual needs a weight
    // vector, so this is a device buffer of `head_dim` ones used as unit weight.
    float* gemma_vnorm_ones_ = nullptr;
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
    // Gemma 4 batched-prefill layer: same layer-major contract as
    // transformer_prefill, but with the Gemma decoder structure (per-layer
    // head dim, KV sharing, weightless V-norm, GeGLU, 4-norm sandwich, sliding
    // window, PLE, layer scale). `ple_input_batch` is [n_tokens × n_layers ×
    // ple_input_dim] — token t's layer-l slice at (t*n_layers + l)*ple_dim.
    void transformer_prefill_gemma4(int layer, int start_pos, int n_tokens,
                                    half* x_batch, const half* ple_input_batch);
    int  decode_step(int pos);
    // Replay-decode counterpart of decode_step that runs the captured
    // CUDA graph instead of launching ~450 kernels host-side. The graph
    // is built once after the first non-graph decode step.
    int  decode_step_graph(int pos);
    void build_cuda_graph(int pos);
    // Graph-capture-friendly variant of transformer_layer used while
    // recording decode_graph_. Same kernel sequence as transformer_layer,
    // but rope+KV-store and flash-attention read `*d_pos_` instead of
    // a captured int — so the graph stays valid across decode positions.
    void transformer_layer_graph(int layer, half* x);
    bool check_memory_and_thermal(int pos);
    // Has the JLLM_DECODE_GRAPH env var been set to a non-zero value?
    // Cached on first read. Default off.
    bool decode_graph_enabled() const;
    // Has the JLLM_BATCHED_PREFILL env var been set to a non-zero value?
    // Cached on first read.
    bool batched_prefill_enabled() const;
};

}  // namespace jllm
