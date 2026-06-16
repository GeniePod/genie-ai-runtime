# Gemma 4 E2B support — implementation plan

Goal: run **google/gemma-4-E2B-it** (GGUF) correctly on the Orin Nano, including
Per-Layer Embeddings (PLE) — the feature that even llama.cpp does not yet
implement in its forward graph ([ggml-org/llama.cpp#22243](https://github.com/ggml-org/llama.cpp/issues/22243)).
Once it generates correct, coherent output here, port the validated math to
TensorRT-Edge-LLM (issue [#72](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/72)).

This is a large, cross-cutting change: the engine today is hardcoded to the
Llama/Qwen pre-norm + SwiGLU + single-head-dim + single-RoPE path. Gemma 4
diverges on almost every axis. Work lands in independently buildable, Jetson-
tested slices; nothing about this can be compiled or validated on an x86 dev box
(CUDA-only runtime), so every slice is verified on the device.

## Architecture (from HF `modeling_gemma4.py` + llama.cpp `gemma4`)

`gemma4_text`, 35 layers, hidden 1536, vocab 262144, tied embeddings, bf16.

| Aspect | Value | genie today |
| --- | --- | --- |
| Attention pattern | 4 sliding : 1 full (`sliding_window_pattern=6`), last layer full | single causal path, no per-layer type |
| Head dim | sliding **256** / full **512** (per attention type) | one `head_dim` for all layers |
| KV heads | 1 (MQA) | GQA supported |
| RoPE | sliding θ=1e4 full-rotary; full θ=1e6 partial-rotary 0.25 ("proportional") | one `rope_theta`, full rotary |
| Sliding window | 512 | no window mask |
| KV sharing | last `num_kv_shared_layers=20` reuse a prior layer's K/V | none |
| MLP | GeGLU (`gelu_pytorch_tanh`), double-wide on KV-shared layers | SiLU SwiGLU, single width |
| Norms / layer | 4 (input, post-attn, pre-ffn, post-ffn) + PLE post-norm + final | 2 (pre-attn, pre-ffn) |
| RMSNorm | `x * w` (HF bakes no +1; confirm GGUF matches) | `x * w` ✓ |
| Embedding | scaled by √hidden = √1536 | no scaling |
| Final logits | softcap 30: `30*tanh(logits/30)` | none |
| Tokenizer | SentencePiece / unigram (scores), 262144 | BPE merges only |
| **PLE** | per-layer 256-d embedding lookup + projection + gate + norm, injected each layer | none |

### PLE (the hard part)

Per layer `i`, after the FFN residual:

```
g  = gelu_tanh( per_layer_input_gate[i] @ h )        # [256]
p  = per_layer_projection[i] @ (g * ple_input[i])    # [1536]
h  = h + post_per_layer_input_norm[i]( p )
```

`ple_input` is `(token_identity + context_projection) * 2^-0.5`, where
`token_identity = embed_tokens_per_layer[token]` (table `262144 × 35 × 256`,
scaled √256) and `context_projection = RMSNorm( (model_proj @ inputs_embeds) *
hidden^-0.5 )`. The per-layer-embedding table is the bulk of the on-disk weight
(~2.3 B params) and is why "E2B" = *effective* 2 B.

### OPEN QUESTION — AltUp / LAuReL?

llama.cpp registers `altup_*` / `laurel_*` tensors under arch `gemma4`
(`llama-arch.cpp`, comment "gemma 3n"). HF's `Gemma4TextDecoderLayer` uses
**only** PLE — no AltUp/LAuReL. We must confirm from the **actual GGUF tensor
list** whether E2B carries `blk.*.altup_*` / `blk.*.laurel_*`. If present, the
forward pass is materially larger (AltUp = K-way alternating residual streams;
LAuReL = learned augmented residual). **Resolve before Phase B.**

## GGUF metadata keys (llama.cpp `gemma4.*`)

`block_count`, `embedding_length`, `feed_forward_length`, `attention.head_count`(=8),
`attention.head_count_kv`(=1), `attention.key_length`/`value_length` (full head dim),
`attention.key_length_swa`/`value_length_swa` (sliding head dim),
`attention.sliding_window`(512), `attention.sliding_window_pattern`(6),
`attention.shared_kv_layers`(20), `embedding_length_per_layer_input`(256),
`rope.freq_base`(1e6), `rope.freq_base_swa`(1e4), `final_logit_softcapping`(30).
> Confirm the `key_length` vs `key_length_swa` → full/sliding mapping from the dump.
> Note: `model.cpp` caps the metadata loop at 256 entries and matches keys with
> `strstr` (substring) — the per-layer-input and `_swa` keys must be matched
> *before* their shorter prefixes (`embedding_length`, `key_length`).

## Phases (each builds + is tested on the Jetson)

- **A — Recognition & host pieces** (no CUDA math):
  - A1: parse `gemma4` arch + hparams into `ModelConfig`; **fail loud** in
    `Engine::load` until the forward pass exists (no silent mis-run as Llama).
  - A2: SentencePiece/unigram tokenizer path (host C++; **unit-testable on x86**
    against HF reference token IDs — the one slice verifiable off-device).
- **B — Dense Gemma 4 forward (no PLE yet)**: GeGLU kernel, embedding √-scale,
  4-norm layer wiring, per-attention-type head dim (256/512) in attention + KV
  cache, dual RoPE (θ + partial rotary) selected per layer, sliding-window mask,
  KV-sharing, final-logit softcap. Output will be llama.cpp-level (subtly off).
- **C — PLE**: per-layer embedding lookup + projection + gate + norm injection;
  the per-layer-embedding table load + memory accounting. This closes the gap to
  correct E2B. (If the dump shows AltUp/LAuReL, add a Phase C′ for those.)
- **D — Validation**: greedy-decode logit/argmax parity vs HF `transformers` on
  a fixed prompt set; coherence, tok/s, and KV/PLE memory inside the 8 GB budget.

## Verification strategy

- Host (x86): SPM tokenizer token-ID parity vs HF; pure-C++ reference of each new
  math kernel where feasible (genie already keeps CPU fallbacks for norms).
- Device (Jetson): per-phase generation smoke; Phase D logit parity vs HF.
- Reference HF impl cached at `/tmp/modeling_gemma4.py` during bring-up.
