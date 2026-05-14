# Changelog

## v0.1.0-alpha.3 — 2026-05-15

Path B: layer-major batched prefill. Validated on Jetson Orin Nano
Super 8 GB, Qwen3-4B Q4_K_M, 25 W MAXN SUPER.

| | main (alpha.2) | alpha.3 (Path B) | Δ |
|---|---|---|---|
| Prefill | 8.2 tok/s | **15.4 tok/s** | **+88% (1.88×)** |
| TTFT | 2200 ms | **1181 ms** | **−47%** |
| Decode | 7.5 tok/s | 7.5 tok/s | unchanged (out of scope) |
| Output | reference | bit-identical | ✓ |

### Added
- `gemm_quant_batched()` — K-quant GEMM that reads each weight value
  from DRAM once and re-uses it across `N` prompt tokens. Each warp
  owns one output row and holds `N` register accumulators.
- `flash_attention_prefill_batched()` — single launch processes all
  `N` query tokens against the (now-populated) K/V cache, with
  causal-mask `seq_len = start_pos + token + 1` per query.
- `Engine::transformer_prefill()` — layer-major prefill: batched
  RMSNorm → batched QKV → batched QK-norm → per-token RoPE+KV-store →
  batched attention → batched Wo + residual → batched FFN
  (RMSNorm + gate/up + SwiGLU) → batched W_down + residual.
- `Engine::transformer_layer_attn_compute()` and
  `transformer_layer_attn_block()` — split the post-QKV tail so
  prefill can hoist Wo and both residuals out of the per-token loop.
- `ScratchPool::mark()` / `rewind_to()` — save/restore allocation
  cursor, used by `transformer_prefill` to keep `x_batch` below the
  per-layer rewind point.
- `GenStats::ttft_ms` — time-to-first-token (prompt-submitted →
  first sampled token delivered).
- `JLLM_BATCHED_PREFILL` env-var. Default on; set to `0` to opt out.

### Changed
- `transformer_layer` refactored to delegate to the new
  attn-block / ffn-block helpers; per-token / decode behavior is
  bit-identical to alpha.2.
- `gemv_quant_batched_q4k/q5k/q6k_kernel`s ship at 56-64 registers,
  zero spill stores, validated against `gemv_quant` for byte equality.
- First-token sampling now reuses the last prefill hidden state
  instead of re-running a full forward pass (Path A, originally in
  alpha.2 but called out here because TTFT is the visible metric).

### Known limits
- Decode is still per-token. At 7.5 tok/s we're at ~13% of LPDDR5
  peak; reference llama.cpp on this hardware hits ~18 tok/s. Tracked
  as Path C in #19.

## v0.1.0-alpha.1 — 2026-05-13

Initial seeded release. The codebase is lifted verbatim from the
[ai-hpc/ai-hardware-engineer-roadmap / Projects / jetson-llm-runtime](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Projects/jetson-llm-runtime)
framework (31 files, ~4500 LOC). Status: code-complete, compiles cleanly,
needs hardware validation on Orin Nano.

### Added

- `src/memory/` — `MemoryBudget`, `OOMGuard`, `KVCachePool`, `ScratchPool`.
  Every allocation accounted for before inference starts so the runtime
  can coexist with `whisper-server` + Piper + Home Assistant on a
  7.6 GB iGPU.
- `src/jetson/` — Orin HAL: `PowerState` (nvpmodel 7–25 W), `ThermalState`
  (adaptive backoff), `JetsonInfo` (probed once at startup), `LiveStats`.
- `src/kernels/` — SM 8.7 CUDA kernels: `gemv_q4` (INT4 dequant-fused
  matrix-vector), flash `attention`, `fused_norm`, `rope`, `softmax`,
  `convert`, `vec_add`. Tuned for 48 KB shared memory and 16 SMs.
- `src/engine/` — Transformer forward pass: GGUF loader, tokenizer
  (encode/decode), top-k/top-p/temperature sampling, layer orchestration.
- `src/server/` — OpenAI-compatible HTTP: `/v1/chat/completions`, `/health`,
  `/v1/models`. Drop-in target for swapping out `llama-server` from
  GenieClaw.
- CMake build, with explicit aarch64-Jetson-only guard (rejects x86,
  discrete GPUs, Windows, macOS).
- `scripts/setup_jetson.sh`, `scripts/bench.sh`, `scripts/profile.sh`,
  `scripts/test_plan.sh`.
- Tests: `tests/test_kernels.cu`, `tests/test_memory.cpp`,
  `tests/test_model_load.cpp`.
- Documentation: architecture, build, engine, GGUF, jetson-HAL, kernels,
  memory, performance, server, testing.

### Known limits

- No on-device runtime validation yet (this is what alpha.1 → alpha.2
  exists to do).
- Internal symbols use `jllm`/`jllm_*` namespacing inherited from the
  seed framework. Renaming to a GenieClaw-aligned `geniert_*` namespace
  is deferred to v0.2 to keep the alpha churn-free.
- GenieClaw integration is read-only for now — both runtimes will run
  side-by-side starting in alpha.8 before any default flip.
