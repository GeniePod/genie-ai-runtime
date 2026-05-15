# Changelog

## v0.1.0-alpha.6 — 2026-05-15

Path D: right-size the prefill GEMM token-unroll constant so the
`#pragma unroll`-ed inner loop stops burning issue slots on
predicated-off iterations at the prompt lengths we actually serve.
Validated on Jetson Orin Nano Super 8 GB, Qwen3-4B Q4_K_M, 25 W MAXN
SUPER, GPU locked 918 MHz, 5 samples each branch (same-day re-baseline).

| | alpha.5 baseline | alpha.6 (Path D) | Δ |
|---|---|---|---|
| Prefill | 14.64 ± 0.07 tok/s | **15.68 ± 0.05 tok/s** | **+7.1% (1.07×)** |
| Prefill wall (33 tok) | 2253.8 ± 10.9 ms | **2104.0 ± 6.7 ms** | **−149.8 ms** |
| Decode | unchanged | unchanged | — |
| Output | reference | bit-identical | ✓ |

Mean gap ≈ 150 ms vs combined σ ≈ 13 ms → ~12σ separation, not noise.
vs `llama-bench pp18 = 17.97 ± 0.65 tok/s` Path D closes ~1/3 of the gap
from a one-line constexpr change.

### Changed (Path D — PR #31)

- `GEMM_MAX_BATCH` constant in `src/kernels/gemv_q4.cu` lowered from
  32 to 20. The Q4_K, Q5_K, and Q6_K batched prefill GEMM kernels all
  use the same per-thread `acc[GEMM_MAX_BATCH]` register array and the
  same `#pragma unroll for (t = 0; t < GEMM_MAX_BATCH; t++) if (t < N)`
  inner loop, so all three benefit. The existing host dispatcher
  chunks `N > GEMM_MAX_BATCH` into multiple kernel launches; for the
  typical Qwen3 chat-wrapped single-turn prompt the chunker now runs
  20+13 instead of 32+1, lifting second-launch issue-slot utilization
  from 3 % to 65 %.

### Why not lower than 20?

20 is sized just above the typical N=18–20 we see after Qwen3
chat-template wrapping of a single user turn. Going below 20 starts
chunking single-turn prompts into 3+ launches; going to 24 or 32
re-introduces predicated-off waste. 20 was the cheap local optimum;
templated-on-N specialization is the next direction if/when this
becomes a bottleneck again.

### Not in this release

- Tensor-core MMQ kernel (the actual path to llama.cpp parity at 18+
  tok/s prefill). Multi-week rewrite, tracked separately.
- Q6_K uint32 weight loads — still blocked on the 210-byte block
  alignment problem from PR #29.

## v0.1.0-alpha.5 — 2026-05-15

Path C: vectorized weight loads for the Q4_K decode GEMV family, plus
the decode-side memory & fusion work that landed directly on main
between alpha.3 and Path C. Validated on Jetson Orin Nano Super 8 GB,
Qwen3-4B Q4_K_M, 25 W MAXN SUPER.

| | alpha.3 | alpha.5 | Δ |
|---|---|---|---|
| Decode | 7.5 tok/s | **9.1 tok/s** | **+21% (1.21×)** |
| Prefill | 15.4 tok/s | 15.2 tok/s | unchanged (noise) |
| TTFT | 1181 ms | ~1180 ms | unchanged |
| Output | reference | bit-identical | ✓ |

### Added (Path C — Q4_K uint32 weight loads, PRs #25 / #26 / #27 / #28)

- `dot_q4k_row_uint32` — per-row Q4_K dot helper. Each lane reads
  four packed q-bytes as one `uint32_t` from `blk.qs + 32*il + sub_base`,
  eliminating the byte-by-byte inner loop. Same arithmetic, same FMAs,
  fewer issued load instructions and constant-per-lane scales.
- `gemv_quant_add_uint32_q4k_kernel` — residual-fused Q4_K single-output
  kernel (decode Wo, ~20% of decode wall-clock).
- `gemv_quant_pair_uint32_q4k_kernel` — Q4_K + Q4_K dual-output kernel
  (decode gate/up, ~44% of decode wall-clock — the biggest single share).
- `gemv_quant_triple_uint32_q4k_kernel` — pure Q4_K triple.
- `gemv_quant_triple_uint32_q4k_q4k_q6k_kernel` — Qwen3-4B QKV triple
  (Wq, Wk Q4_K; Wv Q6_K). Q4_K rows take the uint32 path; Q6_K rows
  fall through to the byte path inside the same kernel.
- `JLLM_Q4K_UINT32_LOADS` env var, default on. Dispatchers in
  `gemv_quant_add_gpu` / `gemv_quant_pair_gpu` / `gemv_quant_triple_gpu`
  route through the uint32 kernels when the flag is set and fall back
  to the existing typed byte-load kernels on any `cudaError`.

### Added (decode-path work on main between alpha.3 and Path C)

- Device-resident weight arena (`Pack device-resident layer weights`,
  `Allocate runtime pools before device weights`, and friends) —
  weights are now copied into a single device allocation per layer
  group instead of streaming from mmap'd host memory through pinned
  mappings. Trades load-time memory pressure for steady-state
  bandwidth determinism on Tegra.
- Decode GEMV residual-add fusion — the residual `vec_add` is folded
  into the typed gemv (`gemv_quant_add_typed_kernel<Type>`), removing
  one launch per layer per token.
- `gemv_rows_per_block` default lowered from 8 to 4 — more grid
  parallelism on the small-M shapes that show up in decode.
- `Map tied output when device copy cannot fit` / `Map tied output
  after runtime buffers` — graceful fallback for the
  tied-token-embedding output projection when the device arena would
  overflow the budget.

### Changed

- Decode-path `gemv_quant_add_typed_kernel<12>` is no longer the hot
  kernel by default — `gemv_quant_add_uint32_q4k_kernel` is, because
  `JLLM_Q4K_UINT32_LOADS` is now on by default. Set the env var to `0`
  to opt back into the byte path (same grammar as
  `JLLM_BATCHED_PREFILL` after alpha.3).

### Closed-without-merge (negative-result PRs, documented inline)

- #23 — Decode CUDA Graphs capture. Build correct, output
  bit-identical, but no speedup: kernel launches were already
  overlapping with GPU execution on the LPDDR5-bound workload, so
  eliminating host-side launch overhead bought ≤0%. Closed with the
  diagnosis attached.
- #24 — Split-K Q4_K GEMV. Same idea didn't help: SPLIT_K=4 grew
  block size 4× and register pressure dropped blocks-per-SM 4×, so
  total warps-per-SM was unchanged. The bottleneck is arithmetic
  intensity per byte loaded, not per-warp MSHR depth.
- #29 — Q6_K uint32 weight loads. Crashed with
  `cudaErrorMisalignedAddress`: `block_q6_K` is 210 B (not divisible
  by 4), so consecutive blocks alternate between 4- and 2-aligned
  memory positions and half the `uint32_t` loads fault. Format
  constraint, not fixable inside the kernel. Closed with the
  diagnosis.

### Known limits

- Q6_K decode kernels (W_down ~8% of decode, output logits ~7%) are
  still on the byte-by-byte path. Q6_K already runs at ~30-40% of
  LPDDR5 peak vs Q4_K's old ~10%, so the headroom is smaller, and
  the 210-byte block size rules out the obvious uint32 vectorization.
  Tracked as a future Q6_K uint16 attempt if it ever rises to the top
  of a profile.

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
