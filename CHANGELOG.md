# Changelog

## v0.1.0-alpha.9 — 2026-05-16

Path F: persistent KV cache for multi-turn conversations. Per-turn
state is saved to disk at turn end and hydrated at turn start; the
next turn's prefill skips the matched prefix between cached and new
prompt tokens. Acceptance criterion from #6 was ≥ 80 % reduction in
second-turn prompt-eval time; we hit 50 % on prefill (848 → 426 ms)
and 48 % on TTFT (857 → 444 ms) on a representative 2-turn
conversation where 24 / 36 cached tokens match the new prompt prefix.
Prefill / decode throughput per turn is unchanged from alpha.8 — the
gain is purely from skipping the matched prefix.

| | Turn 1 (cold) | Turn 2 (warm) | Δ |
|---|---|---|---|
| Prompt tokens | 33 | 39 | — |
| Hydrated from cache | — | **24 tokens** (8 ms load) | — |
| Prefill (work + wall) | 33 / 848 ms | 15 / 426 ms | **−50 %** |
| **TTFT** | **857 ms** | **444 ms** | **−48 %** |

### Path F series

| PR | Phase | What |
|---|---|---|
| #46 | F1 — serialization round-trip | Standalone test: 144 MB FP16 + 72 MB INT8 KV configs round-trip byte-identical; truncated files rejected via fstat-vs-header-body size check |
| #47 | F2 — conv-id surface | `--conv-id <id>` on CLI, `conversation_id` field in HTTP `/v1/chat/completions`. Engine logs, no behavior change. |
| #48 | F3 — save on turn end | Atomic `<path>.tmp` + fsync + rename. Wrote the full pool (144 MB / 5.6 s) — addressed in F3b |
| #49 | F3b — pack only used_tokens | Body now `n_layers × used_tokens × entry_bytes`; **34× smaller and faster** (165 ms / 5.4 MB at typical sizes) |
| #50 | F4a — persist token IDs | Format v2: tokens region between header and body. Off-by-one fix: previous code claimed KV[N+completion-1] held a real token; actually it held zero/stale |
| #51 | F4b — hydrate on turn start | `peek_kv_header` + `try_hydrate_kv`. Validates fingerprint and shape, longest-common-prefix match, scatters into pool, skips prefill for matched range. **This release.** |

### Added

- `src/persistence/kv_cache_file.{cpp,h}` — save/load/peek helpers.
- `KVCachePool::gather_used_to_host`, `scatter_from_host(zero_remaining=true)`, `packed_used_bytes` — per-layer gather/scatter between GPU pool and a host buffer of packed used-positions.
- `validate_conversation_id` — `[A-Za-z0-9_-]{1,64}` accepted; everything else dropped at the CLI/HTTP boundary so a path-injection attempt can't reach the engine.
- FNV-1a-64 model fingerprint over `(file size || first 256 B of GGUF)`. Stored in the header; refuses to hydrate a cache built against a different model.
- `JLLM_KV_CACHE_DIR` env var (default `/opt/jllm/data/kv-cache`).
- Engine adds `gguf_path_` and `model_fingerprint_[32]` members; `try_hydrate_kv(prompt_tokens, conv_id)` private method that returns the matched prefix length.

### File format v2

```
[header 128 B] [tokens used_tokens × 4 B] [KV body body_bytes]
```

Header bumped from v1 (F3). Body is per-layer packed: each layer holds `used_tokens × eb` bytes (keys then values). Old v1 files don't load (size check refuses them). Path F was alpha-only; no migration path needed.

### Skip rules at save time

| Condition | Action |
|---|---|
| `conv_id` empty / invalid | single-shot, no save |
| committed_tokens < 4 | trivial, skip eMMC write |
| Overflow CPU pool was used | fast-pool-only in v1; warn-log + skip |
| Any I/O error | warn-log + skip |

### Skip rules at hydrate time

| Condition | Action |
|---|---|
| Cache file missing or malformed | cold prefill |
| Model fingerprint mismatch | cold prefill (model changed since save) |
| Shape mismatch (layers / heads / head_dim / max_context / kv_bytes) | cold prefill |
| Longest common prefix = 0 | cold prefill (different conversation prefix) |
| Match ≥ 1 token | scatter the matched range, prefill the rest |

### Known cosmetic issue (not behavior)

After hydrate, the prefill log line over-reports tok/s:

```
[engine] Prefill: 39 tokens in 426 ms (91.6 tok/s)
```

The 39 is the prompt size; the 91.6 is `39 / 0.426`. Actual work was 15 prefill tokens (39 − 24 hydrated), giving the real rate of ~35 tok/s. Wall-clock and TTFT are correct. Will be cleaned up alongside F5 if it stays a nuisance.

### Not in this release

- **F5 — LRU eviction + size cap.** `JLLM_KV_CACHE_DIR` grows unbounded today; production deployments should externally enforce a cap, or use a per-day rotation. F5 is the next planned phase.
- **Overflow-pool serialization.** Cache today is fast-pool-only; conversations that spilled to the CPU overflow pool aren't saved.
- **Cache-aware prompt logging.** `Prefill: <prompt_tokens>` doesn't distinguish hydrated from prefilled tokens; the saved-tokens count separately would be clearer.

## v0.1.0-alpha.8 — 2026-05-16

Path E E5: multi-warp cooperative MMQ Q4_K prefill kernel. Replaces
alpha.7's single-warp `gemm_mmq_q4k_kernel` (E4 variant) with a
4-warp-per-CUDA-block cooperative version that shares one dequanted
16×32 FP16 A-tile across 4 contiguous N-stripes (8 tokens each).
Dequant cost amortized 4×. Validated on Jetson Orin Nano Super 8 GB,
same hardware/model/prompt as alpha.7.

| | scalar fallback | alpha.8 (Path E E5) | Δ vs scalar | Δ vs alpha.7 (E4) |
|---|---|---|---|---|
| Prefill | 16.45 tok/s (2006 ± 5 ms) | **38.68 tok/s (853 ± 3 ms)** | **+135 %** | **+37 %** |
| TTFT | 2014 ms | **~862 ms** | **−57 %** | −27 % |
| Decode | 10.0 tok/s | 10.0 tok/s | unchanged | unchanged |
| Output | reference | character-for-character identical | ✓ | ✓ |

Mean gap ≈ 1153 ms vs combined σ ≈ 6 ms → ~190σ separation.
**vs `llama-bench pp18 = 17.97 ± 0.65 tok/s`: genie-ai-runtime now
leads by +115 %.** Cumulative prefill since alpha.2: **+372 %**.

### Path E series (continued)

| PR | Phase | Status | What |
|---|---|---|---|
| #41 | E5 — multi-warp standalone | merged | 4 warps cooperate on one A-tile across 4 N-stripes; 39 regs/thread; ~407 GFLOPS aggregate |
| #42 | E5b — integration | **merged** | This release. +37 % vs alpha.7, +135 % vs scalar |

### Changed

- `gemm_mmq_q4k_kernel` body replaced with the multi-warp variant.
  Same name, same signature, same `JLLM_MMQ_Q4K` env flag — only the
  launch shape differs (`grid.x` divisor 8 → 32, block size 32 → 128).
- Dispatcher comment + stderr announcement updated to "multi-warp".

### Added (kernel internals)

- `MMQ_Q4K_N_WARPS = 4`, `MMQ_Q4K_BLOCK_N = 32` constants.
- Per-(row, sub-block) scale precompute now uses **128 threads × 1
  entry each** (vs alpha.7's 32 lanes × 4 entries).
- Cooperative dequant: 128 threads × 4 elements = 16 × 32 staging
  tile. Thread `t` covers `row = t/8`, cols `(t&7)*4 + [0..3]` — 8
  threads share each row's block_q4_K read (L1-cache friendly).
- Block-wide `__syncthreads()` replaces `__syncwarp()` at the
  precompute / dequant / MMA boundaries.

### Why the win held end-to-end

E3b (#37) regressed prefill 27 % despite a +5 % standalone GFLOPS gain
on Wo. E5b (#42) translates 1.66× standalone GFLOPS into 1.37× end-to-end
prefill (and +135 % vs scalar). The difference: the multi-warp kernel
is **uniform across shapes** (~405 GFLOPS on every Qwen3-4B prefill
GEMM in #41), so the integration mix doesn't get bottlenecked on a
single weak shape. With E3 we only tested Wo (M=2560) standalone and
got blindsided when the integrated workload was dominated by gate/up
(M=9728) and down (K=9728); E5 covers all six shapes uniformly.

### Not in this release

- **Q5_K / Q6_K MMQ.** Still scalar. Q5_K port is the obvious next
  step (likely a clean adaptation of E5's structure with a slightly
  different dequant). Q6_K remains blocked on the 210-byte alignment
  problem from #29 for any vectorized inner load.
- **Further occupancy / pipelining.** Current kernel is at 2.7 %
  tensor-core utilization, still well below peak. The remaining
  knobs (B-fragment shared-mem staging, async `cp.async` prefetch,
  compile-time N specialization) would each cost more engineering
  than the gain we'd see on Orin Nano Super's bandwidth budget.

## v0.1.0-alpha.7 — 2026-05-16

Path E: tensor-core MMQ Q4_K prefill GEMM. Replaces the scalar
`gemm_quant_batched_q4k_kernel`'s CUDA-core FMAs with
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on SM 8.7.
Validated on Jetson Orin Nano Super 8 GB, Qwen3-4B Q4_K_M, 25 W MAXN
SUPER, GPU locked 918 MHz, 5 samples each branch (same-day
re-baseline).

| | alpha.6 baseline | alpha.7 (Path E) | Δ |
|---|---|---|---|
| Prefill | 16.6 tok/s (1993 ± 2 ms) | **28.3 tok/s (1166 ± 2 ms)** | **+70.8 % (1.71×)** |
| TTFT | 2000 ms | **1179 ms** | **−41 %** |
| Decode | 10.0 tok/s | 10.0 tok/s | unchanged |
| Output | reference | sensibly-identical | ✓ |

Mean gap ≈ 827 ms vs combined σ ≈ 2 ms → ≫100σ separation.
**vs `llama-bench pp18 = 17.97 ± 0.65 tok/s`: genie-ai-runtime now
leads by +57 %**.

### Path E series

| PR | Phase | Status | What |
|---|---|---|---|
| #34 | E1 — smoke test | merged | Validate `mma.sync.m16n8k16` compiles + runs on SM 8.7 |
| #35 | E2 — single-tile skeleton | merged | Q4_K → FP16 shared-mem dequant + 1 MMA tile, M=N=16, K=256 |
| #36 | E3 — full GEMM standalone | merged | Arbitrary M/N/K; 74 GFLOPS on Wo; correctness PASS |
| #37 | E3b — integration attempt | **closed, negative result** | Naive kernel regressed prefill 27 %. 139 regs/thread, 32× redundant per-(row, sb) scale work across lanes. |
| #38 | E4 — kernel rework | merged | Per-(row, sb) scale precompute + drop dequant unroll. 96 regs, ≥ 210 GFLOPS on all six prefill shapes, ~245 aggregate. |
| #39 | E4b — integration | **merged** | This release. +70.8 % end-to-end prefill, output sensibly identical. |

### Added (Path E)

- `gemm_mmq_q4k_kernel` in `src/kernels/gemv_q4.cu` — one warp per
  CUDA block computes a 16 × 8 output tile via 16 m16n8k16 MMAs per
  Q4_K block. Shared memory holds (1024 B) the FP16 staging tile and
  (1024 B) the per-(row, sub-block) (d, dm) scale table.
- `JLLM_MMQ_Q4K` env var, default on. Set to `0` to opt back into the
  scalar `gemm_quant_batched_q4k_kernel` for A/B or in case a future
  model trips an MMA-specific assumption.
- Dispatcher hook in `gemm_quant_batched_gpu` routing Q4_K through the
  MMQ kernel when the flag is set. Falls back to the scalar kernel on
  `cudaError` (same defensive pattern as Path C).
- Stderr announcement once per process so an A/B run plainly shows
  which path is active.

### Why the win

1. **Compute-throughput.** Tensor-core FP16 peak on SM 8.7 is
   ~15 TFLOPS vs ~2 TFLOPS for CUDA-core FP16. The Q4_K prefill
   GEMM kernels were running at ~12 % CUDA-core utilization
   (nsys at alpha.5). Tensor cores unlocked a ~7.5× compute ceiling
   that we converted to ~2× kernel speedup in practice.
2. **Memory-traffic shape.** Activations land naturally as col-major
   B fragments for `.row.col` MMA, no shuffle needed. Q4_K weights
   reach the tensor cores via a once-per-sub-block shared-memory
   dequant tile (16 × 32 FP16 = 1024 B) — pays the dequant cost once
   per 256 multiply-accumulates.

### Not in this release

- **Q5_K / Q6_K MMQ.** Still on the scalar batched kernels. Q6_K
  blocks remain blocked on the 210-byte alignment problem from #29
  for any vectorized inner load. Future direction: Q5_K MMQ may be
  worth doing (different alignment); Q6_K likely needs a separate
  dequant strategy.
- **Multi-warp MMQ blocks (E5).** Current kernel is one warp per
  CUDA block at ~33 % SM occupancy. A multi-warp cooperative
  variant (N warps share one dequanted A-tile across multiple
  N-stripes) could close more of the gap to tensor-core peak. Will
  do if/when the workload demands it.
- **Tuning above SM 8.7.** Path E targets Orin Nano Super; no
  testing on Orin NX, AGX, or any non-Jetson SM 8.6/8.9 part yet.

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
