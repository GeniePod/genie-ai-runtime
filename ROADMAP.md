# Roadmap — what we did and how

This is the long-form history. For the user-facing intro and quickstart
see [`README.md`](README.md); for per-release notes see
[`CHANGELOG.md`](CHANGELOG.md). This file is the **path-by-path narrative**
— every numbered Path we ran, what it tried, what worked, what didn't,
and the verified perf delta on Jetson Orin Nano Super 8 GB.

Target hardware throughout: **Jetson Orin Nano Super 8 GB, SM 8.7,
L4T R36.4, CUDA 12.6, Qwen3-4B-Q4_K_M.gguf, 25 W MAXN SUPER**.

## Status: v1.0.0 shipped

Cumulative result of the alpha track, vs the alpha.2 first-tokens baseline:

| Metric                                | alpha.2 | v1.0.0           | Δ |
| ------------------------------------- | ------- | ---------------- | --- |
| Prefill (33-tok cold)                 | 8.2 tok/s   | **38.0 tok/s**   | **+363 %** |
| Decode                                | 7.5 tok/s   | **9.9 tok/s**    | **+32 %** |
| Cold TTFT                             | 2200 ms     | **877 ms**       | **−60 %** |
| Warm-turn TTFT (Path F, 67 % prefix)  | n/a         | **444 ms**       | new |
| KV pool memory @ 1024 ctx             | n/a         | **74 MB**        | new (−49 % vs alpha.11 FP16) |
| Model load — warm                     | —           | **1.3 s**        | new (cold is 30 s, NVMe-bound) |

vs `llama-bench pp18 = 17.97 ± 0.65 tok/s` on the same hardware +
model: **+115 % prefill**.

Output stays sensibly-identical to scalar/FP16 reference across the
entire path lineage (Path E reorders float adds via `mma.sync`; INT8
KV's per-(layer, pos, kv_head) absmax-scale loses ~0.79 % per element;
both produce character-identical text on the reference prompt).

## How we got here, path by path

The work landed as a sequence of numbered Paths. Each was an umbrella
GitHub issue with per-phase PRs. Verified-result tables hang off each PR.

### Path A — first coherent tokens (alpha.2)

The bring-up problem. Get a real GGUF model loading on Jetson and
producing text that isn't garbage. Solved Qwen3's per-head Q/K RMSNorm
and NeoX RoPE, tied output embeddings, BPE merges, special tokens,
no-think chat template by default. Closed [#1](https://github.com/GeniePod/genie-ai-runtime/issues/1)
(first coherent tokens) and [#8](https://github.com/GeniePod/genie-ai-runtime/issues/8)
(Jetson power reporting).

### Path B — batched prefill (alpha.3 → alpha.4)

K-quant GEMM, batched RMSNorm + QK-norm + SwiGLU, chunked-prefill
attention. Reads each weight value from DRAM once per layer and re-uses
it across all N prompt tokens — eliminates the per-token weight
re-fetch that was the dominant prefill cost. **Prefill 8.2 → 15.4 tok/s
(+88 %).** PRs [#13](https://github.com/GeniePod/genie-ai-runtime/pull/13) → [#17](https://github.com/GeniePod/genie-ai-runtime/pull/17); default flip [#18](https://github.com/GeniePod/genie-ai-runtime/pull/18).

### Path C — decode K-quant GEMV uint32 loads (alpha.5)

Each lane reads four packed q-bytes as one `uint32_t` from `blk.qs`.
Eliminates the byte-by-byte inner loop on the hot decode kernels (Wo,
gate/up pair, QKV triple) and folds the residual add into the gemv
itself. Q6_K rows stay on the byte path (their 128-byte block layout
doesn't satisfy 4-byte alignment). **Decode 7.5 → 9.1 tok/s (+21 %).**
PRs [#25](https://github.com/GeniePod/genie-ai-runtime/pull/25) (Wo) →
[#26](https://github.com/GeniePod/genie-ai-runtime/pull/26) (gate/up) →
[#27](https://github.com/GeniePod/genie-ai-runtime/pull/27) (QKV triple) →
default flip [#28](https://github.com/GeniePod/genie-ai-runtime/pull/28).

Negative result: split-K GEMV variant was slower than the residual-fused
single-kernel ([#23](https://github.com/GeniePod/genie-ai-runtime/issues/23),
[#24](https://github.com/GeniePod/genie-ai-runtime/issues/24)).

### Path D — right-sized prefill unroll (alpha.6)

Drops `GEMM_MAX_BATCH` from 32 to 20 so the `#pragma unroll`-ed token
loop stops burning issue slots on predicated-off iterations. At the
typical chat-wrapped N ≈ 33 the host dispatcher chunks 32 + 1 → 20 + 13,
raising second-launch utilization from 3 % to 65 %. **+7.1 % prefill,
byte-identical output.** PR [#31](https://github.com/GeniePod/genie-ai-runtime/pull/31).

### Path E — tensor-core MMQ Q4_K prefill GEMM (alpha.7 → alpha.8)

The big prefill win. Replaces the scalar Q4_K batched kernel's
CUDA-core FMAs with `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
on SM 8.7. A 16 × 32 FP16 staging tile is dequantized from one Q4_K
block into shared memory; 16 × 8 = 128 (d, dm) scale pairs precomputed
once per block. **4 warps per CUDA block share one dequanted A-tile
across 4 contiguous N-stripes** (32 tokens at a time, vs 8 for the
single-warp variant) — dequant cost amortized 4×. 39 regs/thread,
~407 GFLOPS aggregate on Qwen3-4B prefill shapes. Output sensibly-
identical (FP16-ULP-bounded float-add drift only). **Prefill 15.68 →
38.68 tok/s (+147 %).**

PRs [#34](https://github.com/GeniePod/genie-ai-runtime/pull/34) (smoke
test) → [#35](https://github.com/GeniePod/genie-ai-runtime/pull/35)
(skeleton) → [#36](https://github.com/GeniePod/genie-ai-runtime/pull/36)
(GEMM kernel) → [#38](https://github.com/GeniePod/genie-ai-runtime/pull/38)
(precomputed scales) → [#39](https://github.com/GeniePod/genie-ai-runtime/pull/39)
(integrate behind flag, alpha.7) → [#41](https://github.com/GeniePod/genie-ai-runtime/pull/41)
(multi-warp dequant) → [#42](https://github.com/GeniePod/genie-ai-runtime/pull/42)
(default flip, alpha.8). Plan + audit: [#33](https://github.com/GeniePod/genie-ai-runtime/issues/33) (closed completed).

Long-prompt scaling validated across kernel N = 33 / 88 / 235: ~2.36×
speedup over the scalar fallback is uniform across a 7× range of
prompt sizes (E5 stays at ~26 ms / prefill-token, scalar at ~62 ms).
See [`docs/performance.md`](docs/performance.md) for the detailed
table.

### Path F — persistent KV cache (alpha.9 → alpha.10)

Per-conversation KV state saved to disk at turn end and hydrated at
turn start. File format is per-layer-packed (only `used_tokens`
positions per layer, not the full pool) with the prompt token IDs
inline, so a follow-up turn finds the longest common prefix vs its new
prompt and skips prefill for the matched range. `--conv-id <id>` on
the CLI, or `conversation_id` field on the HTTP server's
`/v1/chat/completions`. Atomic save via `<path>.tmp` + `fsync` + `rename`;
truncated files rejected at load by a header-vs-fstat-size check;
FNV-1a-64 model fingerprint refuses caches built against a different
GGUF. F5 adds a per-process cache budget (default 1 GB) with
oldest-by-mtime eviction and stale `*.tmp` cleanup. **Warm-turn TTFT
857 → 444 ms (−48 %)** in a 2-turn conversation where the second
prompt shares 24/36 = 67 % of its tokens with the first.

PRs [#46](https://github.com/GeniePod/genie-ai-runtime/pull/46) (serialization
roundtrip) → [#47](https://github.com/GeniePod/genie-ai-runtime/pull/47)
(conv-id surface) → [#48](https://github.com/GeniePod/genie-ai-runtime/pull/48)
(save on turn end) → [#49](https://github.com/GeniePod/genie-ai-runtime/pull/49)
(pack only used tokens, 34× save speedup) → [#50](https://github.com/GeniePod/genie-ai-runtime/pull/50)
(format v2 — persist token IDs) → [#51](https://github.com/GeniePod/genie-ai-runtime/pull/51)
(hydrate on turn start, alpha.9) → [#53](https://github.com/GeniePod/genie-ai-runtime/pull/53)
(LRU eviction + stale-tmp cleanup, alpha.10). Plan: [#45](https://github.com/GeniePod/genie-ai-runtime/issues/45).

### Path G — decode throughput optimization (paused)

The decode-side equivalent of Path E. **G1** (Q6_K scale hoist) and
**G2-lite** measured neutral or slightly negative — the compiler was
already hoisting the scale loads implicitly. Closed both as no-ops.
Path G stays open at [#58](https://github.com/GeniePod/genie-ai-runtime/issues/58)
for a future attempt with a different attack (likely Q6_K block-layout
redesign for uint16 vectorization). Negative result kept for posterity
under [#19](https://github.com/GeniePod/genie-ai-runtime/issues/19).

### Path H — speculative decoding (deferred)

Feasibility-audited. Conclusion: with Qwen3-4B as the target model and
no smaller good draft model in the Qwen3 family, the win is bounded by
the draft model's quality on home-agent traffic. Deferred until either
a good draft candidate appears or we have a fine-tuned domain model
(per [`genie-ai-model`](https://github.com/GeniePod/genie-ai-model)).
Umbrella issue closed as deferred.

### Path I — INT8 KV cache (alpha.11 → alpha.12, default since alpha.12)

Per-(layer, pos, kv_head) absmax-scaled INT8 storage for the KV pool.
Halves the KV body bytes (144 MB → 72 MB at Qwen3-4B 1024 ctx) plus a
small ~2.25 MB per-head scales region. Output sensibly-identical to
FP16 at temp=0 on the reference prompt; word-level drift bounded by
INT8 precision floor (~0.79 % per-element). Throughput is unchanged at
chat-typical contexts; **the win is memory headroom** — 70 MB freed,
enough to push the working KV from 1024 to ~2000 tokens at the same
pool size, or to leave headroom for concurrent voice/HA traffic.

PRs [#63](https://github.com/GeniePod/genie-ai-runtime/pull/63) (per-head
scale storage) → [#64](https://github.com/GeniePod/genie-ai-runtime/pull/64)
(fp16_to_int8 wiring + kernel bug fixes) → [#65](https://github.com/GeniePod/genie-ai-runtime/pull/65)
(per-position scales through attention; remove broken N-sequential
fallback) → [#68](https://github.com/GeniePod/genie-ai-runtime/pull/68)
(default flip + Path F interop guard, alpha.12). Plan + audit:
[#62](https://github.com/GeniePod/genie-ai-runtime/issues/62).

**alpha.11 interop carve-out** — Path F's on-disk format v2 doesn't
carry the per-(layer, pos, kv_head) scales, so under `--int8-kv` the
server SKIPS save with a warning. Closing that gap is the v3 format
work tracked at [#67](https://github.com/GeniePod/genie-ai-runtime/issues/67).

### Production hardening (v1.0 cycle)

Server rewrite ([#5](https://github.com/GeniePod/genie-ai-runtime/issues/5),
closed): dropped the 246-line raw-sockets + hand-rolled-JSON server,
rewrote on cpp-httplib + nlohmann/json (the same building blocks
llama.cpp uses), full OpenAI-shape parsing, SSE streaming with
client-disconnect-cancels-generation, thread-per-request mutex-guarded
engine, opt-in build, systemd unit + installer. Phases tracked at
PRs [#73](https://github.com/GeniePod/genie-ai-runtime/pull/73),
[#74](https://github.com/GeniePod/genie-ai-runtime/pull/74),
[#75](https://github.com/GeniePod/genie-ai-runtime/pull/75).

Qwen3 reasoning support ([#76](https://github.com/GeniePod/genie-ai-runtime/issues/76),
closed): server-side `ThinkSplit` state machine extracts
`<think>...</think>` reasoning from the rest of the answer and
surfaces it under DeepSeek-style `reasoning_content` (separate
non-streaming field; separate `delta.reasoning_content` SSE chunks).
PR [#77](https://github.com/GeniePod/genie-ai-runtime/pull/77).

CLI observability: per-run `[engine] Model loaded in X ms (Y MB/s)`
timing + `scripts/bench_load.sh` cold/warm bench (cold 30 s NVMe-bound,
warm 1.3 s pagecache hit, 22.8× delta).

## What's next

Forward-looking work, in priority order:

| # | What | Why |
| -- | ---- | --- |
| [#4](https://github.com/GeniePod/genie-ai-runtime/issues/4)  | 100-iter × 1000-token stability soak | Harness shipped (PR [#70](https://github.com/GeniePod/genie-ai-runtime/pull/70)); needs the actual run. Gates the v1.0 → v1.x soak claims. |
| [#7](https://github.com/GeniePod/genie-ai-runtime/issues/7)  | 24-hour soak + packaging + genie-claw default flip | The final genie-claw-replaces-llama-server gate. |
| [#67](https://github.com/GeniePod/genie-ai-runtime/issues/67) | Path F format v3 — persist per-(layer, pos, kv_head) INT8 scales | Closes the INT8-KV + persistent-KV carve-out (server can drop FP16 default and unify with CLI). |
| [#58](https://github.com/GeniePod/genie-ai-runtime/issues/58) | Path G v2 — decode throughput +50–100 % | Different attack from the closed G1/G2-lite — likely Q6_K block-layout redesign. |
| [#56](https://github.com/GeniePod/genie-ai-runtime/issues/56) | Cold-turn TTFT reduction beyond Path E + F | Streaming pre-allocation, deeper kernel pipelining. |
| —  | LoRA-tuned GeniePod model integration | Lives in [`genie-ai-model`](https://github.com/GeniePod/genie-ai-model). Shortens runtime prompt 5–10× → TTFT 5–11 s instead of 40–80 s on long-context home-agent requests. |

## Non-goals

- **Multi-model serving** — single Jetson, single model. Dropped during issue #5 rescope.
- **Cross-platform** — x86, discrete GPUs, Windows, macOS not supported. Build refuses to configure off aarch64.
- **Pretraining / training** — inference engine only. Training lives off-Jetson; fine-tuning lives in [`genie-ai-model`](https://github.com/GeniePod/genie-ai-model).
- **Auth / TLS** — LAN-only deployment for v1.x. Revisit when the genie-claw integration story needs it.
- **Generic OpenAI-compat clients beyond chat completions** — embeddings, infill, function-call deltas, tool routing all out of scope unless genie-claw needs them.

## Pattern that worked

Three patterns worth keeping for whoever picks this up next:

1. **Path-based umbrella issues.** Each numbered Path got its own issue
   with a phase table, risks called out up front, and per-phase PRs that
   each posted a verified-result comment on Jetson before merge. Made
   negative results easy to absorb without losing momentum (Path G's
   neutral results, Path C's split-K dead end).

2. **Sensibly-identical, not byte-identical.** Once Path E's `mma.sync`
   reordered float adds we accepted FP16-ULP-bounded drift as the
   quality bar instead of byte equality. Same for INT8 KV. Without
   that we'd have been stuck on scalar FMAs forever.

3. **Honest perf re-baselines.** Each release (alpha.7 → alpha.8 → alpha.12)
   re-measured the prior-release number same-day with the same prompt
   and machine state, instead of trusting cross-day numbers. The
   alpha.11 entry in CHANGELOG specifically called out "we measured
   noise, not improvement" twice and rolled back claims.
