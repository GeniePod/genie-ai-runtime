# genie-ai-runtime

Jetson Orin-tuned LLM inference runtime — memory-first, power-aware,
with pre-allocated KV/scratch pools. Built to serve [`GenieClaw`](https://github.com/GeniePod/genie-claw)
on a 7.6 GB iGPU without crowding out whisper.cpp + Piper + Home Assistant.

**Target hardware:** Jetson Orin Nano Super 8 GB (SM 8.7, 102 GB/s, 67 TOPS GPU)
**Not supported:** x86, discrete GPUs, Windows, macOS — Jetson only.

## Status

`v0.1.0-alpha.12` — Path I (proper INT8 KV cache) shipped and made the
default. Saves ~50 % of KV pool memory (e.g. 144 MB → 74 MB for
Qwen3-4B at 1024 ctx) with quality validated against FP16 on a
representative prompt. Throughput is unchanged within noise; the win
is memory headroom for longer contexts and concurrent voice/HA on
Jetson Orin Nano Super 8 GB.

Use `--fp16-kv` to opt back into full-precision KV (no semantic
difference today, but useful if you observe quality issues on a
particular workload).

Current validated path:
- Coherent Qwen3 instruct output with automatic chat template and no-think mode.
- GGUF tokenizer loads Qwen BPE merges and special tokens.
- Qwen3 architecture fixes: 128-dim attention heads, Q/K RMSNorm, NeoX RoPE,
  tied output embeddings.
- **Batched prefill (Path B)**: K-quant GEMM, batched RMSNorm + QK-norm +
  SwiGLU, chunked-prefill attention. Reads each weight value from DRAM
  once per layer and re-uses it across all N prompt tokens.
- **Decode K-quant GEMV uint32 path (Path C)**: each lane reads four
  packed q-bytes as one `uint32_t` from `blk.qs`. Eliminates the
  byte-by-byte inner loop on the hot decode kernels (Wo, gate/up pair,
  QKV triple). Folds the residual add into the gemv itself. Q6_K still
  on the byte path (block layout doesn't satisfy 4-byte alignment).
- **Right-sized prefill unroll (Path D)**: drops `GEMM_MAX_BATCH` from
  32 to 20 so the `#pragma unroll`-ed token loop stops burning issue
  slots on predicated-off iterations. At the typical chat-wrapped
  N≈33 the host dispatcher chunks 32+1 → 20+13, raising second-launch
  utilization from 3 % to 65 %. +7.1 % prefill, byte-identical output.
- **Tensor-core MMQ Q4_K prefill GEMM (Path E, multi-warp)**: replaces
  the scalar Q4_K batched kernel's CUDA-core FMAs with
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on SM 8.7. A
  16×32 FP16 staging tile is dequantized from one Q4_K block into
  shared memory; 16 × 8 = 128 (d, dm) scale pairs precomputed once
  per block. **4 warps per CUDA block share one dequanted A-tile
  across 4 contiguous N-stripes (32 tokens at a time, vs 8 for
  alpha.7's single-warp variant)** — dequant cost amortized 4×.
  39 regs/thread, ~407 GFLOPS aggregate on Qwen3-4B prefill shapes,
  output text sensibly identical (FP16-ULP-bounded float-add drift
  only).
- **Persistent KV cache (Path F)**: per-conversation KV state saved to
  disk at turn end and hydrated at turn start. File format is
  per-layer-packed (only `used_tokens` positions per layer, not the
  full pool) with the prompt token IDs inline, so a follow-up turn
  finds the longest common prefix vs its new prompt and skips prefill
  for the matched range. `--conv-id <id>` on the CLI, or
  `conversation_id` field on the HTTP server's
  `/v1/chat/completions`. Atomic save via `<path>.tmp` + `fsync` +
  `rename`; truncated files rejected at load by a header-vs-fstat-size
  check. FNV-1a-64 model fingerprint refuses caches built against a
  different GGUF. **F5 (alpha.10)**: per-process cache budget (default
  1 GB) with oldest-by-mtime eviction; stale `*.tmp` files (default
  > 60 s) cleaned at next save.
- **INT8 KV cache (Path I, default since alpha.12)**: per-(layer, pos,
  kv_head) absmax-scaled INT8 storage for the KV pool. Halves the KV
  body bytes (144 MB → 72 MB at Qwen3-4B 1024 ctx) plus a small
  ~2.25 MB per-head scales region. Output sensibly-identical to FP16
  at temp=0 on the reference prompt; word-level drift bounded by INT8
  precision floor (~0.79 % per-element). Throughput is unchanged at
  chat-typical contexts; the win compounds at long context where
  attention's KV-read bandwidth share grows. `--fp16-kv` opts back
  into full-precision KV. **Note:** when `--int8-kv` is active, Path F
  save is skipped (saved file format v2 doesn't yet carry the scales
  region; v3 follow-up tracked at
  [#67](https://github.com/GeniePod/genie-ai-runtime/issues/67)). Use
  `--fp16-kv` if you need persistent KV today.
- Device-resident layer weights — copied into a per-layer device arena
  at load time instead of streaming from mmap'd host memory.
- Jetson power reporting handles L4T R36 sysfs paths and `nvpmodel` wattage
  strings such as `NV Power Mode: 25W`.

Latest on-device measurement: Qwen3-4B Q4_K_M, 25 W MAXN SUPER, GPU
locked at 918 MHz, 18-token user prompt (kernel sees N≈33 after Qwen3
chat-template wrap).

| | alpha.2 | alpha.3 | alpha.5 | alpha.6 | alpha.7 | alpha.8 | alpha.9² | alpha.10² | alpha.11² | alpha.12² ³ | Cumulative Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Prefill (33-tok cold)  | 8.2 tok/s | 15.4 tok/s | 15.2 tok/s | 15.68 tok/s | 28.16 tok/s | 38.68 tok/s | 38.7 tok/s | 38.8 tok/s | 38.8 tok/s | **38.0 tok/s** | **+363 %** |
| TTFT (33-tok cold)     | 2200 ms | 1181 ms | ~1180 ms | ~1170 ms | ~1170 ms | ~862 ms | 858 ms | 859 ms | 859 ms | **877 ms** | **−60 %** |
| Decode (40-tok decode) | 7.5 tok/s | 7.5 tok/s | **9.1 tok/s** | 9.1 tok/s | 9.1 tok/s | 9.1 tok/s | 9.1 tok/s | 10.0 tok/s | 10.0 tok/s | **9.9 tok/s** | **+32 %** |
| KV pool memory         | n/a       | n/a        | n/a        | n/a         | n/a         | n/a         | n/a       | n/a       | 144 MB    | **74 MB**       | **−49 % vs alpha.11** |
| Output | reference | bit-identical | bit-identical | bit-identical | sensibly-identical¹ | sensibly-identical¹ | sensibly-identical¹ | sensibly-identical¹ | sensibly-identical¹ | sensibly-identical¹ | ✓ |

### Sustained-prompt measurement (added 2026-05-16)

A second row captures a more serving-realistic shape: 57-token prompt
(N≈57 after chat-wrap of a 24-token user message) + 200 generated
tokens. Per-prefill-token gets a few % better than the 33-tok number
because dispatcher overhead amortizes over more N; decode drops
slightly as attention work scales with KV context length.

| Shape | Prefill | TTFT | Decode | KV mem (alpha.12 INT8) |
|---|---|---|---|---|
| 33-tok cold | 38.0 tok/s | 877 ms | 9.9 tok/s | 74 MB |
| 57-tok prefill + 200-tok sustained decode | **40.0 tok/s** | 1436 ms | **9.4 tok/s** | 74 MB |

(alpha.11 FP16 equivalents on the same shapes: 38.8 / 859 / 10.0 and
40.8 / 1410 / 9.5 — within 1–2 % of alpha.12 INT8 across the board.)

### Cold vs warm model load (added 2026-05-16)

Model-load wall time measured on the same Jetson with
[`scripts/bench_load.sh`](scripts/bench_load.sh), which drops the OS
pagecache (`echo 3 > /proc/sys/vm/drop_caches`) between runs. Qwen3-4B
Q4_K_M, 2.4 GB GGUF on NVMe, 25 W MAXN SUPER:

| | load_ms | effective MB/s |
|---|---|---|
| Cold (pagecache dropped, NVMe-bound) | 30258 | 79 |
| Warm (pagecache hit) | 1325 | 1797 |

Cold − warm = ~29 s on this hardware (cold is 22.8× slower). The cold
number is bounded by NVMe sequential read; once the GGUF is resident in
the page cache the model comes up in ~1.3 s, which is what makes
interactive re-launches feel instant. Any normal CLI run now prints the
number it observed:

```
[engine] Model loaded in 1325 ms (1797 MB/s)
```

¹ Path E's `mma.sync` reorders float adds differently than scalar FMAs;
byte-equality breaks at FP16 ULP, generated text remains
character-for-character the same on the reference prompt.

² alpha.9, alpha.10, and alpha.12 are **feature releases**, not
throughput releases. Prefill / decode / cold-TTFT match the prior
release within measurement noise. What they add:
- alpha.9 = Path F (persistent KV → **warm-turn TTFT 444 ms**, see below)
- alpha.10 = Path F5 (cache dir capped at 1 GB, oldest LRU-evicted, stale `*.tmp` cleaned)
- alpha.11 = release-hygiene cleanup (broken `--int8-kv` stub disabled; perf re-baselined)
- alpha.12 = Path I (proper INT8 KV cache, default; ~50 % KV memory saved)

³ alpha.12 numbers are with the new INT8 KV default. The +10 ms TTFT
delta vs alpha.11 (FP16) is per-position scale-lookup overhead in
attention; throughput is within noise. The **memory row is the
headline** — KV pool drops 144 MB → 74 MB (72 MB INT8 body + 2.25 MB
per-head scales). Set `--fp16-kv` to opt back into the alpha.11
numbers if you observe quality issues on a particular workload.

Same-day re-baseline used for alpha.7 → alpha.8 Δ (scalar fallback path
re-measured at 16.45 tok/s, alpha.8 = 38.68 tok/s, mean gap 1153 ms vs
combined σ ≈ 6 ms → ~190σ separation). **vs `llama-bench pp18 =
17.97 ± 0.65 tok/s` genie-ai-runtime now leads by +115 %.**

alpha.9 adds the multi-turn warm-start win on top: in a 2-turn
conversation where the second prompt shares 24 / 36 = 67 % of its
tokens with the first turn, **TTFT drops 857 → 444 ms (−48 %)** on
the second turn vs cold prefill. Prefill / decode throughput per
turn is unchanged; the gain is purely from skipping the matched
prefix. alpha.10 layers a 1 GB LRU cap on the cache dir so persistent
KV is production-safe — same warm-turn TTFT win, no unbounded disk
growth.

Long-prompt scaling validated 2026-05-16 across kernel N = 33 / 88 / 235:
the ~2.36× speedup over the scalar fallback is essentially uniform
across a 7× range of prompt sizes (E5 stays at ~26 ms / prefill-token,
scalar at ~62 ms / token). Generated text remains coherent at every
length. See [`docs/performance.md`](docs/performance.md) for the
detailed scaling table.

### TTFT — cold vs warm

Cold-turn TTFT is bounded by prefill physics: `TTFT ≈ N × ms/prefill-token
+ ~15 ms`. alpha.10's `ms/prefill-token` is ~27 ms (vs alpha.2's ~80 ms),
so cold TTFT has dropped by ~65 % at every prompt length. It still
*scales* linearly with N, though:

| Prompt | Kernel N | Cold TTFT |
|---|---|---|
| Short (`"Write a one-sentence summary..."`)            | 33   | ~860 ms |
| Medium (`"Explain in one short paragraph..."`)         | 45   | ~1250 ms |
| Medium-long (`"Explain the difference between..."`)    | 88   | ~2330 ms |
| Long (paragraph-form context)                          | 235  | ~6300 ms |

For multi-turn conversation traffic, **warm-turn TTFT** is the
relevant metric — Path F's hydrate skips prefill for the matched
prefix between the cached and new prompts:

| Scenario | TTFT |
|---|---|
| Cold (33-tok prompt, no cache) | 857 ms |
| Warm (39-tok prompt, 24 hydrated, 15 prefilled) | **444 ms** (−48 %) |
| Fully hydrated (prompt is exact prefix of cache) | < 200 ms — only one decode step |

vs llama.cpp on the same shapes: cold TTFT is ~2× faster (we run
prefill at 36 tok/s vs llama-bench's 18 tok/s) and warm TTFT has no
llama.cpp equivalent (no persistent KV). Further cold-TTFT
improvements (streaming, deeper kernel pipelining) tracked separately;
see open issues.

Path B detail: PRs [#13](https://github.com/GeniePod/genie-ai-runtime/pull/13)
→ [#17](https://github.com/GeniePod/genie-ai-runtime/pull/17), default
flip in [#18](https://github.com/GeniePod/genie-ai-runtime/pull/18).
Path C detail: PRs [#25](https://github.com/GeniePod/genie-ai-runtime/pull/25)
(Wo) → [#26](https://github.com/GeniePod/genie-ai-runtime/pull/26)
(gate/up) → [#27](https://github.com/GeniePod/genie-ai-runtime/pull/27)
(QKV triple) → default flip [#28](https://github.com/GeniePod/genie-ai-runtime/pull/28).
Path D detail: PR [#31](https://github.com/GeniePod/genie-ai-runtime/pull/31).
Path E detail: PRs [#34](https://github.com/GeniePod/genie-ai-runtime/pull/34)
(smoke test) → [#35](https://github.com/GeniePod/genie-ai-runtime/pull/35)
(single-tile skeleton) → [#36](https://github.com/GeniePod/genie-ai-runtime/pull/36)
(full GEMM kernel) → [#38](https://github.com/GeniePod/genie-ai-runtime/pull/38)
(per-(row, sb) scale precompute) → [#39](https://github.com/GeniePod/genie-ai-runtime/pull/39)
(integrate behind `JLLM_MMQ_Q4K`, alpha.7) → [#41](https://github.com/GeniePod/genie-ai-runtime/pull/41)
(multi-warp cooperative dequant) → [#42](https://github.com/GeniePod/genie-ai-runtime/pull/42)
(integrate multi-warp variant, alpha.8).
Path E plan + first-integration negative result: [#33](https://github.com/GeniePod/genie-ai-runtime/issues/33).
Path F detail: PRs [#46](https://github.com/GeniePod/genie-ai-runtime/pull/46)
(serialization round-trip) → [#47](https://github.com/GeniePod/genie-ai-runtime/pull/47)
(conv-id surface) → [#48](https://github.com/GeniePod/genie-ai-runtime/pull/48)
(save on turn end) → [#49](https://github.com/GeniePod/genie-ai-runtime/pull/49)
(pack only used tokens, 34× save speedup) → [#50](https://github.com/GeniePod/genie-ai-runtime/pull/50)
(format v2: persist token IDs) → [#51](https://github.com/GeniePod/genie-ai-runtime/pull/51)
(hydrate on turn start, alpha.9) → [#53](https://github.com/GeniePod/genie-ai-runtime/pull/53)
(LRU eviction + stale-tmp cleanup, alpha.10).
Path F plan: [#45](https://github.com/GeniePod/genie-ai-runtime/issues/45).
Path I detail: PRs [#63](https://github.com/GeniePod/genie-ai-runtime/pull/63)
(per-head scale storage) → [#64](https://github.com/GeniePod/genie-ai-runtime/pull/64)
(fp16_to_int8 wiring + kernel bug fixes) → [#65](https://github.com/GeniePod/genie-ai-runtime/pull/65)
(per-position scales through attention; remove broken N-sequential fallback) →
[#68](https://github.com/GeniePod/genie-ai-runtime/pull/68)
(default flip + Path F interop guard, alpha.12).
Path I plan + audit: [#62](https://github.com/GeniePod/genie-ai-runtime/issues/62).
Pending: [#67](https://github.com/GeniePod/genie-ai-runtime/issues/67) —
Path F format v3 to persist INT8 scales (closes the alpha.12 interop carve-out).
Plan + earlier negative results in
[#19](https://github.com/GeniePod/genie-ai-runtime/issues/19).

## Why

Existing runtimes are not designed for 8 GB unified memory shared with
voice STT, TTS, denoise, and a Home Assistant container:

- **llama.cpp** — portable, generic CUDA kernels, no Jetson memory
  awareness. Current default in GenieClaw; the runtime this project aims
  to replace.
- **TensorRT-LLM** — fast but datacenter-shaped (A100/H100), too heavy
  for Orin Nano's iGPU budget.
- **genie-ai-runtime** — memory-first, power-aware, Orin-tuned CUDA
  kernels (SM 8.7), pre-allocated KV/scratch pools. Single
  binary, single GGUF model file, single shared-memory budget that
  fits alongside `whisper-server` and `genie-core`.

## Architecture (modules)

| Module | Header | Responsibility |
| --- | --- | --- |
| `src/memory/` | `jllm_memory.h` | `MemoryBudget`, `OOMGuard`, `KVCachePool`, `ScratchPool` — every allocation accounted for before inference starts |
| `src/jetson/` | `jllm_jetson.h` | `PowerState` (nvpmodel 7–25 W), `ThermalState` (adaptive backoff), `JetsonInfo`, `LiveStats` |
| `src/kernels/` | `jllm_kernels.h` | Orin SM 8.7 CUDA: `gemv_q4` (INT4 dequant-fused), flash `attention`, `fused_norm`, `rope`, `softmax`, `convert` |
| `src/engine/` | `jllm_engine.h` | GGUF model load, transformer forward pass, tokenizer, top-k/top-p/temperature sampling |
| `src/server/` | — | OpenAI-compatible HTTP `/v1/chat/completions`, `/health`, `/v1/models` |

Master header: [`include/jllm.h`](include/jllm.h).

## Integration with GenieClaw

`genie-ai-runtime` ships an HTTP server (`src/main_server.cpp`) whose
`/v1/chat/completions` shape matches `llama-server`'s `/completion`
endpoint closely enough that GenieClaw can swap backends by changing
one config value (`llm_model_path` plus a future
`llm_backend = "llama-server" | "genie-ai-runtime"` toggle).

The plan is to land genie-ai-runtime as **opt-in** in GenieClaw's
alpha.8 cycle, run both backends in parallel on the same Jetson for a
week, A/B compare on the issue #19 latency banner, and flip the
default once parity is confirmed.

## Build

See [`docs/build.md`](docs/build.md). Quick path on a Jetson:

```
# Prereqs: CUDA toolkit, CMake ≥ 3.18, gcc/g++ aarch64-linux-gnu
git clone https://github.com/GeniePod/genie-ai-runtime.git
cd genie-ai-runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Outputs:
- `build/jetson-llm` — single-prompt / interactive CLI
- `build/jetson-llm-server` — HTTP server (drop-in replacement target for
  `llama-server`)

## Run

### Single prompt

```
./build/jetson-llm -m /path/to/model.gguf -p "Hello"
```

### Interactive chat

Drops into a REPL — type a prompt, hit enter, the response streams to
stdout. `quit` / `exit` (or Ctrl-C) leaves the loop; in-flight
generation stops gracefully.

```
./build/jetson-llm -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf -i --chat
```

- `-i` interactive loop
- `--chat` wraps each line with the Qwen chat template (auto-applied
  when the model name contains `instruct`/`chat` — explicit on the
  example above to be unambiguous; pass `--raw` to disable)
- Each turn is independent (no chat history). For persisted multi-turn
  history, add `--fp16-kv --conv-id mychat`:

  ```
  ./build/jetson-llm -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf \
      -i --chat --fp16-kv --conv-id mychat
  ```

  (Path F save is skipped under `--int8-kv` today, see alpha.12 note
  above and [#67](https://github.com/GeniePod/genie-ai-runtime/issues/67).)

A typical session looks like:

```
Loading model...
[engine] Model loaded in 1325 ms (1797 MB/s)
Model: Qwen3 4B Instruct Awq (36 layers, 32 heads, 8 KV heads, 2560 dim)

Entering interactive mode. Type 'quit' to exit.

> hello
Hello! How can I assist you today?
[10 tokens, 10.9 tok/s, peak 0 MB, 0.0°C]

> can you turn on the camera?
Sure! To turn on the camera, follow these steps depending on your device:
...
[256 tokens, 9.6 tok/s, peak 2683 MB, 58.5°C]

> quit
```

### Server (HTTP, OpenAI-compatible)

```
./build/jetson-llm-server -m /path/to/model.gguf -p 8080
```

### Flag reference

Short flags only. CLI: `-m` model, `-p` prompt, `-n` max tokens,
`-c` context, `-t` temperature, `-i` interactive, `-v` verbose,
`-h` help. Server: `-m` model, `-p` **port** (not prompt — heads up),
`-c` context, `--fp16-kv` to disable INT8 KV cache.

`--conv-id <id>` (Path F, see [#45](https://github.com/GeniePod/genie-ai-runtime/issues/45)) tags a request with a persistent-KV conversation identifier (`[A-Za-z0-9_-]{1,64}`). The HTTP server accepts the same id via a `conversation_id` field in `/v1/chat/completions`.

## Runtime Flags

Fast CUDA paths are enabled by default. Per-kernel fallbacks remain available
for debugging:

```
JLLM_BATCHED_PREFILL=0   # disable Path B (layer-major batched prefill); default: on
JLLM_Q4K_UINT32_LOADS=0  # disable Path C (Q4_K uint32 weight loads on decode); default: on
JLLM_MMQ_Q4K=0           # disable Path E (tensor-core MMQ Q4_K prefill GEMM); default: on
JLLM_KV_CACHE_DIR=...    # Path F: location for persistent KV files; default: /opt/jllm/data/kv-cache
JLLM_KV_CACHE_MAX_MB=N   # Path F5: total *.bin budget in MB (oldest LRU-evicted); default: 1024, 0 disables
JLLM_KV_CACHE_STALE_TMP_S=N  # Path F5: age (s) past which leftover *.tmp files get cleaned; default: 60
# (alpha.12) --int8-kv now WORKS and is the default. --fp16-kv opts back into full-precision KV.
JLLM_FAST_GEMV=0         # use CPU reference K-quant GEMV
JLLM_FAST_EMBD=0         # use CPU reference token embedding dequantization
JLLM_FAST_NORM=0         # use CPU reference RMSNorm
JLLM_FAST_ATTN=0         # use CPU reference decode attention (also disables chunked-prefill attention)
JLLM_FAST_SAMPLE=0       # use full-vocab reference sampling path
JLLM_DEVICE_OUTPUT=0     # disable automatic output projection device copy
JLLM_DEVICE_LAYERS=36    # opt into layer weight device copies; set N to cap copied layers
JLLM_MAPPED_WEIGHTS=0    # experimental: skip mapped-host CUDA weights and prefer device copies
JLLM_KV_OVERFLOW=1024    # optional extra CPU overflow tokens (default: 0)
JLLM_GEMV_ROWS=8         # legacy 8-row GEMV launch shape; default is 4 since the decode-side residual-fusion landed
JLLM_PROFILE=1           # print per-token decode timing breakdown
JLLM_DEBUG_KERNELS=1     # print first-token kernel diagnostics
```

For Qwen instruct/chat models, CLI prompts are chat-wrapped by default. Use
`--raw` to disable the template or `--think` to allow Qwen3 thinking output.

## Test

See [`TESTING.md`](TESTING.md) for the full test plan. Quick check:

```
cd build && ctest --output-on-failure
```

## Roadmap

Full plan in [`ROADMAP.md`](ROADMAP.md). Short version:

| Phase | Weeks | Goal |
| --- | --- | --- |
| Early validation | 1–2 | First coherent tokens on Jetson, baseline vs. llama.cpp |
| Core optimization | 3–5 | Kernel tuning ≥ 20% faster decode, 1000+ token stability |
| Production features | 6–8 | HTTP streaming, multi-model compat, systemd unit |
| Advanced capabilities | 9–11 | Speculative decoding, persistent KV cache |
| **v1.0** | 12 | 24-hour stability test, packaging, docs complete |

## License

MIT — see [`LICENSE`](LICENSE).

This is a permissive choice on purpose: `genie-ai-runtime` is
infrastructure that other projects (including non-AGPL ones) should be
able to embed cheaply. The integrating product, GenieClaw, stays AGPL-3.0.

## Related

- [GenieClaw](https://github.com/GeniePod/genie-claw) — the local home AI
  assistant this runtime is built for. Alpha.7 currently uses llama.cpp's
  `llama-server`; genie-ai-runtime is the planned replacement.
- [Original framework / roadmap context](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Projects/jetson-llm-runtime).
