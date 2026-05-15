# genie-ai-runtime

Jetson Orin-tuned LLM inference runtime — memory-first, power-aware,
with pre-allocated KV/scratch pools. Built to serve [`GenieClaw`](https://github.com/GeniePod/genie-claw)
on a 7.6 GB iGPU without crowding out whisper.cpp + Piper + Home Assistant.

**Target hardware:** Jetson Orin Nano Super 8 GB (SM 8.7, 102 GB/s, 67 TOPS GPU)
**Not supported:** x86, discrete GPUs, Windows, macOS — Jetson only.

## Status

`v0.1.0-alpha.8` — Path E E5 (multi-warp cooperative MMQ Q4_K prefill
GEMM) merged on top of alpha.7's E4 single-warp MMQ. Validated on
Jetson Orin Nano Super 8 GB with `Qwen3-4B-Q4_K_M.gguf` — **prefill
38.7 tok/s, ahead of llama.cpp's 17.97 by +115 %.**

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
- Device-resident layer weights — copied into a per-layer device arena
  at load time instead of streaming from mmap'd host memory.
- Jetson power reporting handles L4T R36 sysfs paths and `nvpmodel` wattage
  strings such as `NV Power Mode: 25W`.

Latest on-device measurement: Qwen3-4B Q4_K_M, 25 W MAXN SUPER, GPU
locked at 918 MHz, 18-token user prompt (kernel sees N≈33 after Qwen3
chat-template wrap).

| | alpha.2 | alpha.3 | alpha.5 | alpha.6 | alpha.7 | alpha.8 | Cumulative Δ |
|---|---|---|---|---|---|---|---|
| Prefill | 8.2 tok/s | 15.4 tok/s | 15.2 tok/s | 15.68 tok/s | 28.16 tok/s | **38.68 ± 0.1 tok/s** | **+372 %** |
| TTFT | 2200 ms | 1181 ms | ~1180 ms | ~1170 ms | ~1170 ms | **~860 ms** | **−61 %** |
| Decode | 7.5 tok/s | 7.5 tok/s | **9.1 tok/s** | 9.1 tok/s | 9.1 tok/s | 9.1 tok/s | **+21 %** |
| Output | reference | bit-identical | bit-identical | bit-identical | sensibly-identical¹ | sensibly-identical¹ | ✓ |

¹ Path E's `mma.sync` reorders float adds differently than scalar FMAs;
byte-equality breaks at FP16 ULP, generated text remains
character-for-character the same on the reference prompt.

Same-day re-baseline used for alpha.7 → alpha.8 Δ (scalar fallback path
re-measured today at 16.45 tok/s, alpha.8 = 38.68 tok/s, mean gap
1153 ms vs combined σ ≈ 6 ms → ~190σ separation). **vs `llama-bench
pp18 = 17.97 ± 0.65 tok/s` genie-ai-runtime now leads by +115 %.**

Long-prompt scaling validated 2026-05-16 across kernel N = 33 / 88 / 235:
the ~2.36× speedup over the scalar fallback is essentially uniform
across a 7× range of prompt sizes (E5 stays at ~26 ms / prefill-token,
scalar at ~62 ms / token). Generated text remains coherent at every
length. See [`docs/performance.md`](docs/performance.md) for the
detailed scaling table.

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

```
# CLI (single prompt)
./build/jetson-llm -m /path/to/model.gguf -p "Hello"

# CLI (interactive chat loop)
./build/jetson-llm -m /path/to/model.gguf -i

# Server (HTTP, OpenAI-compatible)
./build/jetson-llm-server -m /path/to/model.gguf -p 8080
```

Short flags only. CLI: `-m` model, `-p` prompt, `-n` max tokens,
`-c` context, `-t` temperature, `-i` interactive, `-v` verbose,
`-h` help. Server: `-m` model, `-p` **port** (not prompt — heads up),
`-c` context, `--fp16-kv` to disable INT8 KV cache.

`--conv-id <id>` (Path F, see [#45](https://github.com/GeniePod/genie-ai-runtime/issues/45)) tags a request with a persistent-KV conversation identifier (`[A-Za-z0-9_-]{1,64}`). The HTTP server accepts the same id via a `conversation_id` field in `/v1/chat/completions`. Currently plumbing only — the engine logs the id but does not yet persist or hydrate KV state. F3 wires the save/load hooks against the on-disk format from [#46](https://github.com/GeniePod/genie-ai-runtime/pull/46).

## Runtime Flags

Fast CUDA paths are enabled by default. Per-kernel fallbacks remain available
for debugging:

```
JLLM_BATCHED_PREFILL=0   # disable Path B (layer-major batched prefill); default: on
JLLM_Q4K_UINT32_LOADS=0  # disable Path C (Q4_K uint32 weight loads on decode); default: on
JLLM_MMQ_Q4K=0           # disable Path E (tensor-core MMQ Q4_K prefill GEMM); default: on
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
