# genie-ai-runtime

Jetson Orin-tuned LLM inference runtime — memory-first, power-aware,
with pre-allocated KV/scratch pools. Built to serve [`GenieClaw`](https://github.com/GeniePod/genie-claw)
on a 7.6 GB iGPU without crowding out whisper.cpp + Piper + Home Assistant.

**Target hardware:** Jetson Orin Nano Super 8 GB (SM 8.7, 102 GB/s, 67 TOPS GPU)
**Not supported:** x86, discrete GPUs, Windows, macOS — Jetson only.

## Status

`v0.1.0-alpha.3` — Path B (layer-major batched prefill) merged and
default-on. Validated on Jetson Orin Nano Super 8 GB with
`Qwen3-4B-Q4_K_M.gguf`.

Current validated path:
- Coherent Qwen3 instruct output with automatic chat template and no-think mode.
- GGUF tokenizer loads Qwen BPE merges and special tokens.
- Qwen3 architecture fixes: 128-dim attention heads, Q/K RMSNorm, NeoX RoPE,
  tied output embeddings.
- Default GPU decode kernels for K-quant GEMV, RMSNorm, and single-token
  attention on Orin SM 8.7.
- **Batched prefill**: K-quant GEMM, batched RMSNorm + QK-norm + SwiGLU,
  chunked-prefill attention. Reads each weight value from DRAM once per
  layer and re-uses it across all N prompt tokens.
- Jetson power reporting handles L4T R36 sysfs paths and `nvpmodel` wattage
  strings such as `NV Power Mode: 25W`.

Latest on-device measurement: Qwen3-4B Q4_K_M, 25 W MAXN SUPER, GPU
locked at 918 MHz, 18-token prompt.

| | alpha.2 (per-token) | alpha.3 (Path B) | Δ |
|---|---|---|---|
| Prefill | 8.2 tok/s | **15.4 tok/s** | **+88% (1.88×)** |
| TTFT | 2200 ms | **1181 ms** | **−47%** |
| Decode | 7.5 tok/s | 7.5 tok/s | unchanged |
| Output | reference | bit-identical | ✓ |

Path B detail: PRs [#13](https://github.com/GeniePod/genie-ai-runtime/pull/13)
→ [#17](https://github.com/GeniePod/genie-ai-runtime/pull/17), default
flip in [#18](https://github.com/GeniePod/genie-ai-runtime/pull/18).
Decode is the remaining bottleneck — see issue
[#19](https://github.com/GeniePod/genie-ai-runtime/issues/19) for the
Path C plan.

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

## Runtime Flags

Fast CUDA paths are enabled by default. Per-kernel fallbacks remain available
for debugging:

```
JLLM_BATCHED_PREFILL=0  # disable Path B (layer-major batched prefill); default: on
JLLM_FAST_GEMV=0  # use CPU reference K-quant GEMV
JLLM_FAST_EMBD=0  # use CPU reference token embedding dequantization
JLLM_FAST_NORM=0  # use CPU reference RMSNorm
JLLM_FAST_ATTN=0  # use CPU reference decode attention (also disables chunked-prefill attention)
JLLM_FAST_SAMPLE=0  # use full-vocab reference sampling path
JLLM_DEVICE_OUTPUT=0  # disable automatic output projection device copy
JLLM_DEVICE_LAYERS=36  # opt into layer weight device copies; set N to cap copied layers
JLLM_MAPPED_WEIGHTS=0  # experimental: skip mapped-host CUDA weights and prefer device copies
JLLM_KV_OVERFLOW=1024  # optional extra CPU overflow tokens (default: 0)
JLLM_GEMV_ROWS=4  # use the previous 4-row GEMV launch shape (default: 8)
JLLM_PROFILE=1  # print per-token decode timing breakdown
JLLM_DEBUG_KERNELS=1  # print first-token kernel diagnostics
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
