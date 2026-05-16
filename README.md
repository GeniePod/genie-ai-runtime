# genie-ai-runtime

LLM inference engine for **Jetson Orin Nano Super 8 GB**. Built for one
job: serve a single small model fast and predictably in the memory
budget left over after voice STT, TTS, denoise, and a Home Assistant
container.

**`v1.0.0`** — first stable release. See [`ROADMAP.md`](ROADMAP.md) for
how we got here (the alpha-track path-by-path narrative) and what comes
next.

```
$ ./build/jetson-llm -m models/Qwen3-4B-Q4_K_M.gguf -p "Hello"
[engine] Model loaded in 1325 ms (1797 MB/s)
Model: Qwen3 4B Instruct Awq (36 layers, 32 heads, 8 KV heads, 2560 dim)
Hello! How can I assist you today?
[engine] Prefill: 13 tokens in 453 ms (28.7 tok/s)
[engine] Decode:   10 tokens in 916 ms (10.9 tok/s)
[engine] TTFT:    463 ms
```

## At a glance

- **Target HW:** Jetson Orin Nano Super 8 GB (SM 8.7, 102 GB/s, 67 TOPS GPU). Not portable.
- **Models:** GGUF only. Validated on Qwen3-4B-Q4_K_M; the architecture path supports any Qwen-family / Llama-3-family model the loader can parse.
- **Kernels:** custom CUDA for SM 8.7 — INT4 dequant-fused GEMV, tensor-core MMQ Q4_K prefill, flash attention, fused RMSNorm + RoPE + SwiGLU.
- **Memory model:** pre-allocated KV and scratch pools accounted before any inference starts; OOM-guard prevents crashes.
- **KV cache:** INT8 by default (alpha.12, FP16-ULP-bounded drift); FP16 opt-in. Persistent across turns (Path F) when a `conversation_id` is given.
- **Two binaries:**
  - `jetson-llm` — single-prompt / interactive CLI (default build).
  - `jetson-llm-server` — OpenAI-compatible HTTP server (opt-in: `-DJLLM_BUILD_SERVER=ON`).

## Why this exists

Existing runtimes aren't shaped for 8 GB unified memory that's already
sharing real-time voice traffic with the LLM:

- **llama.cpp** — portable, generic CUDA kernels, no Jetson memory
  awareness. The runtime [genie-claw](https://github.com/GeniePod/genie-claw)
  uses today; the one this project replaces.
- **TensorRT-LLM** — fast but datacenter-shaped (A100 / H100); too heavy
  for Orin Nano's iGPU budget.
- **genie-ai-runtime** — Orin-tuned CUDA kernels, pre-allocated pools,
  power-aware, single binary, single GGUF, single shared-memory budget
  that fits alongside `whisper-server` and `genie-core`.

## Performance (Qwen3-4B Q4_K_M, 25 W MAXN SUPER)

Headline numbers on Jetson Orin Nano Super 8 GB at v1.0.0:

| Workload | Number |
| -------- | ------ |
| Prefill (33-tok cold)                          | **38.0 tok/s** |
| Decode                                         | **9.9 tok/s** |
| Cold TTFT (33-tok prompt)                      | **877 ms** |
| Warm-turn TTFT (Path F hydrated, 67% prefix)   | **444 ms** |
| KV pool memory (1024 ctx)                      | **74 MB** (INT8 default) |
| Model load — cold (NVMe-bound)                 | **30 s** |
| Model load — warm (pagecache hit)              | **1.3 s** |

vs `llama-bench pp18 = 17.97 ± 0.65 tok/s`: **+115 % prefill** on the
same hardware + model. See [`ROADMAP.md`](ROADMAP.md) for the
alpha-track perf evolution (alpha.2 → v1.0) and the path-by-path
breakdown of how each gain landed.

## Quickstart

### Build

Prereqs: JetPack 6 (L4T R36.x, CUDA 12.6), CMake ≥ 3.20, gcc/g++ for
C++17, git. See [GeniePod/genie-os#1](https://github.com/GeniePod/genie-os/issues/1)
for the full install list for a fresh Jetson.

```bash
git clone https://github.com/GeniePod/genie-ai-runtime.git
cd genie-ai-runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Outputs:
- `build/jetson-llm` — the CLI
- `libjetson_llm_core.a` — the engine, link this from genie-claw

The HTTP server is opt-in (genie-claw embeds the engine library
directly). Add `-DJLLM_BUILD_SERVER=ON` to also produce
`build/jetson-llm-server`.

### CLI — single prompt

```bash
./build/jetson-llm -m /path/to/model.gguf -p "Hello"
```

### CLI — interactive chat

```bash
./build/jetson-llm -m /path/to/model.gguf -i --chat
```

`quit` / `exit` / Ctrl-C leaves the loop. `--think` allows Qwen3
reasoning output; omit for the no-think default.

### Server — HTTP, OpenAI-compatible

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DJLLM_BUILD_SERVER=ON
cmake --build build -j$(nproc)

./build/jetson-llm-server -m /path/to/model.gguf -p 8080
```

OpenAI-shape endpoints, SSE streaming, Qwen3 reasoning split into
`reasoning_content`. Full surface in [`docs/server.md`](docs/server.md).
For systemd deployment: `sudo ./scripts/setup.sh && sudo systemctl enable --now jetson-llm-server`.

## Project layout

| Module           | Header                              | Responsibility |
| ---------------- | ----------------------------------- | -------------- |
| `src/memory/`    | `include/jllm_memory.h`             | `MemoryBudget`, `OOMGuard`, `KVCachePool`, `ScratchPool` — every byte accounted before inference starts |
| `src/jetson/`    | `include/jllm_jetson.h`             | `PowerState` (nvpmodel 7–25 W), `ThermalState`, `JetsonInfo`, `LiveStats` |
| `src/kernels/`   | `include/jllm_kernels.h`            | Orin SM 8.7 CUDA — INT4 dequant-fused GEMV, tensor-core MMQ Q4_K prefill, flash attention, fused norm + RoPE + SwiGLU, FP16↔INT8 KV convert |
| `src/engine/`    | `include/jllm_engine.h`             | GGUF load, transformer forward pass, tokenizer, sampling |
| `src/persistence/` | `src/persistence/kv_cache_file.h` | Persistent KV cache (Path F) — atomic on-disk format, LRU eviction, model fingerprint |
| `src/server/`    | —                                   | Optional cpp-httplib server. OpenAI shape, SSE, reasoning split |

Master header: [`include/jllm.h`](include/jllm.h).

## Documentation

| Doc | Purpose |
| --- | --- |
| [`ROADMAP.md`](ROADMAP.md)                              | The alpha-track narrative (every Path, every perf win, every PR). What we did and how. |
| [`CHANGELOG.md`](CHANGELOG.md)                          | Per-release notes alpha.0 → v1.0.0 |
| [`docs/build.md`](docs/build.md)                        | Detailed build + prereqs |
| [`docs/server.md`](docs/server.md)                      | HTTP server reference (endpoints, request fields, SSE, systemd) |
| [`docs/architecture.md`](docs/architecture.md)          | Module-level design notes |
| [`docs/kernels.md`](docs/kernels.md)                    | CUDA kernel reference |
| [`docs/memory.md`](docs/memory.md)                      | KV / scratch / OOM-guard design |
| [`docs/performance.md`](docs/performance.md)            | Detailed perf scaling tables (alpha-track) |
| [`docs/jetson-hal.md`](docs/jetson-hal.md)              | Power / thermal / nvpmodel handling |
| [`docs/qwen3-vs-our-runtime.md`](docs/qwen3-vs-our-runtime.md) | Architecture notes specific to Qwen3 |
| [`docs/testing.md`](docs/testing.md)                    | Test layout + run instructions |
| [`docs/validation-week1.md`](docs/validation-week1.md)  | Week 1 bring-up notes (historical) |

## Integration with genie-claw

[`genie-claw`](https://github.com/GeniePod/genie-claw) (the local home
AI assistant this runtime is built for) links `libjetson_llm_core.a`
directly and calls `engine.generate()` in-process. The HTTP server is
only for non-embedded consumers and for A/B testing against
`llama-server`.

genie-claw currently runs on llama-server; the flip to genie-ai-runtime
is gated on the 24 h soak (issue [#7](https://github.com/GeniePod/genie-ai-runtime/issues/7)).

## Tests

```bash
cd build && ctest --output-on-failure
```

Stability soak harness for long-running validation:

```bash
./scripts/soak.sh /path/to/model.gguf --iters 100 --tokens 1024
```

Bench harnesses for cold/warm load and llama.cpp comparison live under
[`scripts/`](scripts/). See [`docs/testing.md`](docs/testing.md).

## License

MIT — see [`LICENSE`](LICENSE). Permissive on purpose so genie-claw
(AGPL-3.0) and other consumers can embed the engine cheaply.

## Related

- [`genie-claw`](https://github.com/GeniePod/genie-claw) — the home AI assistant this runtime serves.
- [`genie-ai-model`](https://github.com/GeniePod/genie-ai-model) — LoRA fine-tunes shrinking the runtime prompt by baking system / tool / persona into weights.
- [`genie-os`](https://github.com/GeniePod/genie-os) — base JetPack image for the GeniePod stack.
