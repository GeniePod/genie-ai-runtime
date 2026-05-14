# Performance

## Hardware Specs (Orin Nano Super)

| Spec | Value |
|------|-------|
| GPU TOPS (INT8, sparse) | 67 |
| Memory | 8 GB LPDDR5 |
| Bandwidth | 102 GB/s |
| CUDA cores | 1024 (8 SMs × 128) |
| Tensor Cores | 32 |
| Power modes | 7W / 10W / 15W / 25W MAXN SUPER |
| Ridge point | 0.66 OP/byte |

The DLA is not used — the LLM workload runs entirely on the GPU.

## Current Validated Result

Hardware: Jetson Orin Nano Super 8 GB, L4T 36.4, CUDA 12.6, 25 W MAXN
SUPER, GPU locked at 918 MHz.
Model: `Qwen3-4B-Q4_K_M.gguf` (2381 MB)
Prompt: `Hello, who are you?` (18 tokens after Qwen chat template)
Context: 1024 tokens

| Metric | alpha.3 (Path B default-on) |
|--------|---|
| Prompt tokens | 18 |
| Prefill | 1168 ms, **15.4 tok/s** |
| Decode tokens | 47 |
| Decode | 6307 ms, **7.5 tok/s** |
| **TTFT** | **1181 ms** |
| Peak memory | 3161 MB |
| Peak temperature | 53.0°C |
| Output | `Hello! I'm Qwen, a large-scale language model developed by Alibaba Group. I can help with a variety of tasks, including answering questions, writing articles, creating stories, and more. How can I assist you today?` |

Validated fast paths are all default:

| Path | Runtime switch to disable |
|------|---------------------------|
| Layer-major batched prefill (Path B) | `JLLM_BATCHED_PREFILL=0` |
| K-quant GEMV / GEMM | `JLLM_FAST_GEMV=0` |
| Token embedding dequantization | `JLLM_FAST_EMBD=0` |
| RMSNorm | `JLLM_FAST_NORM=0` |
| Decode + chunked-prefill attention | `JLLM_FAST_ATTN=0` |

The output projection is materialized into CUDA device memory automatically
when the memory budget allows it. Set `JLLM_DEVICE_OUTPUT=0` to force the
mapped-host path. Transformer layer weights stay mapped-host by default because
duplicating them on top of a pinned full-file mapping costs memory without a
measured speedup. For experimental llama.cpp-style placement, set
`JLLM_MAPPED_WEIGHTS=0 JLLM_DEVICE_LAYERS=36` to skip full-file CUDA host
registration and copy layer weights into CUDA allocations instead. Extra CPU
overflow KV tokens are opt-in with `JLLM_KV_OVERFLOW=<tokens>`; the default
keeps memory available for hot decode weights first.

These numbers are correctness/brings-up numbers, not final throughput targets.
The largest remaining cost is still K-quant weight bandwidth and the final
vocabulary projection/sampling path.

## Path B — Layer-major Batched Prefill (2026-05-15)

Series: PRs #13 → #17 + default-flip #18.

Same hardware, model, prompt, and output as the alpha.2 baseline. Each
PR was validated byte-identical to the previous main before merging.

| PR | Change | Prefill | TTFT |
|---|---|---|---|
| main (alpha.2) | per-token prefill baseline | 8.2 tok/s | 2200 ms |
| #13 | scaffolding (no kernel change) | 8.2 (byte-equal) | 2200 ms |
| #14 | batched QKV GEMM | 9.0 (+11%) | 2003 ms |
| #15 | + batched gate/up + SwiGLU | 12.0 (+33%) | 1520 ms |
| #16 | + batched Wo + W_down + residuals | 14.8 (+23%) | 1231 ms |
| #17 | + chunked-prefill attention | **15.4 (+4%)** | **1181 ms** |
| #18 | flip default to on | (no perf change) | — |

**Final: prefill 1.88×, TTFT down 47%**, decode unchanged at 7.5 tok/s
because decode is N=1 and there's nothing to amortize.

### Why the wins

The bandwidth math undercounted the actual gain in every PR; we kept
beating bandwidth-only estimates by 3-10×. Two effects beyond raw
weight bandwidth:

1. **Kernel launch overhead.** A per-token Qwen3-4B prefill ran ~250
   kernel launches per token × 18 tokens = 4500 launches; the batched
   path runs ~10 launches per layer × 36 layers ≈ 360. At ~5-10 µs per
   launch that's tens of ms saved.
2. **L2 cache pressure.** With per-token, each prompt token's full
   weight set is streamed before the next one starts, so the LPDDR5
   controller sees the same weight pages 18 times. Layer-major
   touches each weight page once and lets the L2 keep activations
   hot across the per-layer loop.

The two effects together explain why each individual PR over-delivered
relative to the static bandwidth estimate.

### Path C — decode

Decode at 7.5 tok/s is ~13% of LPDDR5 ceiling (38 tok/s on this
model). Reference llama.cpp on the same hardware hits ~18 tok/s.
There is real headroom; the plan is tracked in issue
[#19](https://github.com/GeniePod/genie-ai-runtime/issues/19) and
needs a profiler pass before any kernel work begins.

## Why LLM Decode is Bandwidth-Bound

LLM autoregressive decode generates one token at a time. Each token requires reading the **entire** weight matrix from DRAM:

```
Llama 3.2 3B INT4 — one decode step:

  Weight read:    1.5 GB from DRAM
  Compute:        6 GFLOP (3B × 2 ops)
  Time to read:   1.5 GB / 102 GB/s = 14.7 ms
  Time to compute: 6 GFLOP / 67 TOPS = 0.09 ms

  → 99.4% of time is reading weights
  → Compute utilization: 0.6%
  → Theoretical max: ~68 tokens/sec (bandwidth-limited)
```

Smaller model = fewer bytes to read = more tokens/sec. Quantization directly translates to throughput.

## Historical Target Estimates

The estimates below are retained as targets from the original roadmap. They
should not be read as current measured performance.

### TinyLlama 1.1B (Q4_K_M, 669 MB) — Test Model

| Metric | 25W mode | 15W mode |
|--------|----------|----------|
| Prompt eval (512 tok) | ~200 tok/s | ~120 tok/s |
| Decode (128 tok) | ~65 tok/s | ~40 tok/s |
| Peak memory | ~2.5 GB | ~2.5 GB |
| Peak temperature | <60°C | <55°C |

### Llama 3.2 3B (Q4_K_M, 1.8 GB) — Target Model

| Metric | 25W mode | 15W mode |
|--------|----------|----------|
| Prompt eval (512 tok) | ~65 tok/s | ~40 tok/s |
| Decode (128 tok) | ~25 tok/s | ~15 tok/s |
| Peak memory | ~4 GB | ~4 GB |
| Peak temperature | <70°C | <60°C |

### Phi-4 Mini 3.8B (Q4_K_M, 2.3 GB) — Stress Test

| Metric | 25W mode |
|--------|----------|
| Prompt eval (512 tok) | ~50 tok/s |
| Decode (128 tok) | ~20 tok/s |
| Peak memory | ~4.5 GB |

*All values are estimates. Actual performance depends on thermal design, context length, and prompt content.*

## Profiling

### Nsight Systems — Timeline Profile

```bash
./scripts/profile.sh models/tinyllama.gguf
# Creates: profile_YYYYMMDD_HHMMSS.nsys-rep
```

Shows:
- Kernel-by-kernel timeline
- Memory transfer timing
- CPU-GPU synchronization points
- Kernel launch overhead

### Key Metrics to Watch

```bash
nsys stats profile.nsys-rep
```

Expected output:
```
Kernel                              Time%    Calls
────────────────────────────────────────────────────
jllm::gemv_q4_kernel               38.2%    312
jllm::flash_attention_decode_kernel 28.1%    156
jllm::fused_rmsnorm_residual_kernel 11.4%    312
jllm::swiglu_kernel                  7.8%    156
jllm::rope_kernel                    4.2%    312
jllm::vec_add_kernel                 3.1%    312
other                                7.2%    ...
```

### Benchmark Script

```bash
./scripts/bench.sh models/tinyllama.gguf
```

Records:
- System state (power mode, GPU freq, RAM, temperature)
- Short generation (128 tokens): tok/s
- Long generation (256 tokens): tok/s
- Memory profile during inference (RSS every second)
- Temperature during inference

## Optimization Targets

### Level 1: Fused Kernels (Done)

| Operation | Without fusion | With fusion | Saving |
|-----------|---------------|-------------|--------|
| RMSNorm + residual | 3 kernels, 6 DRAM ops | 1 kernel, 3 DRAM ops | 2× |
| SwiGLU | 2 kernels, 4 DRAM ops | 1 kernel, 2 DRAM ops | 2× |
| Dequant + GEMV | 2 kernels, 2× bandwidth | 1 kernel, 1× bandwidth | 3.5× |

### Level 2: Tile Size Tuning (Done)

| Parameter | Desktop GPU (H100) | Orin Nano | Why different |
|-----------|-------------------|-----------|---------------|
| GEMV block | 256 threads | 128 threads | Fewer SMs, less occupancy pressure |
| Attention tile | 128 KV tokens | 64 KV tokens | 48 KB shared (not 228 KB) |
| GEMM tile | 128×128 | 64×64 | Less shared memory per SM |

### Level 3: CUDA Graphs (Implemented)

Captures decode step as a graph, replays with single `cudaGraphLaunch()`.
Saves ~1 ms per step from kernel launch overhead.

### Level 4: Future Optimizations

| Optimization | Expected gain | Effort |
|-------------|---------------|--------|
| Tensor Core WMMA for prefill | 2–3× prefill speed | 2 days |
| Persistent kernels (stay resident on SM) | 10–20% decode | 3 days |
| Custom memory allocator (bypass CUDA) | 5% less overhead | 2 days |
| INT4 KV cache (not just INT8) | 2× more context | 1 day |
| Speculative decoding with draft model | 1.5–3× tok/s | 3 days |

## Roofline Analysis

```
TFLOPS
  │
  │                    ──── 67 TOPS peak (INT8)
  │                 ╱
  │              ╱
  │           ╱
  │        ╱  ← slope = 102 GB/s bandwidth
  │     ╱
  │  ╱
  └───────────────── Arithmetic Intensity (OP/byte)
        ↑
  ridge = 0.66

  LLM decode AI ≈ 0.5 OP/byte → severely left of ridge → bandwidth-bound
  Prefill GEMM AI ≈ 50+ OP/byte → right of ridge → compute-bound
```

All decode optimizations must reduce bandwidth (quantization, fusion, caching).
Prefill optimizations should maximize Tensor Core utilization.

## Power Efficiency

| Model | Tokens/sec | Power (W) | Tokens/Joule |
|-------|-----------|-----------|-------------|
| TinyLlama 1.1B @ 25W | ~65 | 25 | 2.6 |
| TinyLlama 1.1B @ 7W | ~20 | 7 | 2.9 |
| Llama 3.2 3B @ 25W | ~25 | 25 | 1.0 |
| Llama 3.2 3B @ 7W | ~8 | 7 | 1.1 |

*7W mode is more power-efficient (tokens/joule) despite being slower.*
