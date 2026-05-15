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

| Metric | alpha.7 (Path B + Path C + Path D + Path E, all default-on) |
|--------|---|
| Prompt tokens | 18 (kernel sees N=33 after chat-template wrap) |
| Prefill | 1166 ms, **28.3 tok/s** (5-sample, σ ≈ 2 ms) |
| Decode tokens | 32 |
| Decode | 3186 ms, **10.0 tok/s** |
| **TTFT** | **1179 ms** |
| Peak memory | ~1846 MB |
| Peak temperature | ~52 °C |
| Output | `Jetson Orin Nano is ideal for local LLM inference due to its high performance, low power consumption, and advanced AI capabilities, making it suitable for ...` |

Output text is **character-for-character identical** between the alpha.6 scalar
path and the alpha.7 Path E tensor-core path at temp=0 on the reference prompt,
even though `mma.sync` reorders float adds vs scalar FMAs and breaks
byte-equality at the FP16 ULP. Across Path B / C / D / E the generated text has
remained the same on this prompt.

Validated fast paths are all default:

| Path | Runtime switch to disable |
|------|---------------------------|
| Layer-major batched prefill (Path B) | `JLLM_BATCHED_PREFILL=0` |
| Q4_K uint32 decode weight loads (Path C) | `JLLM_Q4K_UINT32_LOADS=0` |
| Right-sized prefill GEMM unroll (Path D) | (compile-time constant, no env switch) |
| Tensor-core MMQ Q4_K prefill GEMM (Path E) | `JLLM_MMQ_Q4K=0` |
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

## Path C — Q4_K uint32 weight loads (2026-05-15)

Series: PRs #25 / #26 / #27 + default-flip #28. Two negative results
(#23 CUDA Graphs, #24 split-K) closed without merge — instructive,
documented inline. One more (#29 Q6_K uint32) failed on a block-size
alignment constraint, also closed.

Same hardware, model, prompt, and output as alpha.3. Each merged PR
was validated byte-identical to the previous main.

| PR | Change | Decode | Δ |
|---|---|---|---|
| main (alpha.3) | per-token byte loads | 7.5 tok/s | — |
| #23 ✗ | CUDA Graphs decode capture | 7.3 (−2.7%) | closed, no merge |
| #24 ✗ | split-K Q4_K GEMV | 7.6 (−1%) | closed, no merge |
| #25 | Q4_K Wo uint32 (residual-fused) | 8.0 | +6.7% |
| #26 | + Q4_K gate/up pair uint32 | 8.7 | +16.0% |
| #27 | + Q4_K QKV triple uint32 (V on byte path) | 8.9 | +18.7% |
| #28 | flip default to on | (no perf change) | — |
| #29 ✗ | Q6_K uint32 (block_q6_K is 210 B) | crashed, closed | — |
| **alpha.5 main** | | **9.1 tok/s** | **+21%** |

### What worked

The Q4_K block (`block_q4_K`, 144 bytes, divisible by 4) lets every
block start at a 4-byte-aligned address in row memory. That makes
the byte-by-byte `qs[]` reads in `dot_q4k_row` legally replaceable
with a single `uint32_t` load per lane per block. Lane mapping:

```
il       = lane >> 3            // which 32-byte qs sub-block (0..3)
sub_base = (lane & 7) << 2      // byte offset within sub-block
```

Per block per lane:

| | Byte path | uint32 path |
|---|---|---|
| Weight loads | 4 × `ld.global.b8` | **1 × `ld.global.b32`** |
| x loads | 8 × `ld.global.b16` | 8 × `ld.global.b16` (unchanged) |
| FMAs | 8 | 8 (identical math) |
| Scale recomputes | 4 (per inner iter) | 1 (constant per lane per block) |

4× fewer issued weight loads + zero inner-loop control. The wins
landed on three production kernels: `gemv_quant_add_typed_kernel<12>`
(Wo + residual), `gemv_quant_pair_typed_kernel<12,12>` (gate/up), and
`gemv_quant_triple_typed_kernel<12,12,14>` (QKV — Q4_K rows on the
uint32 path; the Q6_K Wv row still byte-by-byte inside the same
kernel).

### What didn't work, and why

**#23 CUDA Graphs (decode capture).** The hypothesis was that
~22,800 host-side `cudaLaunchKernel` calls (~24 µs each on Jetson
L4T) were a meaningful cost. Measured result: ≤0% speedup, slight
regression. The launches were already overlapping with GPU
execution on the LPDDR5-bound workload — eliminating host-side
overhead doesn't help when the CPU was already idle waiting for the
GPU. Lesson: `cudaLaunchKernel` API wall-time in `nsys` is NOT a
proxy for stall time; it includes time spent waiting on the GPU.

**#24 Split-K Q4_K GEMV.** The hypothesis was that per-warp MSHR
depth (~32 outstanding requests per warp) limited DRAM
concurrency. SPLIT_K=4 was supposed to give 4× more warps per row.
Measured result: ≤0%. Block size grew 128 → 512 threads, register
pressure dropped blocks-per-SM 4×, so total warps-per-SM was
unchanged. The bottleneck is arithmetic intensity per byte loaded
(Q4_K does 2 FMAs/byte; Q6_K does 4 — and Q6_K runs 3-4× faster per
peak bandwidth). The fix is in the inner loop, not the outer warp
schedule.

**#29 Q6_K uint32 weight loads.** Same lane-mapping trick should
have worked. It didn't. `block_q6_K` is 210 bytes, not divisible by
4, so consecutive blocks in row memory alternate between 4-aligned
and 2-aligned positions. Half the uint32 weight loads fault with
`cudaErrorMisalignedAddress`, the CUDA context goes sticky, and
every subsequent kernel (including byte-path fallback, RMSNorm,
attention) fails. Closed with the diagnosis. Q6_K can be vectorized
with `uint16` loads (2-byte aligned, which 210 B satisfies) for
roughly half the speedup; not pursued because Q6_K already runs at
30-40% of LPDDR5 peak and W_down + logits sum to only ~15% of
decode wall-clock.

### Where decode time goes at alpha.5

The pre-Path-C profile in
[#19](https://github.com/GeniePod/genie-ai-runtime/issues/19#issuecomment-4453495578)
showed 93% of decode in five K-quant GEMV kernels. Path C
vectorized four of them. The new bottleneck distribution is unknown
until someone reruns `nsys` against alpha.5 — that profile is the
next prerequisite for any further decode optimization.

## Path D — Right-sized Prefill GEMM Unroll (2026-05-15)

Series: PR #31 (single-line constexpr change).

Same hardware as alpha.5. Validation prompt was longer than the
standard `Hello, who are you?` (33 kernel tokens after chat-wrap vs
the standard 18), chosen because Path D's mechanism only fires when
the chunker has to split the work.

| | alpha.5 baseline | alpha.6 (Path D) | Δ |
|---|---|---|---|
| Prefill wall (33 tok) | 2253.8 ± 10.9 ms | **2104.0 ± 6.7 ms** | **−149.8 ms** |
| Prefill | 14.64 ± 0.07 tok/s | **15.68 ± 0.05 tok/s** | **+7.1%** |
| Decode | unchanged (different kernels) | unchanged | — |
| Output | reference | bit-identical | ✓ |

5 samples each branch, same-day same-machine. Mean gap ≈ 150 ms vs
combined σ ≈ 13 ms → ~12σ separation, not noise.

### The mechanism

`gemm_quant_batched_q{4,5,6}k_kernel` each maintain a per-thread
`acc[GEMM_MAX_BATCH]` register array and use a
`#pragma unroll for (t = 0; t < GEMM_MAX_BATCH; t++) if (t < N)`
inner loop. The unroll is necessary so the compiler keeps `acc[]`
in registers — a runtime-bounded loop with dynamic indexing forces
a spill to local memory and erases the bandwidth win. But every
unrolled iteration past `N` still consumes issue slots as predicated
FMA writes.

At `GEMM_MAX_BATCH = 32` with the typical chat-wrapped single-turn
prompt (N ≈ 33 → host dispatcher chunks into 32 + 1):

- First launch: 32/32 = 100 % useful issue slots
- Second launch: 1/32 ≈ 3 % useful → 97 % waste

Dropping `GEMM_MAX_BATCH` to 20 → chunks into 20 + 13:

- First launch: 20/20 = 100 % useful
- Second launch: 13/20 = 65 % useful

The actual win comes from second-chunk repacking plus a smaller
first-chunk unroll for any prompt that fits in one launch. Decode
kernels (`gemv_*` family) are untouched.

### Why 20 specifically

20 sits just above the typical N=18–20 we see after Qwen3
chat-template wrapping a single user turn. Below 20, single-turn
prompts get chunked into three or more launches and the per-launch
fixed cost dominates. At 24 or 32 the predicated-off tail returns.
20 is the cheap local optimum.

A future templated-on-N specialization (compile two or three N
specializations, dispatch at runtime) could close more of the gap to
llama.cpp by hitting 100 % utilization at every common prompt length
— tracked as a follow-up to Path D.

### vs llama.cpp

`llama-bench` on the same Jetson with the same model and `-ngl 22`:

| | Prefill (pp18) | Decode (tg64) |
|---|---|---|
| llama.cpp | 17.97 ± 0.65 tok/s | 6.33 ± 0.25 tok/s |
| genie-ai-runtime alpha.5 | 14.64 (same prompt) | 9.1 |
| genie-ai-runtime alpha.6 | 15.68 | 9.1 |
| Δ vs llama.cpp (alpha.6) | **−2.3 tok/s** | **+2.8 tok/s** |

Path D closed ~1/3 of the prefill gap from a one-line change. The
remainder was closed (and inverted) by Path E — see below.

## Path E — Tensor-core MMQ Q4_K Prefill GEMM (2026-05-16)

Series: PRs #34 (smoke test) → #35 (single-tile skeleton) → #36
(full GEMM standalone, 74 GFLOPS) → #37 ✗ (first integration:
−27 % regression, closed) → #38 (kernel rework, 210–276 GFLOPS) →
**#39 (integration: +70.8 % end-to-end prefill)** → default flip in
the alpha.7 release.

Same hardware as alpha.6. Same prompt and same model.

| | alpha.6 (scalar batched) | alpha.7 (Path E MMQ) | Δ |
|---|---|---|---|
| Prefill | 16.6 tok/s (1993 ± 2 ms) | **28.3 tok/s (1166 ± 2 ms)** | **+70.8 %** |
| TTFT | 2000 ms | **1179 ms** | **−41 %** |
| Decode | 10.0 tok/s | 10.0 tok/s | unchanged |
| Output | reference | sensibly-identical | ✓ |

Mean gap ≈ 827 ms vs combined σ ≈ 2 ms → ≫100σ separation. Output text is
character-for-character identical on the reference prompt at temp=0.

### What worked

Two specific optimizations on top of a naive `mma.sync.m16n8k16` port:

1. **Per-(row, sub-block) scale precompute into shared memory.** A
   Q4_K block has 8 sub-blocks of 32 quants each. Dequantizing one
   row's value requires `dall × sc[sb] × nibble − dmin × m[sb]`. The
   naive kernel ran the FP16→FP32 conversions of `d_raw` / `dmin_raw`
   and the `get_scale_min_k4` bit-packed lookup on every one of the
   32 lanes per (row, sub-block) — only the byte-per-lane changed.
   E4 precomputes 16 × 8 = 128 (d, dm) pairs once per Q4_K block into
   shared memory, distributed across the warp (4 entries per lane).
   The inner dequant loop then reads from shared mem instead of
   reconstructing scales.
2. **Drop `#pragma unroll` on the dequant row loop.** The 16-deep
   unroll forced the compiler to keep 16 copies of every intermediate
   in registers, pushing the kernel to 139 regs / thread and capping
   occupancy at ~29 %. Without the unroll the compiler shares a
   single set of intermediate registers across the 16 iterations,
   dropping register count to 96 and recovering arithmetic
   throughput per thread.

Combined effect: Wo (M=2560 N=33 K=2560) went from 74 GFLOPS (E3) to
210 GFLOPS (E4), and the kernel scales across all six Qwen3-4B
prefill GEMM shapes (gate/up = 276 GFLOPS, down = 215, QKV = 250).
Aggregate ~245 GFLOPS, vs ~126 for the scalar baseline, vs ~15 TFLOPS
hardware tensor-core peak.

### What didn't work, and why

**#37 first integration (closed, negative result).** The naive E3
kernel hit 74 GFLOPS standalone on Wo and the integration projection
suggested +20–30 % prefill. Actual result: prefill regressed from
16.6 → 12.0 tok/s (−27 %). Root cause: the standalone benchmark
only measured Wo (smallest shape, M=2560). The real prefill is
dominated by gate/up (M=9728) and down (K=9728), where the scalar
batched kernel uses 4 warps × `M / rows_per_block` blocks (~9700
total warps for gate/up) and saturates the SMs. The naive 1-warp-
per-CUDA-block MMQ kernel at 33 % occupancy got out-parallelized.
Closed without merge; E4 reworked the kernel and re-attempted in
E4b (#39). Lesson: benchmark *all* the shapes the workload visits,
not just the headline one.

### Why the MMA helps even at only 1.6 % tensor-core peak

vs scalar CUDA-core FMAs at ~12 % compute utilization, the kernel
nominally has 7.5× headroom. We converted ~2× of that. Where the
other ~3.7× went:

| Cost | Approx fraction |
|---|---|
| Q4_K dequant scalar arithmetic (shared-mem precompute + inner) | ~40 % |
| Shared-mem load/store for the staging A-tile | ~15 % |
| B-fragment global memory loads (activations, not cached across MMAs) | ~10 % |
| MMA + accumulator update (the actual work) | ~25 % |
| Boundary checks + launch overhead | ~10 % |

Recovering more of the headroom is what E5 (multi-warp cooperative
dequant) would target: have 4 warps share one dequanted A-tile across
4 N-stripes, amortizing the dequant cost 4× and increasing
arithmetic intensity per shared-mem byte. Tracked as future work in
#33.

### vs llama.cpp

| | Prefill (pp18) | Decode (tg64) |
|---|---|---|
| llama.cpp | 17.97 ± 0.65 tok/s | 6.33 ± 0.25 tok/s |
| genie-ai-runtime alpha.6 | 15.68 | 9.1 |
| genie-ai-runtime alpha.7 | **28.3** | 9.1 |
| Δ vs llama.cpp (alpha.7) | **+10.3 tok/s (+57 %)** | **+2.8 tok/s (+44 %)** |

We now lead on both prefill and decode. The remaining gap to
tensor-core peak is the lever for any next-round optimization.

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
