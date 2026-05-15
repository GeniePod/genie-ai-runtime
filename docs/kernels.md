# CUDA Kernels

All production decode kernels are tuned for Jetson Orin SM 8.7: 48 KB shared
memory, 128-thread blocks, and the 8-SM Orin Nano Super profile validated on
L4T R36 / CUDA 12.6. Runtime hardware probing is used for display and sizing;
the kernels themselves avoid SKU-specific assumptions where possible.

Fast paths are enabled by default after the Week 1 Qwen3 validation. Each path
can be disabled independently for debugging:

```
JLLM_FAST_GEMV=0
JLLM_FAST_EMBD=0
JLLM_FAST_NORM=0
JLLM_FAST_ATTN=0
```

## gemv_q4 — GGML K-Quant Dequant-Fused GEMV

**File:** `src/kernels/gemv_q4.cu`
**Validated:** Q4_K, Q5_K, Q6_K tensors inside Qwen3-4B Q4_K_M.

### What it does

Computes `y[M] = W[M×K] × x[K]` where W is a GGML K-quant tensor. Qwen3
Q4_K_M mixes Q4_K, Q5_K, and Q6_K tensors, so the dispatcher selects the
matching kernel by GGML tensor type.

Weights stay in the mmap'd GGUF file. The loader registers that mmap with CUDA
mapped-host access and gives the kernels a device-visible alias; raw CPU mmap
pointers are never dereferenced by CUDA kernels.

### Why fused dequant matters

Without fusion: read INT4 weights → write FP16 weights to DRAM → read FP16 weights → compute.
With fusion: read INT4 weights → dequantize in registers → compute. Never writes FP16 weights to DRAM.

Bandwidth: K/2 bytes (INT4) vs K×2 bytes (FP16) = **3.5× reduction**.

### Orin tuning

```
Grid:   (ceil(M / 4), 1)
Block:  128 threads = 4 warps
        Each warp handles one output row (M dimension)
        32 lanes stride across K dimension (coalesced uint32 loads)

Reduction: warp shuffle (__shfl_xor_sync) — no shared memory needed
Dequant:   8 INT4 values from one uint32, multiply by group scale
```

### Key code path

```
1. Resolve host GGUF tensor pointer to CUDA mapped device pointer
2. One warp computes one output row
3. Dequantize K-quant blocks in registers
4. Dot product with x
5. Warp shuffle reduce
6. Lane 0 writes y[row]
```

## Q4_K uint32 weight loads (Path C, default-on)

**File:** `src/kernels/gemv_q4.cu` (helper: `dot_q4k_row_uint32`,
kernels: `gemv_quant_add_uint32_q4k_kernel`,
`gemv_quant_pair_uint32_q4k_kernel`, `gemv_quant_triple_uint32_q4k_kernel`,
`gemv_quant_triple_uint32_q4k_q4k_q6k_kernel`).
**Validated:** Bit-identical output to the byte path on Qwen3-4B Q4_K_M
decode; +21% decode tok/s end-to-end. Series: PRs #25 / #26 / #27, default
flip #28.

### What it does

Same math as the byte-by-byte `dot_q4k_row` helper, but each lane reads
**four packed q-bytes as one `uint32_t`** from `blk.qs + 32*il + sub_base`
instead of one byte per inner iteration. The 4-iteration `il` inner loop
disappears: a single warp instruction (32 lanes × 4 bytes) covers all
128 bytes of `qs` for the block in one L1 line.

Lane mapping:

```
il       = lane >> 3            // which 32-byte qs sub-block (0..3)
sub_base = (lane & 7) << 2      // byte offset within sub-block (0,4,...,28)
```

Per block per lane:

| | Byte path | uint32 path |
|---|---|---|
| Weight loads | 4 × `ld.global.b8` | **1 × `ld.global.b32`** |
| x loads | 8 × `ld.global.b16` | 8 × `ld.global.b16` (unchanged) |
| FMAs | 8 | 8 (same K-positions, same arithmetic) |
| Scale lookups | recomputed per inner iter | constant per lane per block |

### Why this is legal

`sizeof(block_q4_K) == 144`, which is divisible by 4. So every block
starts at a 4-aligned address in row memory, and `blk.qs + 32*il + sub_base`
is also 4-aligned (32 and 4 are both multiples of 4). The same pattern
**doesn't apply to `block_q6_K`** (210 B, divisible by 2 but not 4) — see
#29 for the failed attempt.

### Production callers (decode hot path)

| Kernel | Routed to | Decode share |
|---|---|---|
| Wo (residual-fused) | `gemv_quant_add_uint32_q4k_kernel` | ~20% |
| gate/up pair | `gemv_quant_pair_uint32_q4k_kernel` | ~44% (the biggest) |
| QKV triple (Q4_K + Q4_K + Q6_K, Qwen3-4B) | `gemv_quant_triple_uint32_q4k_q4k_q6k_kernel` | ~14% (Q6_K Wv row stays on byte path) |
| QKV triple (Q4_K + Q4_K + Q4_K, other models) | `gemv_quant_triple_uint32_q4k_kernel` | varies |

Each dispatcher (`gemv_quant_add_gpu` / `gemv_quant_pair_gpu` /
`gemv_quant_triple_gpu`) checks `q4k_uint32_loads_enabled()` and the
`ggml_type` mix, then routes accordingly. Any `cudaError` falls through to
the typed byte-load kernel — never crashes generation.

### Bit-equality argument

The 32-lane warp_reduce_sum that follows the per-lane dot is identical
to the byte path. Each lane's accumulator now folds 8 FMAs over one
sub-block's K-positions instead of 8 FMAs spread across four `il`
sub-blocks' K-positions, so float-add associativity is broken in
principle. In practice on Qwen3-4B at FP16 the rounded result matches
the byte path **to the exact bit** on the standard test prompt — verified
across PRs #25 / #26 / #27 (47-token completion matches verbatim each
time). The `JLLM_Q4K_UINT32_LOADS=0` opt-out is there as the rollback for
any deployment that trips a numeric edge case.

## gemm_quant_batched — K-Quant GEMM for Prefill

**File:** `src/kernels/gemv_q4.cu` (kernels: `gemm_quant_batched_q4k_kernel`,
`gemm_quant_batched_q5k_kernel`, `gemm_quant_batched_q6k_kernel`).
**Validated:** All weight matrices in Qwen3-4B Q4_K_M prefill.

### What it does

Computes `y[N × M] = x[N × K] · Wᵀ[K × M]` for Q4_K / Q5_K / Q6_K
weights. Each warp owns one output row `r` and holds `N` register
accumulators; for every weight value loaded from DRAM, the warp issues
`N` FMAs against `x[t][k]`. The weights are streamed from DRAM **once**
and re-used across all `N` query tokens — that's the entire point of
the batched path.

```
Grid:   (ceil(M / rows_per_block), 1)
Block:  rows_per_block × 32 threads (one warp per output row)
Per-warp register state: GEMM_MAX_BATCH (32) fp32 accumulators
```

`N=1` fast-routes to `gemv_quant` so the decode/per-token path is
untouched and bit-identical.

### Why `acc[]` stays in registers

A naïve `for (int t = 0; t < N; t++) acc[t] += ...` with runtime `t`
forces dynamic indexing into the register array, which the NVCC
compiler spills to local memory and tanks the bandwidth win. The
kernel uses `#pragma unroll` over the static `GEMM_MAX_BATCH=32` bound
with an inner `if (t < N)` predicate, so the compiler keeps every
accumulator slot in a real register (verified zero spills in PTXAS
output — 56 regs/thread for Q4_K, 64 for Q5/Q6_K).

### Output layout

`y[token * M + row]` — row-major, token-outer. Matches what
`gemm_quant_batched(attn_proj, lw.wo, ...)` and friends expect when
the next batched kernel slices by token.

### Byte-equality

Within each warp the partial-sum order across `(block, inner_iter,
K-stride)` is identical to `gemv_quant`. Token `t`'s accumulator
receives exactly the same FMAs in the same order, so after
`warp_reduce_sum` the output is bit-identical to the per-token GEMV
call. Validated end-to-end across PRs #14, #15, #16 (47-token
completions match byte-for-byte).

## fused_norm — RMSNorm + Residual Add

**File:** `src/kernels/fused_norm.cu`
**Validated:** layer RMSNorm, Qwen3 Q/K per-head RMSNorm, final RMSNorm.

### What it does

Computes `output = RMSNorm(x) × weight` in one kernel.

Without fusion: 3 kernels, 6 DRAM accesses.
With fusion: 1 kernel, 3 DRAM accesses (read x, read weight, write output).

### Algorithm

```
Pass 1: Load x, compute sum of squares (variance)
  - Each thread handles hidden_dim/blockDim elements
  - Warp shuffle reduce for partial sums
  - Cross-warp reduce via shared memory (4 floats for 4 warps)
  - Compute rrms = rsqrt(variance/dim + eps)

Pass 2: Normalize and scale
  - normed = x * rrms * weight
  - Write output
```

### Shared memory usage

The current kernel does not cache the full hidden vector in shared memory. It
reads the input once for the sum-of-squares reduction and once for the final
scale/write. This avoided an earlier shared-memory layout bug that produced
alternating zeros in Qwen3 RMSNorm output.

## attention — Flash Attention Decode

**File:** `src/kernels/attention.cu`
**Validated:** Qwen3 single-token decode attention with GQA.

### What it does

Single-query attention for decode (one new token). Computes:
```
output = softmax(Q × K^T / sqrt(d)) × V
```

without materializing the full seq×seq attention matrix.

### Algorithm (online softmax)

```
For each KV tile (64 tokens):
  1. Compute Q×K^T for tile (each thread handles some time steps)
  2. Find tile max (warp reduce + block reduce via shared memory)
  3. Update running max, correct previous accumulators by exp(old_max - new_max)
  4. Exponentiate scores, accumulate sum
  5. Accumulate P × V into s_out[head_dim] in shared memory
Final: output = s_out / running_sum
```

### Orin tuning

```
Grid:   (n_heads, 1)  — one block per query head
Block:  128 threads
Shared: ATTN_TILE_KV (64) + head_dim floats for scores + output accumulator
Tile:   64 KV tokens per iteration

KV layout: [seq_len, n_kv_heads, head_dim]
GQA: kv_head = head / (n_heads / n_kv_heads)
INT8 KV: dequantize on-the-fly in the dot product loop
```

### Memory access pattern

- Q: read once from global, stays in L1 (small: 128 × 2 = 256 bytes)
- K: read tile by tile, 64 × 128 × element_size per tile
- V: read tile by tile, same pattern
- Scores: shared memory only (never written to DRAM)
- Output: one write at the end

## attention (chunked prefill) — Flash Attention for N queries

**File:** `src/kernels/attention.cu` (kernel:
`flash_attention_prefill_batched_kernel`).
**Validated:** Qwen3 multi-token prefill in Path B.

### What it does

Same online-softmax inner loop as `flash_attention_decode_kernel`, but
extended to process all `N` query tokens against a populated K/V cache
in **one** kernel launch. Each block computes one `(query_head,
query_token)` pair.

```
Grid:   (n_heads, N)              ← N is the batched prefill width
Block:  128 threads
Per-block seq_len: start_pos + token + 1  (causal mask)
```

For Qwen3-4B with `n_heads = 32` and `N = 18`, grid size goes from
`(32, 1)` per per-token launch × 18 launches = 576 block-launches to
`(32, 18) = 576` blocks in **one** launch. Same total work, lifts
per-SM occupancy from ~4 to ~72 blocks, and removes 17 host-side
launch overheads per layer (× 36 layers = 612 saved launches per
prefill).

### Preconditions

Every K/V position in `[start_pos, start_pos + N)` must be written to
the cache **before** this kernel launches.
`transformer_prefill` enforces that by running the per-token RoPE +
KV-store loop synchronously on the same stream first, then this
kernel.

### Fallback

`fast_attention_enabled() == false` or INT8 KV cache falls back to N
sequential `flash_attention_decode` calls so the existing CPU-reference
attention is preserved for testing.

## rope — Rotary Position Embedding

**File:** `src/kernels/rope.cu`
**Time share:** ~4% of decode time.

### What it does

Applies rotary position encoding in-place to Q and K:
```
q'[2i]   = q[2i] × cos(θ) - q[2i+1] × sin(θ)
q'[2i+1] = q[2i] × sin(θ) + q[2i+1] × cos(θ)
where θ = position / (theta_base ^ (2i / head_dim))
```

### Orin tuning

```
One thread per dimension pair (both Q and K in same launch)
Total threads: (n_heads + n_kv_heads) × head_dim/2
cos/sin computed on-the-fly (cheaper than loading from table on bandwidth-limited Orin)
```

## convert — FP16↔INT8 + SwiGLU

**File:** `src/kernels/convert.cu`

### fp16_to_int8

Per-row absmax quantization for KV cache:
```
scale = max(|row|) / 127
int8_val = round(fp16_val / scale)
```

### fused_swiglu

Computes `output = silu(gate) × up` where `silu(x) = x / (1 + exp(-x))`.
One thread per element. Fusing avoids writing intermediate silu result to DRAM.

## softmax — Logit Softmax

**File:** `src/kernels/softmax.cu`

Used only for final logit→probability conversion (vocab_size elements). Three passes:
1. Find max (numerically stable)
2. Exponentiate and sum
3. Normalize

Single block, 256 threads. Vocab sizes up to 128K.

## Utility Kernels (in decode.cu)

### vec_add

`out[i] = a[i] + b[i]` — used for residual connections between attention and FFN.

### fp16_to_fp32

Converts FP16 logits to FP32 on GPU before D2H copy for sampling.

## Performance Characteristics (Orin Nano Super)

| Kernel | Bottleneck | Registers | Shared mem |
|--------|-----------|-----------|------------|
| gemv_q4/q5/q6 K | Memory bandwidth | 40 | 0 |
| gemm_quant_batched (Q4_K) | Memory bandwidth | 56 | 0 |
| gemm_quant_batched (Q5_K / Q6_K) | Memory bandwidth | 64 | 0 |
| fused_norm | Memory bandwidth | 26 | 128 bytes |
| flash_attention_decode | Memory bandwidth | 40 | (64 + head_dim) × 4 |
| flash_attention_prefill_batched | Memory bandwidth | 40 | (64 + head_dim) × 4 |
| rope | Compute (trig) | 13 | 0 |
| softmax | Memory bandwidth | 23 | ~36 bytes |
| swiglu | Memory bandwidth | 14 | 0 |
| fp16_to_int8 | Memory bandwidth | 14 | 4 bytes |

All `gemm_quant_batched` kernels ship with zero spill stores / loads
in PTXAS output.
