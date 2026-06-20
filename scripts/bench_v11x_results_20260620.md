# bench_v11x results — perf/rope-precomputed-table @ f48a2a5

**Device:** Jetson Orin Nano 8GB, MAXN_SUPER, L4T 36.4 / CUDA 12.6
**Model:** gemma-4-E2B-it-Q4_K_M.gguf
**Prompt:** "Explain the key differences between CPU and GPU architecture." (11 prompt tokens)
**Date:** 2026-06-20

## Decode tok/s — ablation table (16-token run)

| Config | decode tok/s | layers ms | notes |
|---|---|---|---|
| all features ON | 32.5 | 27.5 | full stack (stable 64-tok run = **30.9**) |
| ablate rope_table | 32.8 | 27.4 | rope table: **negligible** for decode |
| ablate fast_attn | 19.4 | 49.6 | warp-parallel decode attn: **-40%** if off |
| ablate graph | 32.5 | 27.5 | CUDA graph: **0%** (never captured on E2B) |
| ablate fast_gemv | 0.2 | 5417 | dp4a GPU GEMV: **ESSENTIAL** (CPU fallback) |
| ablate fast_norm | 21.1 | 45.3 | half2 fused RMSNorm: **-35%** if off |

## Cumulative enablement (off → on)

| Config | decode tok/s |
|---|---|
| baseline (all GPU off, CPU GEMV) | 0.2 |
| + GPU GEMV (dp4a) | 21.0 |
| + fast_norm (= all on) | 32.5 |

## Key findings

1. **fast_gemv (dp4a GPU GEMV) is the critical enabler** — without it decode
   falls back to CPU at 0.2 tok/s (5400 ms/token in the layer loop).
2. **fast_attn (warp-parallel decode attention, iter2) ≈ +68%** marginal
   (19.4 → 32.5): the largest of the per-feature decode levers.
3. **fast_norm (half2 fused RMSNorm, iter6) ≈ +54%** marginal (21.1 → 32.5).
4. **rope_table (iter1): negligible for decode** — RoPE trig is a tiny part of
   a memory-bound decode step; the precomputed table doesn't move the needle.
5. **CUDA graph (iter3/iter4): 0% on gemma-4-E2B.** The Gemma4 graph path
   (iter4) is correctly *never captured* for E2B because E2B uses 20 shared-KV
   layers; the guard logs `decode-graph: Gemma4 shared-KV not supported in
   graph; staying on per-step path`. iter4 only benefits a hypothetical
   non-shared-KV Gemma4 variant.

## Notes / caveats

- Prefill tok/s in this bench is **not meaningful** — the 11-token prompt is
  dominated by fixed overhead. Real prefill throughput is measured on long
  prompts (prior work: ~333 tok/s on a 1261-token prompt on main).
- Clocks were MAXN_SUPER but **not** jetson_clocks-locked; small run-to-run
  variance applies. The within-run ablation deltas are robust; absolute decode
  is ~31 tok/s steady-state.
- Minor cleanup: the `decode-graph: Gemma4 shared-KV...` line prints **every
  token** — should be print-once.
