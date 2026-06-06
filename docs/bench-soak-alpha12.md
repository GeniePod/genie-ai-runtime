# Long-running stability soak — alpha.12

Issue [#4](https://github.com/GeniePod/genie-ai-runtime/issues/4) acceptance run.

> Template — fill in once the soak completes on the Jetson. Drop the auto-generated
> `summary.txt` block into the "Results" section verbatim.

## Setup

| Field           | Value |
| --------------- | --- |
| Model           | Qwen3-4B Q4_K_M (`models/Qwen3-4B-Q4_K_M.gguf`) |
| KV mode         | INT8 (alpha.12 default) — also one FP16 control |
| Power           | `nvpmodel -m 1` (25 W MAXN SUPER) |
| Iterations      | 100 |
| Tokens per iter | 1024 |
| Prompt rotation | 10 prompts (factual / creative / code, cycled) |
| Harness         | `scripts/soak.sh` (this branch) |
| Git             | `<commit sha>` |
| Date            | `<YYYY-MM-DD>` |

## Acceptance bar

- [ ] 100/100 iterations complete (no SIGSEGV, no engine-returns-empty)
- [ ] RSS slope across iters 11..100 < +1 MB/iter
- [ ] FD delta between iter 1 and iter 100 ≤ +2
- [ ] p95 SoC temp < 75 °C (max ≤ 80 °C transient)
- [ ] Median decode tok/s within 10 % of single-shot alpha.12 (9.9 tok/s)
- [ ] 10-sample coherence spot-check passes (on-topic, no token loop)

## Results — INT8 default run

```
<paste $OUTDIR/summary.txt here after the run completes>
```

### Coherence spot-check (10 random iters)

| iter | prompt category | verdict | note |
| ---- | --------------- | ------- | ---- |
|      |                 |         |      |

## Results — FP16 control run (optional)

Same harness, `--fp16-kv`, 100 iters. Used to confirm INT8 didn't regress
stability vs the previous default.

```
<paste second summary.txt here>
```

## Path F + INT8 interop check

Run `scripts/soak_persistent.sh` twice — once `--kv fp16`, once `--kv int8` —
to confirm persistent-KV behavior across the alpha.12 carve-out.

| Mode | Iters | Final cache (MB) | Files written | Expected | Result |
| ---- | ----- | ---------------- | ------------- | -------- | ------ |
| fp16 | 20    |                  |               | grows, ≤ cap | |
| int8 | 20    | 0                | 0             | save skipped (waiting on #67) | |

## Notes

Anomalies, surprises, or anything the next reader needs to know.

## Sign-off

- [ ] All acceptance items above are PASS, or each FAIL has a tracked follow-up issue.
- [ ] Soak run artifacts (`stats.csv`, banner, samples) archived under `soak-runs/`.
- [ ] Issue #4 closed referencing this doc.
