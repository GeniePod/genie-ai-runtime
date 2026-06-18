#!/bin/bash
# soak.sh — Long-running stability soak for jetson-llm (issue #4).
#
# Runs N iterations of large-generation (default 100 × 1024 tokens) over a
# rotating set of prompts and captures per-iteration CSV: peak RSS, FD count,
# SoC temp, decode tok/s, OOM/thermal events, and the generated text length.
#
# Usage:
#   ./scripts/soak.sh model.gguf [--iters 100] [--tokens 1024] [--label alpha12-int8]
#
# Outputs (in ./soak-runs/<label>-<timestamp>/):
#   stats.csv      one row per iteration
#   summary.txt    min/median/p95/max of each metric + pass/fail vs acceptance
#   gen/iter-NN.txt  generated text per iteration (for coherence spot-check)
#   stderr/iter-NN.log  per-iteration jetson-llm stderr
#
# Acceptance bar (from issue #4 rescoped 2026-05-16):
#   - 100/100 iterations complete, no SIGSEGV
#   - RSS slope over iters 11..N < +1 MB/iter (warmup excluded)
#   - FD delta <= +2 between iter 1 and iter N
#   - Peak SoC temp < 75 °C sustained
#   - Median decode tok/s within 10 % of single-shot baseline

set -u
set -o pipefail

# ── Args ────────────────────────────────────────────────────────────────────
MODEL=""
ITERS=100
TOKENS=1024
LABEL="alpha12-int8"
KV_FLAG="--int8-kv"   # alpha.12 default; pass --fp16-kv on cmdline to flip

while [ $# -gt 0 ]; do
    case "$1" in
        --iters)   ITERS="$2";  shift 2 ;;
        --tokens)  TOKENS="$2"; shift 2 ;;
        --label)   LABEL="$2";  shift 2 ;;
        --fp16-kv) KV_FLAG="--fp16-kv"; shift ;;
        --int8-kv) KV_FLAG="--int8-kv"; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            if [ -z "$MODEL" ]; then MODEL="$1"; shift
            else echo "Unknown arg: $1" >&2; exit 1
            fi ;;
    esac
done

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Usage: $0 <model.gguf> [--iters N] [--tokens N] [--label NAME] [--fp16-kv]" >&2
    exit 1
fi

if [ ! -x ./build/jetson-llm ]; then
    echo "ERROR: ./build/jetson-llm not found. Build first." >&2
    exit 1
fi

# ── Output dir ──────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d-%H%M%S)
OUTDIR="soak-runs/${LABEL}-${TS}"
mkdir -p "$OUTDIR/gen" "$OUTDIR/stderr"
CSV="$OUTDIR/stats.csv"
SUMMARY="$OUTDIR/summary.txt"

# ── Prompts: factual / creative / code rotation (10 templates) ─────────────
PROMPTS=(
    "Explain how a CPU cache hierarchy works, from L1 to main memory."
    "Write a short story about a lighthouse keeper who befriends a whale."
    "Implement a Python function that returns the nth Fibonacci number using memoization. Include a brief docstring."
    "Compare and contrast the philosophies of stoicism and existentialism."
    "Describe the steps to deploy a containerized web service to a Kubernetes cluster."
    "Compose a poem in iambic pentameter about the sound of rain on a tin roof."
    "Write a SQL query that joins three tables (users, orders, products) and returns the top 5 products by total revenue in 2024."
    "Explain the basics of how a CRISPR-Cas9 gene edit works at the molecular level."
    "Outline a 30-minute beginner workout routine focused on core strength, with no equipment."
    "Write a C function that safely concatenates two null-terminated strings into a caller-provided buffer, returning the number of bytes written."
)
NPROMPTS=${#PROMPTS[@]}

# ── CSV header ──────────────────────────────────────────────────────────────
echo "iter,prompt_idx,wall_s,prompt_tokens,completion_tokens,prompt_tok_s,decode_tok_s,ttft_ms,peak_mb,peak_c,oom_stops,thermal_pauses,rss_max_kb,fd_max,exit_code,gen_chars" > "$CSV"

# ── Banner ──────────────────────────────────────────────────────────────────
{
    echo "═══════════════════════════════════════════════════"
    echo "  jetson-llm soak — issue #4 (long-running stability)"
    echo "═══════════════════════════════════════════════════"
    echo "  Model:   $MODEL ($(ls -lh "$MODEL" | awk '{print $5}'))"
    echo "  Iters:   $ITERS"
    echo "  Tokens:  $TOKENS"
    echo "  KV:      $KV_FLAG"
    echo "  Label:   $LABEL"
    echo "  Output:  $OUTDIR"
    echo "  Power:   $(sudo nvpmodel -q 2>/dev/null | head -1 || echo 'unknown')"
    echo "  Git:     $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
    echo "  Started: $(date)"
    echo
} | tee "$OUTDIR/banner.txt"

# ── Sampler: poll RSS / FD / temp while a PID is alive ─────────────────────
# Writes one line per sample to $SAMPLE_FILE: timestamp rss_kb fd_count temp_c
sample_loop() {
    local pid="$1" out="$2"
    while kill -0 "$pid" 2>/dev/null; do
        local rss fd temp
        rss=$(awk '/VmRSS/{print $2}' /proc/"$pid"/status 2>/dev/null || echo 0)
        fd=$(ls /proc/"$pid"/fd 2>/dev/null | wc -l)
        temp=$(awk '{printf "%.1f", $1/1000}' /sys/devices/virtual/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
        echo "$(date +%s) $rss $fd $temp" >> "$out"
        sleep 1
    done
}

# ── Stats parser: extract numbers from jetson-llm stderr ───────────────────
# Lines we care about (from src/main.cpp single-prompt branch):
#   "Prompt:  N tokens, F tok/s (F ms)"
#   "Decode:  N tokens, F tok/s (F ms)"
#   "TTFT:    F ms ..."
#   "Memory:  peak N MB"
#   "Thermal: peak F°C"
#   "WARNING: OOM guard stopped generation N time(s)"
parse_stats() {
    local log="$1"
    awk '
      /^Prompt:/  { for (i=1;i<=NF;i++) if ($i=="tokens,") { pt=$(i-1); pts=$(i+1); } }
      /^Decode:/  { for (i=1;i<=NF;i++) if ($i=="tokens,") { ct=$(i-1); cts=$(i+1); } }
      /^TTFT:/    { ttft=$2 }
      /^Memory:/  { mem=$3 }
      /^Thermal:/ { sub(/°C/,"",$3); thrm=$3 }
      /OOM guard/ { oom=$6 }
      /thermal pause/ { thp+=1 }
      END {
        if (pt=="")  pt=0;  if (ct=="")  ct=0
        if (pts=="") pts=0; if (cts=="") cts=0
        if (ttft=="") ttft=0
        if (mem=="")  mem=0; if (thrm=="") thrm=0
        if (oom=="")  oom=0; if (thp=="") thp=0
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n", pt, ct, pts, cts, ttft, mem, thrm, oom, thp
      }
    ' "$log"
}

# ── One warmup iter (not recorded) so first-prompt OS pagecache fills ──────
echo "▸ Warmup..."
./build/jetson-llm -m "$MODEL" -p "Hello" -n 16 $KV_FLAG >/dev/null 2>&1 || true

# ── Main loop ──────────────────────────────────────────────────────────────
for ((i=1; i<=ITERS; i++)); do
    pidx=$(( (i-1) % NPROMPTS ))
    prompt="${PROMPTS[$pidx]}"
    log="$OUTDIR/stderr/iter-$(printf '%03d' $i).log"
    gen="$OUTDIR/gen/iter-$(printf '%03d' $i).txt"
    samp="$OUTDIR/stderr/iter-$(printf '%03d' $i).samples"

    t0=$(date +%s)
    ./build/jetson-llm -m "$MODEL" -p "$prompt" -n "$TOKENS" $KV_FLAG \
        > "$gen" 2> "$log" &
    pid=$!
    sample_loop "$pid" "$samp" &
    sloop=$!
    wait "$pid"; rc=$?
    wait "$sloop" 2>/dev/null
    t1=$(date +%s)
    wall=$((t1 - t0))

    # Per-iter aggregates
    rss_max=$(awk 'BEGIN{m=0}{if($2>m)m=$2}END{print m+0}' "$samp" 2>/dev/null)
    fd_max=$(awk 'BEGIN{m=0}{if($3>m)m=$3}END{print m+0}' "$samp" 2>/dev/null)
    stats=$(parse_stats "$log")
    gen_chars=$(wc -c < "$gen" | tr -d ' ')

    echo "$i,$pidx,$wall,$stats,$rss_max,$fd_max,$rc,$gen_chars" >> "$CSV"

    # Live one-liner
    printf "iter %3d/%d  pidx=%d  wall=%4ds  rc=%d  rss_kb=%-8s fd=%-3s  | %s\n" \
        "$i" "$ITERS" "$pidx" "$wall" "$rc" "$rss_max" "$fd_max" \
        "$(echo "$stats" | awk -F, '{printf "decode=%s tok/s peak=%sMB %s°C oom=%s", $4, $6, $7, $8}')"

    if [ "$rc" -ne 0 ]; then
        echo "  !! non-zero exit on iter $i — see $log" >&2
    fi
done

# ── Summary ────────────────────────────────────────────────────────────────
python3 - "$CSV" "$ITERS" > "$SUMMARY" <<'PY'
import csv, sys, statistics as st

path, iters = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(open(path)))
n_ok = sum(1 for r in rows if r['exit_code'] == '0')

def col(name, cast=float):
    out = []
    for r in rows:
        try:
            out.append(cast(r[name]))
        except (ValueError, KeyError):
            pass
    return out

def quants(xs):
    if not xs: return (0, 0, 0, 0)
    xs = sorted(xs)
    p95 = xs[max(0, int(0.95 * len(xs)) - 1)]
    return (xs[0], st.median(xs), p95, xs[-1])

def line(name, xs, unit=""):
    mn, md, p95, mx = quants(xs)
    print(f"  {name:22s}  min={mn:8.2f}  med={md:8.2f}  p95={p95:8.2f}  max={mx:8.2f} {unit}")

print(f"Iterations:        {n_ok}/{iters} completed (exit_code == 0)")
print()
line("decode tok/s",       col("decode_tok_s"))
line("prompt tok/s",       col("prompt_tok_s"))
line("TTFT (ms)",          col("ttft_ms"),       "ms")
line("peak GPU mem (MB)",  col("peak_mb"),       "MB")
line("peak SoC temp (°C)", col("peak_c"),        "°C")
line("RSS max (KB)",       col("rss_max_kb"),    "KB")
line("FD max",             col("fd_max"))
line("wall (s)",           col("wall_s"),        "s")
line("gen chars",          col("gen_chars"),     "chars")
print()

# RSS slope across iters 11..N (warmup excluded)
warmup = 10
ys = [(int(r['iter']), float(r['rss_max_kb'])) for r in rows
      if r['rss_max_kb'] not in ('', '0') and int(r['iter']) > warmup]
if len(ys) >= 5:
    xs = [p[0] for p in ys]; vs = [p[1]/1024.0 for p in ys]  # MB
    mx, my = sum(xs)/len(xs), sum(vs)/len(vs)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,vs))
    den = sum((x-mx)**2 for x in xs) or 1
    slope = num/den
    print(f"RSS slope (iters {warmup+1}..{len(rows)}): {slope:+.3f} MB/iter")
else:
    slope = 0
    print("RSS slope: not enough data")

# FD delta iter 1 vs iter N
if len(rows) >= 2:
    fd1 = int(rows[0]['fd_max'] or 0)
    fdN = int(rows[-1]['fd_max'] or 0)
    print(f"FD delta (iter 1 -> {len(rows)}): {fdN - fd1:+d}  ({fd1} -> {fdN})")
else:
    fdN = fd1 = 0

# OOM / thermal pause totals
oom_total = sum(int(r['oom_stops'] or 0) for r in rows)
thp_total = sum(int(r['thermal_pauses'] or 0) for r in rows)
print(f"OOM guard hits:    {oom_total}")
print(f"Thermal pauses:    {thp_total}")
print()

# Pass/fail vs acceptance
print("─── Acceptance ─────────────────────────────────────")
ok_iters   = n_ok == iters
ok_slope   = abs(slope) < 1.0
ok_fd      = (fdN - fd1) <= 2
peak_t     = max(col("peak_c") or [0])
ok_temp_p95 = quants(col("peak_c"))[2] < 75.0
def chk(b): return "PASS" if b else "FAIL"
print(f"  100/100 iters       : {chk(ok_iters)}  ({n_ok}/{iters})")
print(f"  RSS slope < 1 MB/it : {chk(ok_slope)}  ({slope:+.3f})")
print(f"  FD delta <= +2      : {chk(ok_fd)}     ({fdN - fd1:+d})")
print(f"  p95 SoC temp < 75°C : {chk(ok_temp_p95)}  (p95={quants(col('peak_c'))[2]:.1f}°C, max={peak_t:.1f}°C)")
PY

echo
echo "═══════════════════════════════════════════════════"
echo "  Soak complete — see $SUMMARY"
echo "═══════════════════════════════════════════════════"
cat "$SUMMARY"
