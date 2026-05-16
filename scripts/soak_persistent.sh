#!/bin/bash
# soak_persistent.sh — Path F + INT8 KV interop soak (issue #4).
#
# Drives jetson-llm with --conv-id across N conversation ids to exercise
# persistent-KV save/load AND LRU eviction (JLLM_KV_CACHE_MAX_MB).
#
# NOTE (alpha.12): persistent-KV format v2 does not carry INT8 scales, so
# under --int8-kv the engine SKIPS save with a warning. This script verifies
# that — under FP16 KV the cache directory grows then bounces against the
# MAX_MB cap; under INT8 KV the save is skipped and the directory stays empty.
# Path F v3 (#67) will lift that.
#
# Usage:
#   ./scripts/soak_persistent.sh model.gguf [--iters 20] [--kv int8|fp16] [--cap-mb 64]

set -u
set -o pipefail

MODEL=""
ITERS=20
KV_MODE="fp16"   # default fp16 here: that's what actually exercises save/load
CAP_MB=64

while [ $# -gt 0 ]; do
    case "$1" in
        --iters)  ITERS="$2"; shift 2 ;;
        --kv)     KV_MODE="$2"; shift 2 ;;
        --cap-mb) CAP_MB="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *)
            if [ -z "$MODEL" ]; then MODEL="$1"; shift
            else echo "Unknown arg: $1" >&2; exit 1
            fi ;;
    esac
done

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Usage: $0 <model.gguf> [--iters N] [--kv int8|fp16] [--cap-mb N]" >&2
    exit 1
fi
if [ ! -x ./build/jetson-llm ]; then
    echo "ERROR: ./build/jetson-llm not found." >&2; exit 1
fi

case "$KV_MODE" in
    int8) KV_FLAG="--int8-kv" ;;
    fp16) KV_FLAG="--fp16-kv" ;;
    *) echo "--kv must be int8 or fp16" >&2; exit 1 ;;
esac

TS=$(date +%Y%m%d-%H%M%S)
OUTDIR="soak-runs/persistent-${KV_MODE}-${TS}"
CACHE_DIR="$OUTDIR/kv-cache"
mkdir -p "$OUTDIR/stderr" "$CACHE_DIR"

export JLLM_KV_CACHE_DIR="$CACHE_DIR"
export JLLM_KV_CACHE_MAX_MB="$CAP_MB"

{
    echo "═══════════════════════════════════════════════════"
    echo "  jetson-llm persistent-KV soak — issue #4 / Path F"
    echo "═══════════════════════════════════════════════════"
    echo "  Model:       $MODEL"
    echo "  Iters:       $ITERS  (cycling 5 conv ids)"
    echo "  KV mode:     $KV_MODE  ($KV_FLAG)"
    echo "  Cache dir:   $CACHE_DIR"
    echo "  Cap (MB):    $CAP_MB"
    echo "  Git:         $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
    echo "  Started:     $(date)"
    echo
} | tee "$OUTDIR/banner.txt"

CSV="$OUTDIR/stats.csv"
echo "iter,conv_id,exit_code,ttft_ms,decode_tok_s,cache_files,cache_mb" > "$CSV"

# 5 ids → with 20 iters each id is hit 4× (cold once, then 3 warm turns).
CONVS=("alpha" "bravo" "charlie" "delta" "echo")

# 5 short user-style prompts (different topics per id, rotating to grow KV).
USER_PROMPTS=(
    "I'm planning a road trip from Boston to Asheville next month."
    "Tell me about the Hubble Space Telescope's most important discoveries."
    "I'm trying to decide between learning Rust or Zig. Help me think it through."
    "What's a quick weeknight dinner I can make with chicken thighs and rice?"
    "Walk me through how a transformer's attention head actually works."
)

./build/jetson-llm -m "$MODEL" -p "Hello" -n 8 $KV_FLAG >/dev/null 2>&1 || true

for ((i=1; i<=ITERS; i++)); do
    cid=${CONVS[$(( (i-1) % ${#CONVS[@]} ))]}
    upr=${USER_PROMPTS[$(( (i-1) % ${#USER_PROMPTS[@]} ))]}" Round $i."
    log="$OUTDIR/stderr/iter-$(printf '%02d' $i).log"

    ./build/jetson-llm -m "$MODEL" -p "$upr" -n 128 $KV_FLAG --conv-id "$cid" \
        > /dev/null 2> "$log"
    rc=$?

    ttft=$(awk '/^TTFT:/{print $2; exit}' "$log")
    dec=$(awk '/^Decode:/{for(i=1;i<=NF;i++) if($i=="tokens,"){print $(i+1); exit}}' "$log")
    [ -z "$ttft" ] && ttft=0; [ -z "$dec" ] && dec=0

    nfiles=$(find "$CACHE_DIR" -maxdepth 1 -type f -name '*.bin' 2>/dev/null | wc -l | tr -d ' ')
    mb=$(du -sm "$CACHE_DIR" 2>/dev/null | awk '{print $1}')

    echo "$i,$cid,$rc,$ttft,$dec,$nfiles,$mb" >> "$CSV"
    printf "iter %2d  conv=%-8s  rc=%d  ttft=%5s ms  dec=%s tok/s  cache=%s files / %s MB\n" \
        "$i" "$cid" "$rc" "$ttft" "$dec" "$nfiles" "$mb"
done

echo
echo "─── Final cache state ─────────────────────────────────"
ls -lh "$CACHE_DIR" 2>/dev/null || true
echo
echo "─── Save-skipped warnings (expected for --int8-kv) ────"
grep -h "save skipped\|skip" "$OUTDIR"/stderr/*.log | sort -u || true
echo
echo "─── Acceptance ────────────────────────────────────────"
if [ "$KV_MODE" = "fp16" ]; then
    echo "  FP16: expect cache_mb to grow and stay <= ${CAP_MB} MB"
    final_mb=$(du -sm "$CACHE_DIR" | awk '{print $1}')
    if [ "$final_mb" -le "$CAP_MB" ]; then
        echo "  PASS: final cache=${final_mb} MB <= cap=${CAP_MB} MB"
    else
        echo "  FAIL: final cache=${final_mb} MB exceeds cap=${CAP_MB} MB"
    fi
else
    echo "  INT8: expect cache to stay empty (save skipped on alpha.12)"
    nf=$(find "$CACHE_DIR" -maxdepth 1 -type f -name '*.bin' | wc -l | tr -d ' ')
    if [ "$nf" -eq 0 ]; then
        echo "  PASS: no .bin files written under --int8-kv (waiting on #67)"
    else
        echo "  FAIL: $nf .bin files written despite --int8-kv"
    fi
fi
