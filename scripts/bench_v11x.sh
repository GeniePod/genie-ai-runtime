#!/bin/bash
# bench_v11x.sh — Benchmark and profile genie-ai-runtime v1.1.x optimizations
#
# Tests each perf/rope-precomputed-table iteration:
#   iter1: RoPE precomputed table      (JLLM_ROPE_TABLE=0 disables)
#   iter2: Warp-parallel decode attn   (JLLM_FAST_ATTN=0 disables)
#   iter3+4: CUDA graph decode         (JLLM_DECODE_GRAPH=0 disables)
#   iter5: dp4a triple GEMV            (JLLM_FAST_GEMV=0 disables all GPU GEMV)
#   iter6: half2 elementwise kernels   (always-on; ablated via rebuilt binary)
#
# Usage:
#   ./scripts/bench_v11x.sh <model.gguf> [options]
#
# Options:
#   --tokens N      tokens to generate (default 64)
#   --profile       emit JLLM_PROFILE=1 per-token breakdown for each run
#   --nsys          run Nsight Systems profile on the default config
#   --decode-only   skip warmup prefill report (not used, for future)
#
# Example:
#   ./scripts/bench_v11x.sh ~/Downloads/gemma-4-E2B-it-Q4_K_M.gguf --profile
#   ./scripts/bench_v11x.sh ~/Qwen3-4B-Q4_K_M.gguf --nsys
#
# Outputs in bench_v11x_<timestamp>/:
#   summary.txt       human-readable speedup table
#   results.tsv       machine-readable per-run table
#   <label>.log       full stderr per run
#   profile_<n>.log   JLLM_PROFILE=1 output per run (--profile only)

set -u

MODEL=""
DO_PROFILE=0
DO_NSYS=0
TOKENS=64
PROMPT="Explain the key differences between CPU and GPU architecture."

while [ $# -gt 0 ]; do
    case "$1" in
        --profile)     DO_PROFILE=1; shift ;;
        --nsys)        DO_NSYS=1; shift ;;
        --tokens)      TOKENS="$2"; shift 2 ;;
        --*)           echo "Unknown flag: $1"; exit 1 ;;
        *)             MODEL="$1"; shift ;;
    esac
done

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Usage: $0 <model.gguf> [--profile] [--nsys] [--tokens N]"
    exit 1
fi

BIN="./build/jetson-llm"
if [ ! -x "$BIN" ]; then
    echo "Binary not found at $BIN — build first:"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
    exit 1
fi

OUTDIR="bench_v11x_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
TSV="$OUTDIR/results.tsv"

echo -e "label\tprefill_tps\tdecode_tps\tlayers_ms_avg\tlogits_ms_avg\tnotes" > "$TSV"

# ── helpers ──────────────────────────────────────────────────────────────────

sysinfo() {
    printf "══════════════════════════════════════════════════════\n"
    printf "  bench_v11x — genie-ai-runtime v1.1.x perf profile\n"
    printf "  Model : %s\n" "$(basename "$MODEL") ($(du -sh "$MODEL" | cut -f1))"
    printf "  Binary: %s\n" "$(git -C . rev-parse --short HEAD 2>/dev/null || echo n/a) $(git -C . rev-parse --abbrev-ref HEAD 2>/dev/null)"
    printf "  Date  : %s\n" "$(date)"
    printf "  Prompt: \"%s\" (%d decode tokens)\n" "$PROMPT" "$TOKENS"
    printf "──────────────────────────────────────────────────────\n"
    printf "  Power : %s\n" "$(sudo nvpmodel -q 2>/dev/null | head -1 || echo unknown)"
    local freq
    freq=$(cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq 2>/dev/null \
           || cat /sys/class/devfreq/*/cur_freq 2>/dev/null | head -1 || echo 0)
    printf "  GPU   : %s MHz\n" "$(awk "BEGIN{printf \"%.0f\", $freq/1000000}")"
    printf "  RAM   : %s MB free\n" "$(free -m | awk '/Mem:/ {print $7}')"
    printf "  Temp  : %s\n" "$(cat /sys/devices/virtual/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf "%.1f°C", $1/1000}' || echo unknown)"
    printf "══════════════════════════════════════════════════════\n"
}

# run_bench <label> <extra_env>
# Runs the binary, parses prefill/decode tok/s and optional profile lines.
run_bench() {
    local label="$1"
    local env="$2"
    local logfile="$OUTDIR/${label}.log"

    printf "  %-42s " "$label"

    local profile_flag=""
    [ "$DO_PROFILE" = "1" ] && profile_flag="JLLM_PROFILE=1"

    eval "env $profile_flag $env $BIN -m '$MODEL' -p '$PROMPT' -n $TOKENS" \
        > /dev/null 2>"$logfile" || true

    # Prefill
    local ptps
    ptps=$(grep -m1 "prefill" "$logfile" 2>/dev/null \
           | grep -oP '[\d.]+(?= tok/s)' | head -1 || echo 0)

    # Decode
    local dtps
    dtps=$(grep -m1 "\bdecode\b.*tok/s" "$logfile" 2>/dev/null \
           | grep -oP '[\d.]+(?= tok/s)' | head -1 || echo 0)

    # CUDA graph active?
    local graph_tag=""
    grep -q "Capturing decode CUDA graph" "$logfile" 2>/dev/null && graph_tag=" [graph]"

    # JLLM_PROFILE averages (emitted every token)
    local avg_layers avg_logits
    avg_layers=$(grep -oP 'avg_layers=\K[\d.]+' "$logfile" 2>/dev/null | tail -1 || echo "—")
    avg_logits=$(grep -oP 'avg_logits=\K[\d.]+' "$logfile" 2>/dev/null | tail -1 || echo "—")

    if [ "$DO_PROFILE" = "1" ] && [ "$avg_layers" != "—" ]; then
        printf "prefill=%-6s  decode=%-6s  layers=%-7s logits=%-7s%s\n" \
               "$ptps" "$dtps" "${avg_layers}ms" "${avg_logits}ms" "$graph_tag"
    else
        printf "prefill=%-6s  decode=%-6s%s\n" "$ptps" "$dtps" "$graph_tag"
    fi

    echo -e "${label}\t${ptps}\t${dtps}\t${avg_layers}\t${avg_logits}\t${env}" >> "$TSV"
}

# ── main ─────────────────────────────────────────────────────────────────────

{
sysinfo

echo ""
echo "Warmup run (2 tokens)..."
$BIN -m "$MODEL" -p "Hello" -n 2 > /dev/null 2>&1 || true

echo ""
echo "── Full baseline (all GPU paths OFF) ─────────────────"
ALL_OFF="JLLM_ROPE_TABLE=0 JLLM_FAST_ATTN=0 JLLM_DECODE_GRAPH=0 JLLM_FAST_GEMV=0 JLLM_FAST_NORM=0"
run_bench "baseline_all_off"         "$ALL_OFF"

echo ""
echo "── Cumulative enablement (one optimization at a time) ─"
run_bench "01_rope_table"            "JLLM_FAST_ATTN=0 JLLM_DECODE_GRAPH=0 JLLM_FAST_GEMV=0 JLLM_FAST_NORM=0"
run_bench "01+02_attn"               "JLLM_DECODE_GRAPH=0 JLLM_FAST_GEMV=0 JLLM_FAST_NORM=0"
run_bench "01+02+03_graph"           "JLLM_FAST_GEMV=0 JLLM_FAST_NORM=0"
run_bench "01+02+03+05_gemv_dp4a"    "JLLM_FAST_NORM=0"
run_bench "01+02+03+05+06_all_on"    ""

echo ""
echo "── Single-feature ablations (all ON, one disabled) ───"
run_bench "ablate_rope_table"        "JLLM_ROPE_TABLE=0"
run_bench "ablate_fast_attn"         "JLLM_FAST_ATTN=0"
run_bench "ablate_graph"             "JLLM_DECODE_GRAPH=0"
run_bench "ablate_fast_gemv"         "JLLM_FAST_GEMV=0"
run_bench "ablate_fast_norm"         "JLLM_FAST_NORM=0"

} 2>&1 | tee "$OUTDIR/summary.txt"

# ── Speedup table ─────────────────────────────────────────────────────────────
{
echo ""
echo "── Decode tok/s — progression table ──────────────────"
printf "\n%-35s  %10s  %10s\n" "Configuration" "decode tps" "vs baseline"
printf "%-35s  %10s  %10s\n" "───────────────────────────────────" "──────────" "──────────"

BASE_TPS=$(awk -F'\t' 'NR==2 {print $3}' "$TSV")

tail -n +2 "$TSV" | while IFS=$'\t' read -r label ptps dtps layers logits notes; do
    if [ -n "$BASE_TPS" ] && [ "$BASE_TPS" != "0" ] && \
       [ -n "$dtps" ] && [ "$dtps" != "0" ]; then
        delta=$(awk "BEGIN{printf \"%+.1f%%\", ($dtps/$BASE_TPS - 1)*100}")
    else
        delta="—"
    fi
    printf "%-35s  %10s  %10s\n" "$label" "$dtps" "$delta"
done
} 2>&1 | tee -a "$OUTDIR/summary.txt"

# ── Nsight Systems ────────────────────────────────────────────────────────────
if [ "$DO_NSYS" = "1" ]; then
{
echo ""
echo "── nsys kernel profile (all optimizations ON) ─────────"
if command -v nsys > /dev/null 2>&1; then
    NSYS_OUT="$OUTDIR/profile_allon"
    echo "  Capturing: ${NSYS_OUT}.nsys-rep ..."
    nsys profile \
        --trace=cuda,nvtx \
        --output="$NSYS_OUT" \
        --force-overwrite=true \
        "$BIN" -m "$MODEL" -p "$PROMPT" -n 32 > /dev/null 2>"$OUTDIR/nsys_allon.log"
    echo ""
    echo "  Top CUDA kernels by total time:"
    nsys stats --report cuda_gpu_kern_sum "${NSYS_OUT}.nsys-rep" 2>/dev/null \
        | grep -v "^$" | head -30
    echo ""
    echo "  To view interactively: nsys-ui ${NSYS_OUT}.nsys-rep"

    if [ "$DO_PROFILE" = "1" ]; then
        echo ""
        echo "  nsys kernel profile (all optimizations OFF) ─────────"
        NSYS_OFF="$OUTDIR/profile_alloff"
        env JLLM_ROPE_TABLE=0 JLLM_FAST_ATTN=0 JLLM_DECODE_GRAPH=0 \
            JLLM_FAST_GEMV=0 JLLM_FAST_NORM=0 \
            nsys profile \
                --trace=cuda,nvtx \
                --output="$NSYS_OFF" \
                --force-overwrite=true \
                "$BIN" -m "$MODEL" -p "$PROMPT" -n 32 > /dev/null 2>"$OUTDIR/nsys_alloff.log"
        echo ""
        echo "  Top CUDA kernels (baseline):"
        nsys stats --report cuda_gpu_kern_sum "${NSYS_OFF}.nsys-rep" 2>/dev/null \
            | grep -v "^$" | head -30
    fi
else
    echo "  nsys not in PATH — install Nsight Systems or add it to PATH"
    echo "  Typically: export PATH=\$PATH:/opt/nvidia/nsight-systems/*/bin"
fi
} 2>&1 | tee -a "$OUTDIR/summary.txt"
fi

echo ""
echo "Results saved to: $OUTDIR/"
echo "  summary.txt  — human-readable table"
echo "  results.tsv  — machine-readable (label / prefill tps / decode tps / ...)"
echo "  *.log        — full stderr per configuration"
