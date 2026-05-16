#!/bin/bash
# bench_load.sh — cold vs warm model-load timing (issue #4 follow-up).
#
# Drops the OS page cache, runs jetson-llm with no prompt (load + unload only),
# then runs it again warm. Reports load_ms cold/warm/delta and effective MB/s.
#
# Requires sudo for /proc/sys/vm/drop_caches.
#
# Usage:
#   ./scripts/bench_load.sh <model.gguf>
#   ./scripts/bench_load.sh /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf

set -u

MODEL="${1:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Usage: $0 <model.gguf>" >&2
    exit 1
fi
if [ ! -x ./build/jetson-llm ]; then
    echo "ERROR: ./build/jetson-llm not found. Build first." >&2
    exit 1
fi

SIZE_MB=$(( $(stat -c%s "$MODEL") / 1024 / 1024 ))

echo "═══════════════════════════════════════════════════"
echo "  jetson-llm load-time bench"
echo "═══════════════════════════════════════════════════"
echo "  Model:   $MODEL ($SIZE_MB MB)"
echo "  Power:   $(sudo nvpmodel -q 2>/dev/null | head -1 || echo 'unknown')"
echo "  Git:     $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo

# Parse "[engine] Model loaded in 1234 ms (567 MB/s)" — field 4 is ms, field 6 has "(NNN".
run_one() {
    local label="$1"
    local out load_ms mbs
    out=$(./build/jetson-llm -m "$MODEL" 2>&1 || true)
    load_ms=$(echo "$out" | awk '/\[engine\] Model loaded in/{print $5; exit}')
    mbs=$(echo "$out" | awk '/\[engine\] Model loaded in/{gsub(/[()]/,""); print $7; exit}')
    if [ -z "$load_ms" ]; then load_ms="?"; fi
    if [ -z "$mbs" ];     then mbs="?";     fi
    printf "  %-6s  load_ms=%-7s  throughput=%s MB/s\n" "$label" "$load_ms" "$mbs" >&2
    echo "$load_ms"
}

sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
cold_ms=$(run_one "COLD")

# Warm: pagecache populated by the cold run above.
warm_ms=$(run_one "WARM")

echo
if [[ "$cold_ms" =~ ^[0-9]+$ ]] && [[ "$warm_ms" =~ ^[0-9]+$ ]]; then
    delta=$((cold_ms - warm_ms))
    ratio=$(awk -v c="$cold_ms" -v w="$warm_ms" 'BEGIN{if(w>0) printf "%.1fx", c/w; else print "n/a"}')
    echo "Cold − Warm: ${delta} ms  (cold is ${ratio} slower)"
fi
