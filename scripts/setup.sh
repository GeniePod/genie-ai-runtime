#!/bin/bash
# setup.sh — install jetson-llm-server as a systemd service (issue #5 phase 4).
#
# Copies:
#   build/jetson-llm-server                       -> /opt/jetson-llm/bin/
#   deploy/systemd/jetson-llm-server.service      -> /etc/systemd/system/
#   (templated env file, kept if already present) -> /etc/jetson-llm/server.env
#
# Does NOT enable or start the service — that's an explicit follow-on
# step so you can edit /etc/jetson-llm/server.env first if needed.
#
# Usage:
#   sudo ./scripts/setup.sh
#   sudo ./scripts/setup.sh --user pat --model /opt/models/qwen3-4b.gguf --port 8080
#
# After:
#   sudo systemctl enable --now jetson-llm-server
#   sudo journalctl -u jetson-llm-server -f

set -eu

USER_NAME="${SUDO_USER:-$(id -un)}"
MODEL_PATH=""
PORT=8080

while [ $# -gt 0 ]; do
    case "$1" in
        --user)    USER_NAME="$2"; shift 2 ;;
        --model)   MODEL_PATH="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: must run as root (use sudo)." >&2
    exit 1
fi

# Resolve paths relative to the script location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN_SRC="$REPO_ROOT/build/jetson-llm-server"
UNIT_SRC="$REPO_ROOT/deploy/systemd/jetson-llm-server.service"

if [ ! -x "$BIN_SRC" ]; then
    echo "Error: $BIN_SRC not found or not executable." >&2
    echo "Build first:" >&2
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DJLLM_BUILD_SERVER=ON" >&2
    echo "  cmake --build build -j\$(nproc)" >&2
    exit 1
fi

if [ ! -r "$UNIT_SRC" ]; then
    echo "Error: $UNIT_SRC missing." >&2
    exit 1
fi

# Default model path: a working Qwen3-4B if it exists, otherwise leave
# blank and let the operator fill in /etc/jetson-llm/server.env.
if [ -z "$MODEL_PATH" ]; then
    for cand in /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf; do
        if [ -r "$cand" ]; then MODEL_PATH="$cand"; break; fi
    done
fi
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/opt/geniepod/models/CHANGE_ME.gguf"
    echo "WARNING: no model auto-detected; using placeholder $MODEL_PATH"
    echo "         edit /etc/jetson-llm/server.env before starting the service."
fi

# Verify the chosen user is in video + render groups (CUDA needs both).
for grp in video render; do
    if id -nG "$USER_NAME" 2>/dev/null | tr ' ' '\n' | grep -qx "$grp"; then
        : # OK
    else
        echo "WARNING: user '$USER_NAME' is not in '$grp' group — CUDA may fail."
        echo "         fix with: sudo usermod -a -G $grp $USER_NAME && relog"
    fi
done

cat <<EOF
Installing jetson-llm-server:
  User:    $USER_NAME
  Binary:  $BIN_SRC
           -> /opt/jetson-llm/bin/jetson-llm-server
  Unit:    $UNIT_SRC
           -> /etc/systemd/system/jetson-llm-server.service
  Env:     /etc/jetson-llm/server.env
           MODEL_PATH=$MODEL_PATH
           PORT=$PORT

EOF

# 1. Binary
install -d -m 755 /opt/jetson-llm/bin
install -m 755 "$BIN_SRC" /opt/jetson-llm/bin/jetson-llm-server

# 2. Env file (don't clobber an existing operator-edited one)
install -d -m 755 /etc/jetson-llm
if [ ! -f /etc/jetson-llm/server.env ]; then
    cat > /etc/jetson-llm/server.env <<EOF
# /etc/jetson-llm/server.env
# Reload after editing:  sudo systemctl restart jetson-llm-server
MODEL_PATH=$MODEL_PATH
PORT=$PORT
EOF
    chmod 644 /etc/jetson-llm/server.env
    echo "Wrote /etc/jetson-llm/server.env"
else
    echo "Kept existing /etc/jetson-llm/server.env (not overwritten)"
fi

# 3. Unit (substitute User=)
sed "s|^User=aihpc$|User=$USER_NAME|" "$UNIT_SRC" \
    > /etc/systemd/system/jetson-llm-server.service
chmod 644 /etc/systemd/system/jetson-llm-server.service

# 4. systemd reload
systemctl daemon-reload

cat <<'EOF'

Installed. To enable + start:

  sudo systemctl enable --now jetson-llm-server
  sudo systemctl status jetson-llm-server
  sudo journalctl -u jetson-llm-server -f

To uninstall:

  sudo systemctl disable --now jetson-llm-server
  sudo rm /etc/systemd/system/jetson-llm-server.service
  sudo rm -rf /opt/jetson-llm /etc/jetson-llm
  sudo systemctl daemon-reload
EOF
