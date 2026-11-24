#!/usr/bin/env bash
# One-time (and after dependency changes): copy code + riverops wheel, install uv and the
# training environment (Python 3.12, torch with CUDA, riverops) on the server, check the GPU.
source "$(dirname "$0")/_common.sh"
ls "$DATA_DIR"/riverops-*.whl >/dev/null || { echo "Put riverops-0.2.0-py3-none-any.whl in $DATA_DIR first." >&2; exit 1; }
echo "→ syncing code to $REMOTE_HOST:$REMOTE_DIR"
# rsync and tmux are needed on the server; a fresh Runpod image may lack them (it runs as root).
remote 'missing=""; for c in rsync tmux; do command -v $c >/dev/null || missing="$missing $c"; done
    if [ -n "$missing" ]; then
        if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
            echo "  installing$missing"; apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $missing >/dev/null
        else echo "missing on the server:$missing" >&2; exit 1; fi
    fi'
remote "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/runs ~/.local/bin"
# --tools-only: stop here (a CPU pod that only moves data needs no training environment).
[[ ${1:-} != --tools-only ]] || { echo "tools ready (rsync, tmux)"; exit 0; }
sync_code
if ! remote "test -x $UV"; then
    echo "→ copying uv to the server"
    scp -q "${SSH_OPTS[@]}" "$(command -v uv)" "$REMOTE_HOST:.local/bin/uv"
fi
echo "→ installing the training environment (first time downloads torch, ~3 GB)"
remote "cd $REMOTE_DIR && $UV sync --group train --quiet && $UV run --group train python -c '
import torch, riverops
print(\"torch\", torch.__version__, \"| CUDA available:\", torch.cuda.is_available(),
      \"|\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\")'"
