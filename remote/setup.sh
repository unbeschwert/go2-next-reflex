#!/usr/bin/env bash
# One-time (and after dependency changes): copy code + riverops wheel, install uv and the
# training environment (Python 3.12, torch with CUDA, riverops) on the server, check the GPU.
source "$(dirname "$0")/_common.sh"
ls "$ROOT"/data/riverops-*.whl >/dev/null || { echo "Put riverops-0.2.0-py3-none-any.whl in data/ first." >&2; exit 1; }
echo "→ syncing code to $REMOTE_HOST:$REMOTE_DIR"
remote "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/runs ~/.local/bin"
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
