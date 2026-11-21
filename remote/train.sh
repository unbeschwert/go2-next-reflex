#!/usr/bin/env bash
# Start a training run on the server in a tmux session (keeps running if you disconnect).
# Usage: remote/train.sh <config.yaml> [run-name]
source "$(dirname "$0")/_common.sh"
config=${1:?usage: remote/train.sh training/configs/<name>.yaml [run-name]}
[[ -f "$ROOT/$config" ]] || { echo "No such config: $config (give the path from the repo root)" >&2; exit 1; }
name=${2:-$(basename "$config" .yaml)-$(date +%Y%m%d-%H%M%S)}
session="snn-${name//[.:]/-}"
echo "→ checking the config locally"
(cd "$ROOT" && uv run python -m training.snn check --config "$config")
echo "→ syncing code"
sync_code
remote "cd $REMOTE_DIR && mkdir -p runs/$name && tmux new-session -d -s '$session' bash remote/_run.sh train '$config' 'runs/$name'"
echo "Started run '$name' (tmux session $session)."
echo "  follow:  remote/status.sh $name -f"
echo "  fetch:   remote/fetch.sh $name"
