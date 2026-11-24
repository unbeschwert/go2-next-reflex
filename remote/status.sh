#!/usr/bin/env bash
# Show running sessions, GPU use and the end of a run's log. Usage: remote/status.sh [run-name] [-f]
source "$(dirname "$0")/_common.sh"
name=${1:-$(latest_run)}
[[ -n "$name" ]] || { echo "No runs on the server yet." >&2; exit 1; }
remote "tmux ls 2>/dev/null | grep -E '^(snn|q)-' || echo 'no training sessions running'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/GPU: /'
echo '--- runs/$name/train.log'"
if [[ "${2:-}" == "-f" ]]; then
    remote "tail -n 40 -f $REMOTE_DIR/runs/$name/train.log"
else
    remote "tail -n 40 $REMOTE_DIR/runs/$name/train.log"
fi
