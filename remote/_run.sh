#!/usr/bin/env bash
# Runs ON THE SERVER inside tmux (started by remote/train.sh or remote/qat.sh).
# Usage: _run.sh train <config> <run-dir>   |   _run.sh qat <run-dir>
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
command=$1; shift
if [[ $command == train ]]; then
    config=$1 run_dir=$2
    args=(train --config "$config" --run-dir "$run_dir")
else
    run_dir=$1
    args=(qat --run-dir "$run_dir")
fi
"$HOME/.local/bin/uv" run --group train python -m training.snn "${args[@]}" 2>&1 | tee -a "$run_dir/train.log"
code=${PIPESTATUS[0]}
echo "=== finished with exit code $code at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$run_dir/train.log"
