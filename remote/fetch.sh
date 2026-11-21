#!/usr/bin/env bash
# Download a run (checkpoints, metrics, deploy/ files) to runs/<name>/. Usage: remote/fetch.sh [run-name]
source "$(dirname "$0")/_common.sh"
name=${1:-$(latest_run)}
[[ -n "$name" ]] || { echo "No runs on the server yet." >&2; exit 1; }
mkdir -p "$ROOT/runs/$name"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_HOST:$REMOTE_DIR/runs/$name/" "$ROOT/runs/$name/"
echo "Fetched to runs/$name/"
[[ -f "$ROOT/runs/$name/deploy/DEPLOY.md" ]] && echo "Next: runs/$name/deploy/DEPLOY.md"
