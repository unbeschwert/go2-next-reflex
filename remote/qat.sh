#!/usr/bin/env bash
# Redo formats → QAT → fixed-point check → export for an existing run, from its float_best.pt
# (e.g. after a QAT failure, or to try other quant settings: edit runs/<name>/config.yaml on the server first).
# Usage: remote/qat.sh <run-name> [--set section.key=value ...]   e.g. --set quant.layer_scaling=true
source "$(dirname "$0")/_common.sh"
name=${1:?usage: remote/qat.sh <run-name> [--set section.key=value ...]}; shift
extra=""; (( $# )) && extra=$(printf " %q" "$@")
session="snn-${name//[.:]/-}-qat"
echo "→ syncing code"
sync_code
remote "cd $REMOTE_DIR && test -f runs/$name/float_best.pt && tmux new-session -d -s '$session' bash remote/_run.sh qat 'runs/$name'$extra" \
    || { echo "runs/$name/float_best.pt not found on the server" >&2; exit 1; }
echo "Started QAT for '$name' (tmux session $session)."
echo "  follow:  remote/status.sh $name -f"
echo "  fetch:   remote/fetch.sh $name"
