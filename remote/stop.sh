#!/usr/bin/env bash
# Stop a running training session. Usage: remote/stop.sh <run-name>
source "$(dirname "$0")/_common.sh"
name=${1:?usage: remote/stop.sh <run-name>}
remote "tmux kill-session -t 'snn-${name//[.:]/-}'" && echo "Stopped $name"
