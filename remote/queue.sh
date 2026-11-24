#!/usr/bin/env bash
# Train several configs on the server, PARALLEL at a time, in one tmux session (survives disconnects).
#
#   remote/queue.sh <name> <config.yaml>...     runs go to runs/<name>/<config basename>/ on the server
#   PARALLEL=3 remote/queue.sh ...              default: auto = GPU memory / 20 GB (an 80x80 run needs ~19 GB)
#   WHEN_DONE=stop STOP_AFTER_MIN=60 ...        on a Runpod pod: after the queue, save runs/<name> to the
#                                               global volume, wait STOP_AFTER_MIN minutes (time to fetch),
#                                               then stop the pod. Default WHEN_DONE=stop on a pod.
#
# Follow: remote/status.sh <name>/<config> -f (one run) or ssh + `tmux attach -t q-<name>`.
# Fetch:  remote/fetch.sh <name>   (all runs of the queue)
source "$(dirname "$0")/_common.sh"
name=${1:?usage: remote/queue.sh <queue-name> <config.yaml>...}
shift
(($#)) || { echo "give at least one config" >&2; exit 2; }
for config in "$@"; do
    [[ -f "$ROOT/$config" ]] || { echo "No such config: $config (path from the repo root)" >&2; exit 1; }
    echo "→ checking $config"
    (cd "$ROOT" && "${PYTHON_CMD[@]}" -m training.snn check --config "$config" >/dev/null) ||
        { echo "config check failed: $config" >&2; exit 1; }
done
echo "→ syncing code"
sync_code
session="q-${name//[.:\/]/-}"
args="$(printf '%q ' "$name" "$@")"
remote "cd $REMOTE_DIR && tmux new-session -d -s '$session' \
    -e PARALLEL='${PARALLEL:-auto}' -e WHEN_DONE='${WHEN_DONE:-}' -e STOP_AFTER_MIN='${STOP_AFTER_MIN:-60}' \
    bash remote/_queue.sh $args"
echo "Started queue '$name' with $# configs (tmux session $session on the server)."
echo "  follow: remote/status.sh $name/<config> -f"
echo "  fetch:  remote/fetch.sh $name"
