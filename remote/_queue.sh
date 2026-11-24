#!/usr/bin/env bash
# Runs ON THE SERVER inside tmux (started by remote/queue.sh). Usage: _queue.sh <name> <config>...
# Env: PARALLEL (auto = GPU memory / 20 GB), WHEN_DONE (stop | none; default stop on a Runpod pod),
# STOP_AFTER_MIN (minutes between the end of the queue and the stop, to fetch the results).
set -uo pipefail
cd "$(dirname "$0")/.."
# Runpod writes the pod's env (RUNPOD_POD_ID, RUNPOD_API_KEY) here; SSH sessions don't always see it.
# shellcheck disable=SC1091
[[ -f /etc/rp_environment ]] && source /etc/rp_environment
name=$1
shift
mkdir -p "runs/$name"
log="runs/$name/queue.log"
say() { echo "[queue $(date '+%H:%M:%S')] $*" | tee -a "$log"; }

parallel=${PARALLEL:-auto}
if [[ $parallel == auto ]]; then
    mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    parallel=$(((${mem_mib:-20480} / 1024) / 20))
    ((parallel >= 1)) || parallel=1
fi
say "$# configs, $parallel at a time: $*"

running=()
for config in "$@"; do
    while ((${#running[@]} >= parallel)); do
        wait -n
        alive=()
        for pid in "${running[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
        running=("${alive[@]}")
    done
    run_dir="runs/$name/$(basename "$config" .yaml)"
    mkdir -p "$run_dir"
    say "start $config → $run_dir"
    bash remote/_run.sh train "$config" "$run_dir" >/dev/null &
    running+=($!)
done
wait
for config in "$@"; do
    run_dir="runs/$name/$(basename "$config" .yaml)"
    say "$(basename "$run_dir"): $(grep -h 'fixed point\|finished with exit code' "$run_dir/train.log" | tr -s ' ' | paste -sd ';' -)"
done
say "queue finished"

# On a Runpod pod: save the runs to the global volume (object storage: no -t, --inplace), then stop.
when_done=${WHEN_DONE:-}
[[ -n $when_done || -z ${RUNPOD_POD_ID:-} ]] || when_done=stop
if [[ -d /workspace-global ]]; then
    dest=/workspace-global/go2-next-reflex/runs/$name
    mkdir -p "$dest" && rsync -r --inplace --checksum "runs/$name/" "$dest/" &&
        say "saved runs/$name to $dest" || { say "saving to the global volume FAILED; not stopping the pod"; when_done=none; }
fi
if [[ $when_done == stop && -n ${RUNPOD_POD_ID:-} && -n ${RUNPOD_API_KEY:-} ]]; then
    say "stopping this pod in ${STOP_AFTER_MIN:-60} min (fetch now: remote/fetch.sh $name; cancel: tmux kill-session)"
    sleep $((${STOP_AFTER_MIN:-60} * 60))
    say "stopping pod $RUNPOD_POD_ID"
    curl -fsS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
        -d '{"action":"stop"}' "${RUNPOD_API_URL:-https://api.runpod.io/v2}/pods/$RUNPOD_POD_ID/action" >/dev/null ||
        say "stop request failed; stop the pod in the Runpod console"
fi
exec bash
