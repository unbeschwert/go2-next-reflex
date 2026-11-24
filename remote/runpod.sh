#!/usr/bin/env bash
# Manage a Runpod GPU pod as the training server (run it locally). Adapted from the user's tabfm-bench
# scripts/runpod/create_pod.sh: Secure Cloud, EU datacenters, one GPU, the account's global volume at
# /workspace-global, SSH on an exposed TCP port. Code, the riverops wheel and data go up over rsync with
# the normal remote/*.sh scripts, so the pod needs no secrets.
#
#   remote/runpod.sh list         GPUs that match (in the order `create` tries them), creates nothing
#   remote/runpod.sh create       create a pod (asks first) and write remote/runpod.env for remote/*.sh
#   remote/runpod.sh status       state, GPU and price of the pod in remote/runpod.env
#   remote/runpod.sh stop         stop it (the container disk is wiped; the global volume stays)
#   remote/runpod.sh terminate    delete it
#
# Then: export REMOTE_ENV=remote/runpod.env; remote/setup.sh; remote/push_data.sh --config <cfg>;
# remote/queue.sh <configs...>; remote/fetch.sh <run>; remote/runpod.sh terminate. See remote/RUNPOD.md.
#
# Environment: RUNPOD_API_KEY (or apikey in ~/.runpod/config.toml); RUNPOD_GLOBAL_VOLUME (id or name,
# default: the account's only one; NO_GLOBAL_VOLUME=1 for none); MIN_VRAM_GB=40; MAX_PRICE=2.50 (USD/h);
# GPU_TYPES (comma-separated ids, preference order); REGION=EUROPE; CLOUD=SECURE; SORT=vram|price;
# MIN_CUDA=13.0 (the locked torch 2.14 wheels use CUDA 13.0); DISK_GB=80; IMAGE; POD_NAME=go2-snn; --yes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_OUT="$ROOT/remote/runpod.env"
KNOWN_HOSTS="$HOME/.ssh/known_hosts_runpod"   # pods reuse IP:port pairs with new host keys
API="${RUNPOD_API_URL:-https://api.runpod.io/v2}"
MIN_VRAM_GB="${MIN_VRAM_GB:-40}"
MAX_PRICE="${MAX_PRICE-2.50}"
SORT="${SORT:-vram}"
GPU_TYPES="${GPU_TYPES:-}"
REGION="${REGION:-EUROPE}"
CLOUD="${CLOUD:-SECURE}"
MIN_CUDA="${MIN_CUDA:-13.0}"
DISK_GB="${DISK_GB:-80}"
IMAGE="${IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"   # only its driver setup and sshd are used
POD_NAME="${POD_NAME:-go2-snn}"
GLOBAL_MOUNT=/workspace-global

die() { echo "runpod: $*" >&2; exit 1; }

command=${1:-}
yes=0
for arg in "${@:2}"; do
    case $arg in
        --yes | -y) yes=1 ;;
        *) die "unknown argument: $arg" ;;
    esac
done
case $command in
    list | create | status | stop | terminate) ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
command -v jq >/dev/null || die "needs jq"

key="${RUNPOD_API_KEY:-}"
if [[ -z $key && -f $HOME/.runpod/config.toml ]]; then
    key="$(sed -nE 's/^[[:space:]]*api_?key[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/Ip' "$HOME/.runpod/config.toml" | head -n1)"
fi
[[ -n $key ]] || die "set RUNPOD_API_KEY or put apikey = \"...\" in ~/.runpod/config.toml (Runpod console → Settings → API keys)"

# api METHOD PATH [BODY]: sets STATUS and BODY. The key goes through a file descriptor, not argv.
api() {
    local args=(-sS -X "$1" -w $'\n%{http_code}')
    [[ $# -lt 3 ]] || args+=(-H "Content-Type: application/json" --data-binary "$3")
    local out
    out="$(curl "${args[@]}" -H @<(printf 'Authorization: Bearer %s\n' "$key") "$API$2")" ||
        die "request to $API$2 failed"
    STATUS="${out##*$'\n'}"
    BODY="${out%$'\n'*}"
}
expect() { # expect CODE WHAT
    [[ $STATUS == "$1" ]] || die "$2 failed (HTTP $STATUS): $(jq -r '.detail // .title // .' <<<"$BODY" 2>/dev/null || echo "$BODY")"
}

pod_from_env() {
    [[ -f $ENV_OUT ]] || die "no $ENV_OUT; create a pod first"
    POD_ID="$(sed -nE 's/^RUNPOD_POD_ID=([^ #]+).*/\1/p' "$ENV_OUT")"
    [[ -n $POD_ID ]] || die "no RUNPOD_POD_ID in $ENV_OUT"
}

case $command in
    status)
        pod_from_env
        api GET "/pods/$POD_ID"
        expect 200 "reading pod $POD_ID"
        jq -r '"pod \(.id): \(.status // .desiredStatus), \(.gpu.displayName // .gpu.id // "?"), $\(.cost // .costPerHr)/h, ssh: \(.ssh.direct.command // "-")"' <<<"$BODY"
        exit 0
        ;;
    stop)
        pod_from_env
        api POST "/pods/$POD_ID/action" '{"action":"stop"}'
        [[ $STATUS == 2* ]] || die "stopping $POD_ID failed (HTTP $STATUS): $BODY"
        echo "stopped pod $POD_ID (no GPU billing; its container disk is wiped)"
        exit 0
        ;;
    terminate)
        pod_from_env
        if ((!yes)); then
            read -r -p "Delete pod $POD_ID? Results not fetched or saved to the global volume are lost. [y/N] " answer
            [[ $answer == [yY]* ]] || die "cancelled"
        fi
        api DELETE "/pods/$POD_ID"
        [[ $STATUS == 2* ]] || die "deleting $POD_ID failed (HTTP $STATUS): $BODY"
        mv "$ENV_OUT" "$ENV_OUT.terminated"
        echo "deleted pod $POD_ID ($ENV_OUT moved to runpod.env.terminated)"
        exit 0
        ;;
esac

# list / create
if [[ -f $ENV_OUT && $command == create ]]; then
    die "$ENV_OUT exists (pod $(sed -nE 's/^RUNPOD_POD_ID=([^ #]+).*/\1/p' "$ENV_OUT")); terminate it first or move the file"
fi

vol_id=""
if [[ ${NO_GLOBAL_VOLUME:-0} != 1 ]]; then
    api GET /global-volumes
    expect 200 "listing global volumes"
    if [[ -n ${RUNPOD_GLOBAL_VOLUME:-} ]]; then
        vol_id="$(jq -r --arg w "$RUNPOD_GLOBAL_VOLUME" \
            '[.globalVolumes[] | select(.id == $w or .name == $w)] | if length == 1 then .[0].id else "" end' <<<"$BODY")"
    else
        vol_id="$(jq -r '.globalVolumes | if length == 1 then .[0].id else "" end' <<<"$BODY")"
    fi
    [[ -n $vol_id ]] || die "set RUNPOD_GLOBAL_VOLUME to one of: $(jq -r '[.globalVolumes[] | "\(.id) \(.name)"] | join(", ")' <<<"$BODY") (or NO_GLOBAL_VOLUME=1)"
    vol_name="$(jq -r --arg id "$vol_id" '.globalVolumes[] | select(.id == $id) | .name' <<<"$BODY")"
fi
api GET /account/ssh-keys
[[ $STATUS != 200 ]] || jq -e '(.keys // []) | length > 0' <<<"$BODY" >/dev/null ||
    echo "warning: no SSH public key registered (Runpod → Settings → SSH keys); SSH into the pod will fail" >&2

# Stock per datacenter (does not distinguish Secure from Community: a GPU without Secure capacity fails
# at creation and the next one is tried), then the GPU types on $CLOUD with enough VRAM.
api GET "/catalog/datacenters?regions=$REGION&include=GPU_AVAILABILITY"
expect 200 "listing datacenters"
stock="$(jq -c '
    reduce (.dataCenters[] | .id as $dc | .gpuAvailability[]? | select(.availability != "NONE")
            | {gpu: .id, dc: $dc}) as $s ({}; .[$s.gpu] += [$s.dc])' <<<"$BODY")"
api GET "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=$CLOUD&minCudaVersion=$MIN_CUDA"
expect 200 "listing GPU types"
candidates="$(jq -c \
    --argjson stock "$stock" --argjson vram "$MIN_VRAM_GB" --arg cloud "${CLOUD,,}" \
    --arg max "$MAX_PRICE" --arg prefer "$GPU_TYPES" --arg sort "$SORT" '
    ($prefer | split(",") | map(select(length > 0))) as $order
    | [.gpus[]
       | select(.memory >= $vram and (.[$cloud] // false) and .price[$cloud] != null)
       | select($max == "" or .price[$cloud] <= ($max | tonumber))
       | {id, memory, price: .price[$cloud], dcs: ($stock[.id] // [])}
       | select(.dcs | length > 0)
       | select(($order | length) == 0 or (.id as $g | any($order[]; . == $g)))]
    | if ($order | length) > 0 then sort_by(.id as $g | $order | index($g))
      elif $sort == "vram" then sort_by(-.memory, .price) else sort_by(.price) end' <<<"$BODY")"
count="$(jq length <<<"$candidates")"
((count > 0)) || die "no $CLOUD GPU with >= ${MIN_VRAM_GB} GB available in $REGION${MAX_PRICE:+ under \$$MAX_PRICE/h} (CUDA >= $MIN_CUDA)"

[[ -z $vol_id ]] || echo "Global volume: $vol_name ($vol_id) → $GLOBAL_MOUNT"
echo "$CLOUD GPUs with >= ${MIN_VRAM_GB} GB in $REGION, CUDA >= $MIN_CUDA (tried in this order):"
jq -r '.[] | [.id, (.memory | tostring) + " GB", "$" + (.price * 100 | round / 100 | tostring) + "/h", (.dcs | join(","))] | @tsv' \
    <<<"$candidates" | awk -F'\t' '{printf "  %-40s %7s %9s  %s\n", $1, $2, $3, $4}'
[[ $command != list ]] || exit 0

body_for() { # body_for INDEX
    jq -c --argjson i "$1" --arg name "$POD_NAME" --arg image "$IMAGE" --arg cloud "$CLOUD" \
        --arg cuda "$MIN_CUDA" --argjson disk "$DISK_GB" --arg vol "$vol_id" --arg mount "$GLOBAL_MOUNT" '
        .[$i] as $gpu
        | {name: $name, image: $image, cloud: $cloud, dataCenterIds: $gpu.dcs,
           gpu: {id: $gpu.id, count: 1, minCudaVersion: $cuda},
           disk: $disk, ports: ["22/tcp"], startSsh: true}
        + (if $vol == "" then {} else {mounts: {global: [{volumeId: $vol, path: $mount}]}} end)' <<<"$candidates"
}
if ((!yes)); then
    read -r -p "Create a pod with the first of these that has capacity (billed per hour until terminated)? [y/N] " answer
    [[ $answer == [yY]* ]] || die "cancelled"
fi

pod_id="" gpu=""
for ((i = 0; i < count; i++)); do
    gpu="$(jq -r --argjson i "$i" '.[$i].id' <<<"$candidates")"
    api POST /pods "$(body_for "$i")"
    case $STATUS in
        201) pod_id="$(jq -r .id <<<"$BODY")"; echo "Created pod $pod_id with $gpu at \$$(jq -r .cost <<<"$BODY")/h"; break ;;
        400 | 5*) echo "  $gpu: $(jq -r '.detail // .title' <<<"$BODY" 2>/dev/null) (HTTP $STATUS), trying the next GPU" ;;
        *) die "creating the pod failed (HTTP $STATUS): $(jq -r '.detail // .title // .' <<<"$BODY" 2>/dev/null)" ;;
    esac
done
[[ -n $pod_id ]] || die "none of the candidate GPUs could be placed; try later, raise MAX_PRICE or lower MIN_VRAM_GB"

ssh_cmd=""
for _ in $(seq 1 "${WAIT_POLLS:-90}"); do
    api GET "/pods/$pod_id"
    if [[ $STATUS == 200 ]]; then
        status="$(jq -r .status <<<"$BODY")"
        ssh_cmd="$(jq -r '.ssh.direct.command // empty' <<<"$BODY")"
        [[ $status != RUNNING || -z $ssh_cmd ]] || break
        [[ $status != ERROR && $status != TERMINATED ]] || die "pod $pod_id is $status"
    fi
    sleep "${WAIT_SECONDS:-10}"
done
[[ -n $ssh_cmd ]] || die "pod $pod_id did not become reachable over SSH (it still bills: remote/runpod.sh terminate)"
host="$(grep -oE '[[:alnum:]_.-]+@[0-9.]+' <<<"$ssh_cmd" | head -1)"
port="$(grep -oE -- '-p[[:space:]]*[0-9]+' <<<"$ssh_cmd" | grep -oE '[0-9]+' | head -1)"
[[ -n $host && -n $port ]] || die "could not parse the SSH command '$ssh_cmd'; write remote/runpod.env by hand"
touch "$KNOWN_HOSTS"
ssh-keygen -q -f "$KNOWN_HOSTS" -R "[${host#*@}]:$port" >/dev/null 2>&1 || true
cat >"$ENV_OUT" <<EOF
# Written by remote/runpod.sh create on $(date '+%Y-%m-%d %H:%M'). Use with: export REMOTE_ENV=remote/runpod.env
REMOTE_HOST=$host
REMOTE_DIR=go2-next-reflex
REMOTE_SSH_OPTS="-o Port=$port -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS"
RUNPOD_POD_ID=$pod_id
RUNPOD_GPU="$gpu"
EOF
echo "Pod $pod_id is running ($ssh_cmd). Wrote $ENV_OUT."
echo "Next: export REMOTE_ENV=remote/runpod.env && remote/setup.sh"
