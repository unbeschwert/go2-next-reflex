# Shared helpers for the remote/*.sh scripts. Not meant to be run directly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# REMOTE_ENV selects another server file, e.g. REMOTE_ENV=remote/runpod.env (written by remote/runpod.sh).
ENV_FILE="${REMOTE_ENV:-remote/remote.env}"
[[ $ENV_FILE == /* ]] || ENV_FILE="$ROOT/$ENV_FILE"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE: copy remote/remote.env.example and fill it in (or create a pod: remote/runpod.sh)." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
# Extra ssh options from the env file, e.g. a pod's port: REMOTE_SSH_OPTS="-o Port=12345". They are
# -o options so that ssh, scp and rsync all take them.
# shellcheck disable=SC2206
[[ -z ${REMOTE_SSH_OPTS:-} ]] || SSH_OPTS+=($REMOTE_SSH_OPTS)
UV='$HOME/.local/bin/uv'

# Local data folder (dataset, riverops wheel) and Python, overridable for a git worktree, which has no
# data/ or .venv of its own: DATA_DIR=<main checkout>/data PYTHON=<main checkout>/.venv/bin/python.
DATA_DIR="${DATA_DIR:-$ROOT/data}"
read -r -a PYTHON_CMD <<<"${PYTHON:-uv run python}"

remote() { ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"; }

# Copy the code (tracked + new, non-ignored files) and the riverops wheel to the server.
sync_code() {
    (cd "$ROOT" && git ls-files -co --exclude-standard |
        rsync -az --files-from=- -e "ssh ${SSH_OPTS[*]}" ./ "$REMOTE_HOST:$REMOTE_DIR/")
    remote "mkdir -p $REMOTE_DIR/data"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$DATA_DIR"/riverops-*.whl "$REMOTE_HOST:$REMOTE_DIR/data/"
}

latest_run() { remote "ls -1t $REMOTE_DIR/runs 2>/dev/null | head -1"; }
