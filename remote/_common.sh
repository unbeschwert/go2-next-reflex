# Shared helpers for the remote/*.sh scripts. Not meant to be run directly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$ROOT/remote/remote.env" ]]; then
    echo "Missing remote/remote.env: copy remote/remote.env.example and fill it in." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$ROOT/remote/remote.env"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
UV='$HOME/.local/bin/uv'

remote() { ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"; }

# Copy the code (tracked + new, non-ignored files) and the riverops wheel to the server.
sync_code() {
    (cd "$ROOT" && { git ls-files -co --exclude-standard; ls data/riverops-*.whl; } |
        rsync -az --files-from=- -e "ssh ${SSH_OPTS[*]}" ./ "$REMOTE_HOST:$REMOTE_DIR/")
}

latest_run() { remote "ls -1t $REMOTE_DIR/runs 2>/dev/null | head -1"; }
