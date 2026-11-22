#!/bin/bash
# The dashboard in a container, without docker compose (same settings as docker-compose.yml).
#
#   docker/run-dashboard.sh build      build the image durance-dashboard:1.0.0
#   docker/run-dashboard.sh start      start it  ->  http://127.0.0.1:8800
#   docker/run-dashboard.sh stop       stop and remove the container (users, audit log stay in the volume)
#   docker/run-dashboard.sh logs       follow its log
#   docker/run-dashboard.sh status     container state + health check
#
# CONFIG=/path/to/config.toml docker/run-dashboard.sh start   uses another config (e.g. another port).
# Needs: Docker usable by this user (group "docker"), the board link for board features, ~/.ssh with the
# board's key and known_hosts. The laptop install (./run.sh) must not use the same port.
# Mounts the folder above dashboard/ as /project (the repository, or ~/go2-hackathon in the laptop layout).
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=durance-dashboard:1.0.0
NAME=durance-dashboard
CONFIG=${CONFIG:-$PWD/docker/config.container.toml}

case "${1:-}" in
  build)
    docker build -t "$IMAGE" .
    ;;
  start)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --restart unless-stopped \
      --network host --hostname "$(hostname)" --user "$(id -u):$(id -g)" \
      --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges:true \
      -e DD_CONFIG=/config/config.toml \
      -v "$CONFIG:/config/config.toml:ro" \
      -v "$HOME/.ssh:/home/durance/.ssh:ro" \
      -v "$(dirname "$PWD"):/project" \
      -v "$HOME/river-captures:/captures" \
      -v durance-state:/state \
      "$IMAGE" >/dev/null
    echo "started $NAME: http://127.0.0.1:$(grep -m1 -E '^port' "$CONFIG" | grep -oE '[0-9]+' || echo 8800)"
    ;;
  stop)
    docker rm -f "$NAME" >/dev/null && echo "stopped $NAME"
    ;;
  logs)
    docker logs -f "$NAME"
    ;;
  status)
    docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}: {{.Status}}'
    ;;
  *)
    sed -n '2,14p' "$0"; exit 1
    ;;
esac
