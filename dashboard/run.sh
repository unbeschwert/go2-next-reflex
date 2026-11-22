#!/bin/bash
# Start the dashboard on this laptop: http://127.0.0.1:8800
#   ./run.sh            (creates .venv on first run)
#   ./run.sh --test     run the test suite
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
  uv venv --system-site-packages .venv
  uv pip install --python .venv/bin/python -r requirements.txt pytest
fi
if [ "${1:-}" = "--test" ]; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 exec .venv/bin/python -m pytest -q
fi
exec .venv/bin/python -m durance_dash "$@"
