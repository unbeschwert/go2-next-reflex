"""python -m durance_dash [--config PATH]  ->  the dashboard on http://127.0.0.1:8800 (see README.md)."""
import argparse
import logging
import sys

import uvicorn

from .config import ConfigError, load


def main() -> None:
    ap = argparse.ArgumentParser(prog="durance_dash")
    ap.add_argument("--config", default=None, help="config.toml (default: $DD_CONFIG or ~/.config/durance-dashboard/config.toml)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load(a.config)
    except (ConfigError, OSError, ValueError) as e:
        sys.exit(f"configuration error: {e}")
    from .app import create_app
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level="info", access_log=False,
                proxy_headers=False, server_header=False, timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
