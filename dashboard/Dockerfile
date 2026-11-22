# Durance AI dashboard: container image (v1).
# Board (SSH), cluster (SSH/Slurm), dataset, settings, manual control and E-STOP work in the container.
# The local simulator / desktop RViz (ROS 2 Jazzy + unitree msgs) and recording (river-tools) still run
# from the laptop install (./run.sh); see README.md "Container".
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DD_CONFIG=/config/config.toml
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-client rsync git ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 1000 --create-home --shell /usr/sbin/nologin durance \
 && mkdir -p /state && chown 1000:1000 /state && chmod 700 /state

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY durance_dash/ durance_dash/
COPY static/ static/
COPY board/ board/
COPY docs/ docs/

USER durance
EXPOSE 8800
# health check on the port of the mounted config (not a fixed 8800: another service may answer there)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import tomllib,urllib.request,sys; p=tomllib.load(open('/config/config.toml','rb')).get('server',{}).get('port',8800); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/api/health', timeout=4).status == 200 else 1)"
CMD ["python", "-m", "durance_dash"]
