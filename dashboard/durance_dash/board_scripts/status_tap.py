"""Follows the gesture bridge's status snapshot and log; prints JSON lines on stdout.

    python3 -c <this> STATUS_FILE LOG_FILE        (sent by the dashboard over SSH, or run locally)

{"type": "status", "data": {...}}  when the snapshot changes (the bridge writes it 10x per second)
{"type": "log", "line": "...", "history": bool}  for every new log line (last 30 lines first)
{"type": "hb", "t": ...}  every 2 s, so the reader can tell a quiet stream from a dead one.
Exits when stdout is closed (the SSH session ended). Python 3.8+, stdlib only.
"""
import json
import os
import sys
import time

status_path, log_path = sys.argv[1], sys.argv[2]


def emit(obj):
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        sys.exit(0)


def open_log():
    try:
        f = open(log_path, "rb")
    except OSError:
        return None, None
    return f, os.fstat(f.fileno()).st_ino


last_mtime, last_hb = None, 0.0
f, ino = open_log()
if f is not None:
    f.seek(0, 2)
    end = f.tell()
    f.seek(max(0, end - 8192))
    tail = f.read().decode(errors="replace").splitlines()[-30:]
    if end > 8192 and tail:
        tail = tail[1:]                       # first line may be cut
    for line in tail:
        emit({"type": "log", "line": line, "history": True})
buf = b""
while True:
    now = time.time()
    try:
        st = os.stat(status_path)
        if st.st_mtime_ns != last_mtime:
            last_mtime = st.st_mtime_ns
            with open(status_path) as sf:
                emit({"type": "status", "data": json.load(sf)})
    except (OSError, ValueError):
        pass
    if f is None:
        f, ino = open_log()
    else:
        try:
            st = os.stat(log_path)
            if st.st_ino != ino or st.st_size < f.tell():      # rotated or truncated: start again
                f.close()
                f, ino = open_log()
                buf = b""
        except OSError:
            pass
        if f is not None:
            chunk = f.read()
            if chunk:
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    emit({"type": "log", "line": line.decode(errors="replace")[:2000], "history": False})
    if now - last_hb >= 2.0:
        emit({"type": "hb", "t": now})
        last_hb = now
    time.sleep(0.05)
