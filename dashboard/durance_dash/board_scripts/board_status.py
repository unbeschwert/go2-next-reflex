"""Runs ON THE KRIA (sent by the dashboard over SSH, nothing is installed). Prints one JSON object.
Read-only (apart from an md5 cache in /tmp): processes, mode, model, firmware, server config, lock, autostart,
bridge settings/status, health.
Python 3.8+, stdlib only (BusyBox userland)."""
import hashlib
import json
import os
import re
import subprocess
import time

OUT = {"ok": True, "errors": []}


def read(path, limit=200000):
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode(errors="replace")
    except OSError:
        return None


def procs():
    found = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            with open("/proc/%s/stat" % pid) as f:
                state = f.read().split(")")[-1].split()[0]
        except OSError:
            continue
        if cmd and state != "Z":
            found[int(pid)] = cmd
    return found


def environ(pid):
    try:
        with open("/proc/%d/environ" % pid, "rb") as f:
            env = dict(x.split(b"=", 1) for x in f.read().split(b"\0") if b"=" in x)
        return {k.decode(): v.decode(errors="replace") for k, v in env.items()}
    except OSError:
        return {}


ps = procs()
NAMES = {"camera": "despot_send_and_forget", "server": "backward-server", "bridge": "live_gesture_bridge.py",
         "viewer": "snn_input_viewer.py", "obstacle": "obstacle_detection", "screen": "board_screen_pub.py",
         "recorder": "despot_debug_camera"}
OUT["processes"] = {k: any(v in c for c in ps.values()) for k, v in NAMES.items()}
bridge = [(p, c) for p, c in ps.items() if "live_gesture_bridge.py" in c]
mode = "stopped"
if bridge:
    pid, cmd = bridge[0]
    domain = environ(pid).get("ROS_DOMAIN_ID", "?")
    if "--require-clear" in cmd:
        mode = "robot-dry" if "--dry-run" in cmd else "robot"
    else:
        mode = "test"
    OUT["bridge"] = {"pid": pid, "cmd": cmd[:400], "ros_domain": domain,
                     "classes": (re.search(r"--classes\s+(\S+)", cmd) or [None, "own"])[1]}
elif OUT["processes"]["camera"] or OUT["processes"]["server"]:
    mode = "camera-only" if OUT["processes"]["camera"] else "stopped"
OUT["mode"] = mode

w = None
try:
    with open("/root/weights.txt", "rb") as f:
        w = f.read()
except OSError:
    OUT["errors"].append("/root/weights.txt missing")
if w is not None:
    OUT["model"] = {"md5": hashlib.md5(w).hexdigest(), "values": len(w.split()), "bytes": len(w)}
OUT["model_classes"] = (read("/root/model_classes") or "dataset").strip() or "dataset"

MD5_CACHE = "/tmp/dd_md5_cache.json"      # path -> [size, mtime_ns, md5]: the bitstream is ~8 MB


def md5_cached(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    try:
        with open(MD5_CACHE) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(path)
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
        return hit[2]
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    cache[path] = [st.st_size, st.st_mtime_ns, h.hexdigest()]
    try:
        with open(MD5_CACHE, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass
    return h.hexdigest()


FW = "/usr/lib/firmware/xilinx/spleat/"
OUT["firmware"] = {"bit_md5": md5_cached(FW + "spleat.bit.bin"), "dtbo_md5": md5_cached(FW + "spleat.dtbo")}

cfg = read("/root/.config/spleat-server/config.yaml") or ""


def section(name):
    """key: value pairs of one top-level section of the server config (it is flat YAML)."""
    m = re.search(r"^%s:\s*\n((?:[ \t]+.*\n?)*)" % name, cfg, re.M)
    return dict(re.findall(r"^[ \t]+(\w+):\s*([^\s#]+)", m.group(1), re.M)) if m else {}


def num(v, kind=int):
    try:
        return kind(v)
    except (TypeError, ValueError):
        return None


ins, outs = section("inputs"), section("outputs")
OUT["server_config"] = {"input": num(ins.get("height")), "scaling": num(ins.get("scaling"), float),
                        "outputs": num(outs.get("channels")),
                        "output_shape": [num(outs.get("channels")), num(outs.get("height", 1)), num(outs.get("width", 1))]}
cam = read("/root/metavision_camera.json")
try:
    OUT["camera_config"] = json.loads(cam) if cam else None
except ValueError:
    OUT["camera_config"] = None
    OUT["errors"].append("metavision_camera.json is not valid JSON")

lock = read("/root/BOARD_LOCK")
OUT["lock"] = lock.strip()[:300] if lock and lock.strip() else None


def systemctl(*args):
    try:
        return subprocess.run(["systemctl"] + list(args), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              timeout=5).stdout.decode().strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


unit = "/etc/systemd/system/durance-auto.service"
OUT["autostart"] = {"installed": os.path.exists(unit), "enabled": systemctl("is-enabled", "durance-auto.service"),
                    "active": systemctl("is-active", "durance-auto.service"),
                    "conf": read("/root/durance_auto.conf"), "disabled_by_file": os.path.exists("/root/DURANCE_AUTO_DISABLE")}
OUT["spleat_service"] = systemctl("is-active", "spleat-server.service")

st = read("/tmp/bridge_status.json")
try:
    OUT["bridge_status"] = json.loads(st) if st else None
except ValueError:
    OUT["bridge_status"] = None
settings = read("/root/bridge_settings.json")
try:
    OUT["settings"] = json.loads(settings) if settings else {}
except ValueError:
    OUT["settings"] = {}
    OUT["errors"].append("bridge_settings.json is not valid JSON")
OUT["helpers"] = {"bridge_settings_py": os.path.exists("/root/bridge_settings.py"),
                  "go2_cmd_py": os.path.exists("/root/go2_cmd.py"),
                  "robot_link": os.path.exists("/root/robot_link.sh"), "laptop_link": os.path.exists("/root/laptop_link.sh")}

health = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": time.time()}
up = read("/proc/uptime")
health["uptime_s"] = float(up.split()[0]) if up else None
la = read("/proc/loadavg")
health["load"] = [float(x) for x in la.split()[:3]] if la else None
mi = read("/proc/meminfo") or ""
mem = {k: int(v) for k, v in re.findall(r"^(MemTotal|MemAvailable):\s+(\d+)", mi, re.M)}
health["mem_free_pct"] = round(100 * mem["MemAvailable"] / mem["MemTotal"], 1) if len(mem) == 2 else None
temps = []
for z in sorted(os.listdir("/sys/class/thermal")) if os.path.isdir("/sys/class/thermal") else []:
    t = read("/sys/class/thermal/%s/temp" % z)
    if t and t.strip().lstrip("-").isdigit():
        temps.append(int(t) / 1000)
health["temp_c"] = max(temps) if temps else None
try:
    s = os.statvfs("/root")
    health["disk_free_mb"] = int(s.f_bavail * s.f_frsize / 1e6)
except OSError:
    health["disk_free_mb"] = None
OUT["health"] = health
print(json.dumps(OUT))
