"""The Kria board over SSH: status, lock, inference modes, logs, model deploy, autostart, settings file."""
from __future__ import annotations

import asyncio
import base64
import json
import re
import socket
import time
from pathlib import Path

from fastapi import HTTPException

from . import models as M
from .config import project_file
from .errors import need
from .sshrun import SSH, q

SCRIPTS = Path(__file__).parent / "board_scripts"
BOARD_FILES = Path(__file__).parent.parent / "board"
LOGS = {"bridge": "/tmp/bridge.log", "obstacle": "/tmp/obstacle.log", "camera": "/tmp/despot.log",
        "viewer": "/tmp/snnview.log", "autostart": "/root/durance_auto.log"}
INFERENCE = {"start": "start", "start-headless": "start --headless", "robot": "robot", "robot-dry": "robot --dry-run",
             "stop": "stop"}
HELPER_FILES = ["live_gesture_bridge.py", "bridge_settings.py", "lidar_follow.py", "robot_link.sh"]
BACKUP_DIR = "/root/dashboard_backups"


def py_cmd(script: str, *args, env_script: str | None = None) -> str:
    """Remote command that runs a local helper script with python3 (nothing is copied to the board)."""
    code = base64.b64encode((SCRIPTS / script).read_bytes()).decode()
    run = f"exec python3 -u -c {q(f'import base64;exec(base64.b64decode({code!r}))')} {q(*args) if args else ''}"
    if env_script:
        return f"bash -c {q(f'source {q(env_script)} >/dev/null 2>&1 || exit 3; {run}')}"
    return run


def parse_lock(raw: str | None) -> dict | None:
    if not raw:
        return None
    m = re.match(r"^\s*([^\s(:]+(?::[^\s(:]+)?)\s*(?:\(([^)]*)\))?\s*:?\s*(.*)$", raw, re.S)
    owner = m.group(1) if m else raw[:40]
    rest = (m.group(3) if m else "").strip()
    since = re.search(r"started\s+([0-9:]+)", rest)
    purpose = re.sub(r",?\s*started\s+[0-9:]+.*$", "", rest).strip() or (m.group(2) if m and m.group(2) else "")
    return {"owner": owner, "purpose": purpose[:160], "since": since.group(1) if since else None,
            "mine": owner.startswith("dashboard:"), "raw": raw[:300]}


class Board:
    def __init__(self, cfg, audit, broker):
        b = cfg.board
        self.cfg, self.audit, self.broker = cfg, audit, broker
        self.ssh = SSH(f"{b.user}@{b.host}", b.control_path, b.ssh_port, b.connect_timeout_s)
        self.models_dir = Path(cfg.paths.models).expanduser()
        self._status: dict | None = None
        self._status_t = 0.0
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------- status
    def tcp_reachable(self, timeout: float = 1.2) -> tuple[bool, float]:
        t0 = time.monotonic()
        try:
            with socket.create_connection((self.cfg.board.host, self.cfg.board.ssh_port), timeout=timeout):
                return True, (time.monotonic() - t0) * 1000
        except OSError:
            return False, 0.0

    async def raw_status(self, max_age: float = 2.0) -> dict:
        async with self._lock:
            if self._status and time.monotonic() - self._status_t < max_age:
                return self._status
            reach, ms = await asyncio.get_running_loop().run_in_executor(None, self.tcp_reachable)
            st: dict = {"reachable": reach, "ping_ms": round(ms, 1), "ssh": False, "error": None}
            if reach:
                r = await self.ssh.run(py_cmd("board_status.py"), timeout=12)
                if r.ok:
                    try:
                        st.update(json.loads(r.out.strip().splitlines()[-1]), ssh=True)
                    except (ValueError, IndexError):
                        st["error"] = "status script returned no JSON"
                else:
                    st["error"] = (r.err or r.out).strip()[-300:] or f"ssh exit {r.rc}"
            else:
                st["error"] = f"board {self.cfg.board.host}:22 not reachable (cable / NetworkManager profile 'kria')"
            self._status, self._status_t = st, time.monotonic()
            return st

    def invalidate(self) -> None:
        self._status_t = 0.0

    async def summary(self, max_age: float = 2.0) -> dict:
        st = await self.raw_status(max_age)
        out = {"reachable": st["reachable"], "ssh": st["ssh"], "ping_ms": st["ping_ms"], "error": st.get("error"),
               "mode": st.get("mode", "unknown") if st["ssh"] else "unknown", "processes": st.get("processes", {}),
               "lock": parse_lock(st.get("lock")), "warnings": []}
        if not st["ssh"]:
            out.update(model=None, autostart=None)
            return out
        md = st.get("model") or {}
        arch = M.COUNTS.get(md.get("values"), (None, None))
        name = M.name_for_md5(self.models_dir, md.get("md5", "")) if md.get("md5") else None
        sc = st.get("server_config") or {}
        classes = st.get("model_classes")
        bits = "16-bit" if (name and "q16" in name) or arch[0] == "best80" else "8-bit"
        label = (f"{'Own gestures' if classes == 'own' else 'Durance AI team gestures'} · {arch[0] or 'unknown network'}"
                 + (f" ({sc.get('input')}×{sc.get('input')}, {bits})" if sc.get("input") else ""))
        out["model"] = {"md5": md.get("md5", "")[:8], "md5_full": md.get("md5"), "name": name, "label": label,
                        "classes": classes, "input": sc.get("input"), "arch": arch[0], "values": md.get("values")}
        a = st.get("autostart") or {}
        conf = parse_conf(a.get("conf") or "")
        out["autostart"] = {"installed": a.get("installed", False), "enabled": a.get("enabled") == "enabled",
                            "active": a.get("active"), "mode": conf.get("mode"), "disabled_by_file": a.get("disabled_by_file")}
        out["health"] = st.get("health")
        out["server_config"] = sc
        out["bridge"] = st.get("bridge")
        out["helpers"] = st.get("helpers")
        w = out["warnings"]
        if sc.get("input") and sc.get("scaling"):
            want = 4096 / (320 // sc["input"]) ** 2
            if abs(sc["scaling"] - want) > 1e-6:
                w.append(f"server config scaling is {sc['scaling']:g}, but {sc['input']}x{sc['input']} input needs "
                         f"{want:g} (despot sends block sums): the network sees wrongly scaled input")
        if arch[1] and sc.get("input") and arch[1] != sc["input"]:
            w.append(f"weights are a {arch[0]} ({arch[1]}x{arch[1]}) network but the server expects {sc['input']}x{sc['input']} input")
        if md.get("values") and not arch[0]:
            w.append(f"weights.txt has {md.get('values')} values: unknown architecture")
        h = st.get("helpers") or {}
        if not h.get("robot_link"):
            w.append("/root/robot_link.sh missing: robot mode cannot reach the Go2 (install board software)")
        if not h.get("bridge_settings_py"):
            w.append("bridge_settings.py not on the board: gesture-control settings and E-STOP pause are not applied by the bridge (install board software)")
        bs = st.get("bridge_status") or {}
        if bs.get("settings_errors"):
            w.append(f"bridge rejected settings: {bs['settings_errors']}")
        if (st.get("health") or {}).get("disk_free_mb") is not None and st["health"]["disk_free_mb"] < 100:
            w.append(f"board disk almost full ({st['health']['disk_free_mb']} MB free)")
        return out

    # ---------------------------------------------------------------- lock
    async def write_lock(self, user: str, purpose: str) -> None:
        purpose = re.sub(r"[^\w .,:/+-]", "", purpose)[:120] or "dashboard"
        text = f"dashboard:{user} (Durance dashboard): {purpose}, started $(date +%H:%M) board time"
        (await self.ssh.run(f'echo "{text}" > /root/BOARD_LOCK', timeout=10)).check("writing /root/BOARD_LOCK")
        self.invalidate()

    async def ensure_lock(self, user: str, purpose: str, confirm: bool) -> None:
        """Before any board change: take the free lock, keep ours, or (confirmed) take over someone else's."""
        st = await self.raw_status(max_age=0)
        if not st["ssh"]:
            raise HTTPException(503, f"board not available: {st.get('error')}")
        lock = parse_lock(st.get("lock"))
        if lock is None:
            await self.write_lock(user, purpose)
            self.audit.log(user, "board.lock.acquire", {"purpose": purpose})
            return
        if lock["mine"]:
            return
        need(confirm, f"The board is locked by {lock['owner']} ({lock['purpose'] or 'no purpose given'}"
                      f"{', since ' + lock['since'] if lock['since'] else ''}). They may be testing right now. "
                      "Take over the lock and continue?", f"board locked by {lock['owner']}")
        await self.write_lock(user, purpose)
        self.audit.log(user, "board.lock.takeover", {"from": lock["owner"], "purpose": purpose})
        self.broker.publish("alert", {"level": "warn", "text": f"{user} took over the board lock from {lock['owner']}"})

    async def lock_action(self, user: str, action: str, purpose: str, confirm: bool) -> dict:
        st = await self.raw_status(max_age=0)
        lock = parse_lock(st.get("lock"))
        if action == "acquire":
            if lock and not lock["mine"]:
                raise HTTPException(409, f"locked by {lock['owner']}: use take over")
            await self.write_lock(user, purpose)
        elif action == "takeover":
            await self.ensure_lock(user, purpose, confirm)
        elif action == "release":
            if lock and not lock["mine"]:
                raise HTTPException(409, f"the lock belongs to {lock['owner']}, not to the dashboard")
            if lock:
                (await self.ssh.run("rm -f /root/BOARD_LOCK", timeout=10)).check("removing the lock")
        else:
            raise HTTPException(422, "action must be acquire, release or takeover")
        self.audit.log(user, f"board.lock.{action}", {"purpose": purpose})
        self.invalidate()
        return {"lock": parse_lock((await self.raw_status(max_age=0)).get("lock"))}

    # ---------------------------------------------------------------- inference
    async def inference(self, user: str, action: str, log) -> dict:
        if action not in INFERENCE:
            raise HTTPException(422, f"action must be one of {sorted(INFERENCE)}")
        if action.startswith("robot") and not ((await self.raw_status(0)).get("helpers") or {}).get("robot_link"):
            raise HTTPException(409, "/root/robot_link.sh is missing on the board: robot mode would not reach the Go2")
        if action.startswith("robot"):        # robot mode needs the camera feed: start it first if needed
            st = await self.raw_status(0)
            if not (st.get("processes") or {}).get("camera"):
                log("camera feed not running: starting it first (inference.sh start --headless)")
                r = await self.ssh.run("bash /root/inference.sh start --headless", timeout=120)
                log(r.out + r.err)
                r.check("inference.sh start")
        log(f"$ bash /root/inference.sh {INFERENCE[action]}")
        r = await self.ssh.run(f"bash /root/inference.sh {INFERENCE[action]}", timeout=150)
        log(r.out.strip())
        if r.err.strip():
            log(r.err.strip())
        self.invalidate()
        self.audit.log(user, "board.inference", {"action": action}, "ok" if r.ok else f"exit {r.rc}")
        r.check(f"inference.sh {INFERENCE[action]}")
        return {"mode": (await self.summary())["mode"]}

    async def logs(self, name: str, lines: int) -> dict:
        lines = max(10, min(int(lines), 2000))
        if name == "server":
            cmd = f"journalctl -u spleat-server -n {lines} --no-pager 2>/dev/null || echo 'journalctl not available'"
        elif name in LOGS:
            cmd = f"tail -n {lines} {q(LOGS[name])} 2>/dev/null || echo '(no {name} log yet)'"
        else:
            raise HTTPException(422, f"name must be one of {sorted(LOGS) + ['server']}")
        r = await self.ssh.run(cmd, timeout=10)
        r.check("reading the log")
        return {"name": name, "lines": r.out.splitlines()[-lines:]}

    # ---------------------------------------------------------------- settings file
    async def read_settings(self) -> dict:
        r = await self.ssh.run(f"cat {q(self.cfg.board.settings_file)} 2>/dev/null || echo '{{}}'", timeout=8)
        r.check("reading the settings")
        try:
            v = json.loads(r.out or "{}")
            return v if isinstance(v, dict) else {}
        except ValueError:
            return {}

    async def write_settings(self, values: dict) -> None:
        data = json.dumps(values, indent=1, sort_keys=True).encode()
        (await self.ssh.put(data, self.cfg.board.settings_file, timeout=10)).check("writing the settings")

    async def send_stop(self, robot: bool) -> str:
        env = self.cfg.board.robot_env if robot else self.cfg.board.test_env
        cmd = f"bash -c {q(f'source {q(env)} >/dev/null 2>&1 || exit 3; timeout 6 python3 /root/go2_cmd.py stop')}"
        r = await self.ssh.run(cmd, timeout=10)
        return "StopMove sent ({})".format("Go2, domain 0" if robot else "board link, domain 42") if r.ok \
            else f"StopMove via go2_cmd.py failed (exit {r.rc})"

    # ---------------------------------------------------------------- deploy / restore
    async def deploy(self, user: str, model: dict, classes: str, log) -> dict:
        if classes not in ("own", "dataset"):
            raise HTTPException(422, "classes must be own or dataset")
        st = await self.raw_status(0)
        cur = st.get("model") or {}
        cur_arch = M.COUNTS.get(cur.get("values"), (None, None))[0]
        if model["values"] != cur.get("values"):
            raise HTTPException(409, f"the board's firmware runs a {cur_arch or 'unknown'} network ({cur.get('values')} values); "
                                f"this model is {model['arch']} ({model['values']} values). Weights must match the firmware: "
                                "switch the firmware setup first.")
        path = self.models_dir / model["path"]
        data = path.read_bytes()
        prev_mode = st.get("mode")
        ts = time.strftime("%Y%m%d-%H%M%S")
        log(f"backing up /root/weights.txt (md5 {cur.get('md5', '?')[:8]})")
        (await self.ssh.run(f"mkdir -p {BACKUP_DIR} && cp /root/weights.txt {BACKUP_DIR}/weights_{ts}.txt && "
                            f"cp /root/model_classes {BACKUP_DIR}/model_classes_{ts} 2>/dev/null; "
                            f"echo {ts} > {BACKUP_DIR}/LAST", timeout=30)).check("backup")
        await self._install_weights(data, model["md5"], classes, prev_mode, log)
        self.audit.log(user, "board.deploy", {"model": model["name"], "md5": model["md5"], "classes": classes, "backup": ts})
        return {"md5": model["md5"], "backup": ts}

    async def restore(self, user: str, log) -> dict:
        r = await self.ssh.run(f"cat {BACKUP_DIR}/LAST", timeout=8)
        if not r.ok or not r.out.strip():
            raise HTTPException(409, "no dashboard backup on the board")
        ts = r.out.strip()
        if not re.fullmatch(r"\d{8}-\d{6}", ts):
            raise HTTPException(500, "backup marker is corrupt")
        r = await self.ssh.run(f"cat {BACKUP_DIR}/weights_{ts}.txt", timeout=60)
        r.check("reading the backup")
        data = r.out.encode()
        cls = (await self.ssh.run(f"cat {BACKUP_DIR}/model_classes_{ts} 2>/dev/null || echo dataset", timeout=8)).out.strip()
        import hashlib
        md5 = hashlib.md5(data).hexdigest()
        st = await self.raw_status(0)
        await self._install_weights(data, md5, cls if cls in ("own", "dataset") else "dataset", st.get("mode"), log)
        self.audit.log(user, "board.restore", {"backup": ts, "md5": md5})
        return {"md5": md5, "backup": ts}

    async def _install_weights(self, data: bytes, md5: str, classes: str, prev_mode: str | None, log) -> None:
        log(f"uploading {len(data) / 1e6:.1f} MB")
        (await self.ssh.put(data, "/root/weights.txt.dd-new", timeout=120)).check("upload")
        r = await self.ssh.run("md5sum /root/weights.txt.dd-new", timeout=20)
        if not r.ok or not r.out.startswith(md5):
            await self.ssh.run("rm -f /root/weights.txt.dd-new")
            raise HTTPException(502, "uploaded file is corrupt (md5 mismatch); nothing changed")
        log("md5 verified; stopping inference and restarting the FPGA server")
        cmd = ("bash /root/inference.sh stop >/dev/null 2>&1; mv -f /root/weights.txt.dd-new /root/weights.txt && "
               f"echo {q(classes)} > /root/model_classes && systemctl restart spleat-server && sleep 3 && "
               "systemctl is-active spleat-server && md5sum /root/weights.txt")
        r = await self.ssh.run(cmd, timeout=90)
        log(r.out.strip() + r.err.strip())
        r.check("installing the weights")
        if md5 not in r.out:
            raise HTTPException(502, "the board reports a different weights.txt after the restart")
        if prev_mode in ("test", "robot", "robot-dry", "camera-only"):
            # never re-enter robot mode by itself: the operator does that explicitly
            log("restarting inference in test mode (RViz)" + (" - robot mode must be started again by hand"
                                                              if prev_mode.startswith("robot") else ""))
            r = await self.ssh.run("bash /root/inference.sh start --headless", timeout=150)
            log(r.out.strip()[-1500:])
        self.invalidate()

    # ---------------------------------------------------------------- board software (helpers)
    async def install_software(self, user: str, log) -> dict:
        src = {f: project_file(self.cfg.paths, f) for f in HELPER_FILES}
        missing = [f for f, p in src.items() if not p.exists()]
        if missing:
            raise HTTPException(500, f"missing on the laptop: {missing}")
        ts = time.strftime("%Y%m%d-%H%M%S")
        log(f"backing up the current files to {BACKUP_DIR}/software_{ts}/")
        (await self.ssh.run(f"mkdir -p {BACKUP_DIR}/software_{ts} && for f in {' '.join(HELPER_FILES)}; do "
                            f"[ -f /root/$f ] && cp /root/$f {BACKUP_DIR}/software_{ts}/; done; true", timeout=20)).check("backup")
        for f in HELPER_FILES:
            mode = "755" if f.endswith(".sh") else "644"
            (await self.ssh.put(src[f].read_bytes(), f"/root/{f}", mode=mode, timeout=60)).check(f"upload {f}")
            log(f"installed /root/{f}")
        r = await self.ssh.run("cd /root && python3 -m py_compile live_gesture_bridge.py bridge_settings.py lidar_follow.py "
                               "&& md5sum " + " ".join(HELPER_FILES), timeout=30)
        log(r.out.strip() + r.err.strip())
        r.check("checking the installed files")
        log("restart inference (Board page) to load the new bridge")
        self.invalidate()
        self.audit.log(user, "board.install_software", {"backup": ts})
        return {"backup": ts}

    # ---------------------------------------------------------------- autostart (SD card mode)
    async def autostart(self) -> dict:
        st = await self.raw_status(0)
        a = st.get("autostart") or {}
        return {"installed": a.get("installed", False), "enabled": a.get("enabled") == "enabled",
                "active": a.get("active"), "disabled_by_file": a.get("disabled_by_file"),
                "conf": parse_conf(a.get("conf") or "") or default_conf()}

    async def autostart_action(self, user: str, action: str, conf: dict | None, log) -> dict:
        if action == "install":
            c = validate_conf(conf or {})
            (await self.ssh.put((BOARD_FILES / "durance_auto.sh").read_bytes(), "/root/durance_auto.sh", mode="755")).check("upload")
            (await self.ssh.put((BOARD_FILES / "durance-auto.service").read_bytes(),
                                "/etc/systemd/system/durance-auto.service", mode="644")).check("upload unit")
            (await self.ssh.put(conf_text(c).encode(), "/root/durance_auto.conf", mode="644")).check("upload conf")
            log("installed /root/durance_auto.sh, /root/durance_auto.conf, durance-auto.service")
            cmd = "systemctl daemon-reload && systemctl enable durance-auto.service && rm -f /root/DURANCE_AUTO_DISABLE"
        elif action == "enable":
            if conf:
                (await self.ssh.put(conf_text(validate_conf(conf)).encode(), "/root/durance_auto.conf")).check("upload conf")
            cmd = "systemctl enable durance-auto.service && rm -f /root/DURANCE_AUTO_DISABLE"
        elif action == "disable":
            cmd = "systemctl disable durance-auto.service"
        elif action == "uninstall":
            cmd = ("systemctl disable durance-auto.service 2>/dev/null; rm -f /etc/systemd/system/durance-auto.service "
                   "/root/durance_auto.sh; systemctl daemon-reload")
        else:
            raise HTTPException(422, "action must be install, enable, disable or uninstall")
        r = await self.ssh.run(cmd, timeout=30)
        log(r.out.strip() + r.err.strip())
        r.check(f"autostart {action}")
        self.audit.log(user, f"board.autostart.{action}", {"conf": conf})
        self.invalidate()
        return await self.autostart()


AUTO_KEYS = {"mode": ("robot", "robot-dry", "test"), "classes": ("own", "dataset")}


def default_conf() -> dict:
    return {"mode": "robot", "classes": "own", "start_delay_s": 20, "wait_go2_s": 120, "keep_pause": False}


def validate_conf(c: dict) -> dict:
    out = default_conf()
    for k, v in c.items():
        if k in AUTO_KEYS:
            if v not in AUTO_KEYS[k]:
                raise HTTPException(422, f"autostart {k} must be one of {AUTO_KEYS[k]}")
            out[k] = v
        elif k == "start_delay_s":
            if not isinstance(v, (int, float)) or not 0 <= v <= 300:
                raise HTTPException(422, "start_delay_s must be 0..300")
            out[k] = int(v)
        elif k == "wait_go2_s":
            if not isinstance(v, (int, float)) or not 10 <= v <= 900:
                raise HTTPException(422, "wait_go2_s must be 10..900")
            out[k] = int(v)
        elif k == "keep_pause":
            if not isinstance(v, bool):
                raise HTTPException(422, "keep_pause must be true or false")
            out[k] = v
        else:
            raise HTTPException(422, f"unknown autostart setting {k}")
    return out


def conf_text(c: dict) -> str:
    return ("# written by the Durance dashboard; read (not sourced) by /root/durance_auto.sh\n"
            f"MODE={c['mode']}\nCLASSES={c['classes']}\nSTART_DELAY_S={c['start_delay_s']}\n"
            f"WAIT_GO2_S={c['wait_go2_s']}\nKEEP_PAUSE={1 if c['keep_pause'] else 0}\n")


def parse_conf(text: str) -> dict:
    vals = dict(re.findall(r"^([A-Z_]+)=([\w.-]*)\s*$", text or "", re.M))
    if not vals:
        return {}
    out = default_conf()
    out["mode"] = vals.get("MODE", out["mode"])
    out["classes"] = vals.get("CLASSES", out["classes"])
    for k, key in (("start_delay_s", "START_DELAY_S"), ("wait_go2_s", "WAIT_GO2_S")):
        if vals.get(key, "").isdigit():
            out[k] = int(vals[key])
    out["keep_pause"] = vals.get("KEEP_PAUSE") == "1"
    return out
