"""Durance AI dashboard: FastAPI application (API in docs/API.md, UI in static/)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, models as M
from .board import Board
from .config import Config, load
from .dataset import Dataset
from .board import parse_lock
from .errors import Confirm, need
from .events import Broker, Tasks
from .live import Live
from .machines import Machines
from .recording import Recorder
from .robot import Robot
from .rviz import RViz
from .security import COOKIE, HEADERS, Audit, Auth, RateLimit, Sessions, Users, client_key
from .settings_svc import Settings
from .sim import Sim
from .spikes import Spikes
from .training import Training

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_BODY = 2 * 1024 * 1024
log = logging.getLogger("durance")


class Services:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        sd = cfg.state_dir
        self.broker = Broker()
        self.tasks = Tasks(self.broker)
        self.audit = Audit(sd / "audit.jsonl")
        self.users = Users(sd / "users.json")
        self.sessions = Sessions(cfg.server.session_hours)
        self.auth = Auth(self.users, self.sessions, self.audit)
        self.login_limit = RateLimit(5, 60)
        self.board = Board(cfg, self.audit, self.broker)
        self.live = Live(cfg, self.broker, self.board)
        self.sim = Sim(cfg, self.broker, self.tasks)
        self.settings = Settings(cfg, self.board, self.sim, self.live, self.audit, self.broker)
        self.robot = Robot(cfg, self.board, self.sim, self.live, self.settings, self.audit, self.broker)
        self.recorder = Recorder(cfg, self.board, self.audit, self.broker)
        self.dataset = Dataset(cfg, self.audit)
        self.spikes = Spikes(cfg, self.dataset)
        cfg_path = cfg.path or str(Path.home() / ".config" / "durance-dashboard" / "config.toml")
        self.machines = Machines(cfg, cfg_path, on_change=lambda: self.training.reload())
        self.training = Training(cfg, self.audit, self.machines)
        self.rviz = RViz(cfg, self.sim, self.tasks, self.audit, self.broker)
        self.estop_active = False
        self.started = time.time()
        self.target_file = sd / "target.json"
        try:
            t = json.loads(self.target_file.read_text()).get("mode")
            if t in ("board", "sim"):
                self.live.source = t
        except (OSError, ValueError):
            pass


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load()
    S = Services(cfg)
    allowed_hosts = {f"{h}:{cfg.server.port}" for h in ("127.0.0.1", "localhost", "[::1]", cfg.server.host)}

    @asynccontextmanager
    async def lifespan(app):
        S.broker.bind(asyncio.get_running_loop())
        live_task = asyncio.create_task(S.live.run())
        log.info("Durance dashboard %s on http://%s:%d (state in %s)", __version__, cfg.server.host, cfg.server.port, cfg.state_dir)
        try:
            yield
        finally:
            live_task.cancel()
            for name, fn in (("robot", S.robot.shutdown), ("recorder", S.recorder.shutdown), ("sim", S.sim.shutdown),
                             ("training", S.training.shutdown), ("rviz", S.rviz.shutdown)):
                try:
                    await asyncio.wait_for(fn(), 10)        # robot first: StopMove before anything else
                except Exception as e:  # noqa: BLE001
                    log.warning("shutdown of %s: %r", name, e)

    app = FastAPI(title="Durance AI dashboard", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.services = S

    # ---------------------------------------------------------------- middleware / errors
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        if host not in allowed_hosts and not cfg.server.allow_lan:
            return JSONResponse({"detail": "unexpected Host header"}, status_code=421)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin.split("://", 1)[-1] != host:
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > MAX_BODY:
                return JSONResponse({"detail": "request too large"}, status_code=413)
        response = await call_next(request)
        for k, v in HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    @app.exception_handler(Confirm)
    async def confirm_handler(request: Request, exc: Confirm):
        return JSONResponse({"detail": exc.detail, "confirm": exc.question}, status_code=409, headers=HEADERS)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": f"internal error: {type(exc).__name__}"}, status_code=500, headers=HEADERS)

    async def body(request: Request) -> dict:
        if request.method in ("GET", "HEAD"):
            return {}
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise HTTPException(413, "request too large")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            raise HTTPException(422, "body is not valid JSON")
        if not isinstance(data, dict):
            raise HTTPException(422, "body must be a JSON object")
        return data

    user = S.auth.user

    async def board_change(u: str, purpose: str, confirm: bool, question: str | None = None) -> None:
        """Ask once for everything a board change implies (the action itself and taking over someone
        else's lock), then take the lock."""
        st = await S.board.raw_status(0)
        if not st.get("ssh"):
            raise HTTPException(503, f"board not available: {st.get('error')}")
        parts = [question] if question else []
        lock = parse_lock(st.get("lock"))
        if lock and not lock["mine"]:
            parts.append(f"The board is locked by {lock['owner']} ({lock['purpose'] or 'no purpose given'}"
                         f"{', since ' + lock['since'] if lock['since'] else ''}); they may be testing right now, "
                         "and the lock will be taken over.")
        if parts:
            need(confirm, " ".join(parts) + ("" if question and len(parts) == 1 else " Continue?"))
        await S.board.ensure_lock(u, purpose, True)

    # ---------------------------------------------------------------- auth
    def auth_state(request: Request, token: str | None = None) -> dict:
        s = S.sessions.get(token or request.cookies.get(COOKIE))
        return {"setup_needed": S.users.empty(), "logged_in": s is not None, "user": s["user"] if s else None,
                "csrf": s["csrf"] if s else None}

    def set_cookie(resp: Response, token: str) -> None:
        resp.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=cfg.server.secure_cookie,
                        max_age=int(cfg.server.session_hours * 3600), path="/")

    @app.get("/api/auth/state")
    async def get_auth_state(request: Request):
        return auth_state(request)

    @app.post("/api/auth/setup")
    async def setup(request: Request, b: dict = Depends(body)):
        S.login_limit.check("setup:" + client_key(request))
        if not S.users.empty():
            raise HTTPException(409, "already set up: log in")
        S.users.create(str(b.get("username", "")), str(b.get("password", "")))
        token, _ = S.sessions.create(b["username"])
        S.audit.log(b["username"], "auth.setup")
        resp = JSONResponse(auth_state(request, token))
        set_cookie(resp, token)
        return resp

    @app.post("/api/auth/login")
    async def login(request: Request, b: dict = Depends(body)):
        key = "login:" + client_key(request)
        S.login_limit.check(key)
        name, pw = str(b.get("username", "")), str(b.get("password", ""))
        ok = await asyncio.to_thread(S.users.verify, name, pw)
        if not ok:
            S.audit.log(name or "?", "auth.login", result="failed")
            raise HTTPException(401, "wrong username or password")
        S.login_limit.reset(key)
        token, _ = S.sessions.create(name)
        S.audit.log(name, "auth.login")
        resp = JSONResponse(auth_state(request, token))
        set_cookie(resp, token)
        return resp

    @app.post("/api/auth/logout")
    async def logout(request: Request, u: str = Depends(user)):
        S.sessions.drop(request.cookies.get(COOKIE))
        S.audit.log(u, "auth.logout")
        resp = JSONResponse({})
        resp.delete_cookie(COOKIE, path="/")
        return resp

    @app.post("/api/auth/password")
    async def password(request: Request, b: dict = Depends(body), u: str = Depends(user)):
        S.login_limit.check("pw:" + u)
        if not await asyncio.to_thread(S.users.verify, u, str(b.get("old", ""))):
            raise HTTPException(403, "current password is wrong")
        await asyncio.to_thread(S.users.set_password, u, str(b.get("new", "")))
        S.sessions.drop_user(u, keep=request.cookies.get(COOKIE))
        S.audit.log(u, "auth.password")
        return {}

    @app.get("/api/health")
    async def health():
        return {"ok": True, "version": __version__, "uptime_s": round(time.time() - S.started)}

    # ---------------------------------------------------------------- events, tasks, audit
    @app.get("/api/events")
    async def events(request: Request, u: str = Depends(user)):
        return StreamingResponse(S.broker.stream(request.is_disconnected), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/tasks")
    async def tasks(u: str = Depends(user)):
        return {"tasks": S.tasks.all()}

    @app.get("/api/tasks/{tid}")
    async def task(tid: str, u: str = Depends(user)):
        t = S.tasks.get(tid)
        if t is None:
            raise HTTPException(404, "no such task")
        return t.view()

    @app.get("/api/audit")
    async def audit(limit: int = 200, u: str = Depends(user)):
        return {"entries": S.audit.tail(max(1, min(limit, 2000)))}

    # ---------------------------------------------------------------- overview, target, E-STOP
    @app.get("/api/overview")
    async def overview(u: str = Depends(user)):
        board = await S.board.summary(max_age=2.0 if S.live.source == "board" else 15.0)
        live = S.live.summary()
        alerts = []
        if board and S.live.source == "board":            # board warnings only matter when it is the target
            for w in board.get("warnings", []):
                alerts.append({"level": "warn", "text": w})
            lock = board.get("lock")
            if lock and not lock["mine"]:
                alerts.append({"level": "warn", "text": f"Board locked by {lock['owner']} ({lock['purpose'] or '-'})"})
            if not board["ssh"] and S.live.source == "board":
                alerts.append({"level": "danger", "text": f"Board not available: {board.get('error')}"})
        if S.estop_active:
            alerts.insert(0, {"level": "danger", "text": "E-STOP active: gesture control is paused"})
        return {"target": {"mode": S.live.source, "label": f"Kria board {cfg.board.host} (SSH)" if S.live.source == "board"
                           else f"Local simulator (ROS domain {cfg.sim.ros_domain})"},
                "board": board, "robot": {"manual": S.robot.manual, "link": await S.robot.current_link() if S.robot.manual is False else S.robot.link},
                "live": live, "sim": S.sim.state() if S.live.source == "sim" else None,
                "training": S.training.status(),
                "safety": {"estop": S.estop_active, "paused": live.get("paused")}, "alerts": alerts}

    @app.post("/api/target")
    async def target(b: dict = Depends(body), u: str = Depends(user)):
        mode = b.get("mode")
        if mode not in ("board", "sim"):
            raise HTTPException(422, "mode must be board or sim")
        if S.robot.manual:
            raise HTTPException(409, "turn manual control off before switching the target")
        S.live.set_source(mode)
        S.target_file.write_text(json.dumps({"mode": mode}))
        S.audit.log(u, "target", {"mode": mode})
        return await overview(u)

    @app.post("/api/estop")
    async def estop(u: str = Depends(user)):
        S.estop_active = S.robot.estop_active = True
        actions: list[str] = []

        async def guarded(name, coro):
            try:
                res = await asyncio.wait_for(coro, 8)
                if res:
                    actions.append(res)
            except Exception as e:  # noqa: BLE001 - one failed path must not stop the others
                actions.append(f"{name} failed: {getattr(e, 'detail', None) or e}")

        async def pause():
            await S.settings.set_paused(u, True, "E-STOP", bypass_lock=True)
            return "gesture control paused (bridge aborts turns/follow within 0.2 s)"

        async def direct_stop():
            if S.live.source == "sim":
                return None
            st = await S.board.raw_status()
            mode = str(st.get("mode", ""))
            out = [await S.board.send_stop(robot=True)] if (st.get("helpers") or {}).get("robot_link") else []
            if mode == "test":
                out.append(await S.board.send_stop(robot=False))
            return "; ".join(out) or None

        await asyncio.gather(guarded("teleop stop", S.robot.estop()), guarded("pause", pause()),
                             guarded("StopMove", direct_stop()))
        if S.robot.manual:
            S.robot.manual = False
            await guarded("manual off", S.robot.shutdown())
            actions.append("manual control switched off")
        S.audit.log(u, "estop", {"actions": actions})
        S.broker.publish("alert", {"level": "danger", "text": f"E-STOP by {u}: " + "; ".join(actions)})
        return {"ok": True, "actions": actions}

    @app.post("/api/estop/clear")
    async def estop_clear(u: str = Depends(user)):
        S.estop_active = S.robot.estop_active = False
        S.audit.log(u, "estop.clear")
        S.broker.publish("alert", {"level": "info", "text": f"E-STOP cleared by {u} (gestures stay paused until resumed)"})
        return {"ok": True}

    # ---------------------------------------------------------------- board
    @app.get("/api/board/status")
    async def board_status(u: str = Depends(user)):
        s = await S.board.summary()
        raw = await S.board.raw_status()
        s.update(time=(raw.get("health") or {}).get("time"), ros_domain=(raw.get("bridge") or {}).get("ros_domain"),
                 input_size=(raw.get("server_config") or {}).get("input"), scaling=(raw.get("server_config") or {}).get("scaling"),
                 weights_md5=(raw.get("model") or {}).get("md5"), spleat_service=raw.get("spleat_service"),
                 camera_config=raw.get("camera_config"))
        return s

    @app.post("/api/board/lock")
    async def board_lock(b: dict = Depends(body), u: str = Depends(user)):
        return await S.board.lock_action(u, str(b.get("action")), str(b.get("purpose") or "dashboard session"),
                                         bool(b.get("confirm")))

    @app.post("/api/board/inference")
    async def board_inference(b: dict = Depends(body), u: str = Depends(user)):
        action = str(b.get("action"))
        q = ("Robot mode: the REAL Go2 will move when it sees gestures (LiDAR obstacle check on). Keep the area "
             "clear and the E-STOP at hand. Start?") if action == "robot" else None
        await board_change(u, f"inference {action}", bool(b.get("confirm")), q)
        t = S.tasks.start("board.inference", f"Board: inference {action}", u,
                          lambda logf: S.board.inference(u, action, logf), exclusive="board")
        return {"task_id": t.id}

    @app.get("/api/board/logs")
    async def board_logs(name: str = "bridge", lines: int = 200, u: str = Depends(user)):
        return await S.board.logs(name, lines)

    @app.get("/api/board/models")
    async def board_models(u: str = Depends(user)):
        raw = await S.board.raw_status() if S.live.source == "board" else {}
        md5 = (raw.get("model") or {}).get("md5")
        ms = await asyncio.to_thread(M.scan, Path(cfg.paths.models).expanduser())
        for m in ms:
            m["deployed"] = m["md5"] == md5
        return {"models": ms, "board_values": (raw.get("model") or {}).get("values")}

    @app.post("/api/board/deploy")
    async def board_deploy(b: dict = Depends(body), u: str = Depends(user)):
        m = await asyncio.to_thread(M.find, Path(cfg.paths.models).expanduser(), str(b.get("model_id")))
        if m is None:
            raise HTTPException(404, "unknown model")
        classes = b.get("classes") or m.get("classes")
        if classes not in ("own", "dataset"):
            raise HTTPException(422, "classes: own or dataset (which gestures were these weights trained on?)")
        metrics = m.get("metrics") or {}
        score = f" (test {metrics['test']:.1%})" if metrics.get("test") is not None else ""
        await board_change(u, f"deploy {m['name']}", bool(b.get("confirm")),
                           f"Deploy {m['name']}{score} as the board's model ({classes} gestures)? The current weights "
                           "are backed up, inference restarts in test mode (robot mode must be started again by hand).")
        t = S.tasks.start("board.deploy", f"Deploy {m['name']}", u, lambda logf: S.board.deploy(u, m, classes, logf),
                          exclusive="board")
        return {"task_id": t.id}

    @app.post("/api/board/restore")
    async def board_restore(b: dict = Depends(body), u: str = Depends(user)):
        await board_change(u, "restore weights", bool(b.get("confirm")), "Put back the weights from before the last deploy?")
        t = S.tasks.start("board.restore", "Restore previous weights", u, lambda logf: S.board.restore(u, logf), exclusive="board")
        return {"task_id": t.id}

    @app.post("/api/board/software")
    async def board_software(b: dict = Depends(body), u: str = Depends(user)):
        await board_change(u, "install board software", bool(b.get("confirm")),
                           "Install the current gesture bridge, bridge_settings.py, lidar_follow.py and robot_link.sh "
                           "from this laptop on the board? The old files are backed up. Restart inference afterwards.")
        t = S.tasks.start("board.software", "Install board software", u, lambda logf: S.board.install_software(u, logf),
                          exclusive="board")
        return {"task_id": t.id}

    @app.get("/api/board/autostart")
    async def board_autostart(u: str = Depends(user)):
        return await S.board.autostart()

    @app.post("/api/board/autostart")
    async def board_autostart_set(b: dict = Depends(body), u: str = Depends(user)):
        action, conf = str(b.get("action")), b.get("conf")
        q = None
        if action in ("install", "enable"):
            mode = (conf or {}).get("mode", "robot")
            q = ("SD-card mode: from the next power-on the board starts by itself"
                 + (" and the REAL Go2 moves on gestures (LiDAR obstacle check on)" if mode == "robot" else "")
                 + ". Disable it here, with systemctl, or by creating /root/DURANCE_AUTO_DISABLE. Continue?")
        await board_change(u, f"autostart {action}", bool(b.get("confirm")), q)
        t = S.tasks.start("board.autostart", f"Autostart: {action}", u,
                          lambda logf: S.board.autostart_action(u, action, conf, logf), exclusive="board")
        return {"task_id": t.id}

    # ---------------------------------------------------------------- live + settings
    @app.get("/api/live/camera")
    async def camera(request: Request, u: str = Depends(user)):
        return StreamingResponse(S.live.camera(request.is_disconnected),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/live/snapshot")
    async def snapshot(u: str = Depends(user)):
        return Response(await S.live.snapshot(), media_type="image/jpeg")

    @app.get("/api/settings")
    async def settings_get(u: str = Depends(user)):
        return await S.settings.get()

    @app.put("/api/settings")
    async def settings_put(b: dict = Depends(body), u: str = Depends(user)):
        values = b.get("values")
        if isinstance(values, dict) and values.get("paused") is False and S.estop_active:
            raise HTTPException(409, "E-STOP is active: clear it first, then resume gestures")
        return await S.settings.put(u, values, bool(b.get("confirm")))

    @app.post("/api/settings/reset")
    async def settings_reset(b: dict = Depends(body), u: str = Depends(user)):
        key = b.get("key")
        if key == "paused" and S.estop_active:
            raise HTTPException(409, "E-STOP is active: clear it first")
        return await S.settings.reset(u, key, bool(b.get("confirm")))

    # ---------------------------------------------------------------- simulation
    @app.get("/api/sim/state")
    async def sim_state(u: str = Depends(user)):
        return S.sim.state()

    @app.post("/api/sim/start")
    async def sim_start(u: str = Depends(user)):
        S.audit.log(u, "sim.start")
        return await S.sim.start(u)

    @app.post("/api/sim/stop")
    async def sim_stop(u: str = Depends(user)):
        if S.robot.manual and S.robot.link == "sim":
            await S.robot.set_manual(u, False, False)
        S.audit.log(u, "sim.stop")
        return await S.sim.stop()

    @app.post("/api/sim/person")
    async def sim_person(b: dict = Depends(body), u: str = Depends(user)):
        return await S.sim.set_person(str(b.get("gesture")))

    @app.post("/api/sim/rviz")
    async def sim_rviz(b: dict = Depends(body), u: str = Depends(user)):
        S.audit.log(u, "sim.rviz", b)
        return await S.sim.rviz(str(b.get("action")), str(b.get("source", "sim")))

    @app.post("/api/sim/view")
    async def sim_view(b: dict = Depends(body), u: str = Depends(user)):
        return await S.sim.set_view(str(b.get("source")))

    @app.post("/api/sim/reset")
    async def sim_reset(u: str = Depends(user)):
        return await S.sim.reset()

    @app.get("/api/sim/model")
    async def sim_model(u: str = Depends(user)):
        return S.sim.model()

    # ---------------------------------------------------------------- RViz container
    @app.get("/api/rviz/state")
    async def rviz_state(u: str = Depends(user)):
        return await S.rviz.state()

    @app.post("/api/rviz/start")
    async def rviz_start(b: dict = Depends(body), u: str = Depends(user)):
        return await S.rviz.start(u, str(b.get("source", "sim")))

    @app.post("/api/rviz/stop")
    async def rviz_stop(u: str = Depends(user)):
        return await S.rviz.stop(u)

    @app.post("/api/rviz/build")
    async def rviz_build(u: str = Depends(user)):
        return {"task_id": await S.rviz.build(u)}

    @app.websocket("/api/rviz/vnc")
    async def rviz_vnc(ws: WebSocket):
        # same checks as HTTP: known Host, same Origin, logged-in session (cookie)
        host = ws.headers.get("host", "")
        origin = ws.headers.get("origin", "")
        if (host not in allowed_hosts and not cfg.server.allow_lan) or (origin and origin.split("://", 1)[-1] != host) \
                or S.sessions.get(ws.cookies.get(COOKIE)) is None:
            await ws.close(code=1008, reason="not allowed")
            return
        await S.rviz.proxy(ws)

    # ---------------------------------------------------------------- manual control
    @app.get("/api/robot/state")
    async def robot_state(u: str = Depends(user)):
        v = S.robot.view()
        if not S.robot.manual:
            v["link"] = await S.robot.current_link()
        return v

    @app.get("/api/robot/commands")
    async def robot_commands(u: str = Depends(user)):
        return S.robot.commands()

    @app.post("/api/robot/manual")
    async def robot_manual(b: dict = Depends(body), u: str = Depends(user)):
        if b.get("enabled") and S.estop_active:
            raise HTTPException(409, "E-STOP is active: clear it first")
        return await S.robot.set_manual(u, bool(b.get("enabled")), bool(b.get("confirm")))

    @app.post("/api/robot/command")
    async def robot_command(b: dict = Depends(body), u: str = Depends(user)):
        if S.estop_active and b.get("name") != "stop":
            raise HTTPException(409, "E-STOP is active")
        return await S.robot.command(u, str(b.get("name")), bool(b.get("confirm")))

    @app.post("/api/robot/move")
    async def robot_move(b: dict = Depends(body), u: str = Depends(user)):
        if S.estop_active:
            raise HTTPException(409, "E-STOP is active")
        try:
            vx, vy, wz = (float(b.get(k, 0.0)) for k in ("vx", "vy", "wz"))
        except (TypeError, ValueError):
            raise HTTPException(422, "vx, vy, wz must be numbers")
        return await S.robot.move(vx, vy, wz)

    @app.post("/api/robot/stop")
    async def robot_stop(u: str = Depends(user)):
        return await S.robot.stop()

    # ---------------------------------------------------------------- recording
    @app.get("/api/record/options")
    async def record_options(u: str = Depends(user)):
        return S.recorder.options()

    @app.get("/api/record/sessions")
    async def record_sessions(u: str = Depends(user)):
        return await asyncio.to_thread(S.recorder.sessions)

    @app.post("/api/record/start")
    async def record_start(b: dict = Depends(body), u: str = Depends(user)):
        if S.live.source != "board":
            raise HTTPException(409, "recording needs the real board (switch the target to Board)")
        return await S.recorder.start(u, b)

    @app.get("/api/record/state")
    async def record_state(u: str = Depends(user)):
        return S.recorder.state

    @app.post("/api/record/control")
    async def record_control(b: dict = Depends(body), u: str = Depends(user)):
        return await S.recorder.control(u, str(b.get("action")))

    @app.get("/api/record/preview")
    async def record_preview(u: str = Depends(user)):
        return Response(S.recorder.preview_jpeg(), media_type="image/jpeg")

    @app.post("/api/record/cut")
    async def record_cut(b: dict = Depends(body), u: str = Depends(user)):
        session = "%02d" % int(b.get("session")) if str(b.get("session", "")).isdigit() else ""
        best, recut = int(b.get("best") or 0), bool(b.get("recut"))
        if not 0 <= best <= 1000:
            raise HTTPException(422, "best: 0..1000")
        t = S.tasks.start("record.cut", f"Cut session {session} into clips", u,
                          lambda logf: S.recorder.cut(u, session, best, recut, logf), exclusive="cut")
        return {"task_id": t.id}

    # ---------------------------------------------------------------- dataset
    @app.get("/api/dataset/sources")
    async def ds_sources(u: str = Depends(user)):
        return await asyncio.to_thread(S.dataset.sources)

    @app.get("/api/dataset/summary")
    async def ds_summary(source: str = "data", u: str = Depends(user)):
        return await asyncio.to_thread(S.dataset.summary, source)

    @app.get("/api/dataset/clips")
    async def ds_clips(source: str = "data", session: str | None = None, cls: str | None = None, offset: int = 0,
                       limit: int = 100, u: str = Depends(user)):
        return await asyncio.to_thread(S.dataset.clips, source, session, cls, max(0, offset), limit)

    @app.get("/api/dataset/clip")
    async def ds_clip(source: str, path: str, size: int = 80, u: str = Depends(user)):
        return await asyncio.to_thread(S.dataset.clip, source, path, size)

    @app.get("/api/dataset/spikes")
    async def ds_spikes(source: str, path: str, model: str | None = None, u: str = Depends(user)):
        return await asyncio.wait_for(asyncio.to_thread(S.spikes.run, source, path, model or None), 120)

    @app.get("/api/dataset/spike-models")
    async def ds_spike_models(u: str = Depends(user)):
        return {"models": await asyncio.to_thread(S.spikes.models)}

    @app.post("/api/dataset/relabel")
    async def ds_relabel(b: dict = Depends(body), u: str = Depends(user)):
        try:
            cls = int(b.get("cls"))
        except (TypeError, ValueError):
            raise HTTPException(422, "cls must be a class number")
        return await asyncio.to_thread(S.dataset.relabel, u, str(b.get("source")), str(b.get("path")), cls, bool(b.get("confirm")))

    @app.post("/api/dataset/reject")
    async def ds_reject(b: dict = Depends(body), u: str = Depends(user)):
        return await asyncio.to_thread(S.dataset.reject, u, str(b.get("source")), str(b.get("path")), bool(b.get("confirm")))

    # ---------------------------------------------------------------- training
    @app.get("/api/train/hosts")
    async def train_hosts(u: str = Depends(user)):
        return await S.training.hosts()

    @app.get("/api/train/options")
    async def train_options(u: str = Depends(user)):
        return S.training.options()

    @app.post("/api/train/submit")
    async def train_submit(b: dict = Depends(body), u: str = Depends(user)):
        return await S.training.submit(u, b)

    @app.get("/api/train/jobs")
    async def train_jobs(host: str | None = None, u: str = Depends(user)):
        return await S.training.jobs(host)

    @app.post("/api/train/cancel")
    async def train_cancel(b: dict = Depends(body), u: str = Depends(user)):
        return await S.training.cancel(u, b.get("host"), str(b.get("job_id")), bool(b.get("confirm")))

    @app.get("/api/train/log")
    async def train_log(job: str, host: str | None = None, lines: int = 200, u: str = Depends(user)):
        return await S.training.log(host, job, lines)

    @app.get("/api/train/runs")
    async def train_runs(host: str | None = None, u: str = Depends(user)):
        return await S.training.runs(host)

    @app.post("/api/train/fetch")
    async def train_fetch(b: dict = Depends(body), u: str = Depends(user)):
        host, run = b.get("host"), str(b.get("run"))
        t = S.tasks.start("train.fetch", f"Fetch {run}", u, lambda logf: S.training.fetch(u, host, run, logf))
        return {"task_id": t.id}

    # ---------------------------------------------------------------- training machines (added in the dashboard)
    @app.get("/api/train/machines/{mid}")
    async def machine_get(mid: str, u: str = Depends(user)):
        return S.machines.details(mid)

    @app.post("/api/train/machines")
    async def machine_save(b: dict = Depends(body), u: str = Depends(user)):
        c = await asyncio.to_thread(S.machines.save, b)
        S.audit.log(u, "train.machine.save", {"machine": c.id, "kind": c.kind})
        return S.machines.details(c.id)

    @app.post("/api/train/machines/{mid}/delete")
    async def machine_delete(mid: str, b: dict = Depends(body), u: str = Depends(user)):
        d = S.machines.details(mid)
        need(bool(b.get("confirm")), f"Remove the training machine '{d['label']}' from the dashboard? Nothing is "
                                     "deleted on the machine itself.")
        S.machines.delete(mid)
        S.audit.log(u, "train.machine.delete", {"machine": mid})
        return {"ok": True}

    @app.get("/api/train/keys")
    async def train_keys(u: str = Depends(user)):
        return {"keys": await asyncio.to_thread(S.machines.keys)}

    @app.post("/api/train/keys")
    async def train_key_new(b: dict = Depends(body), u: str = Depends(user)):
        k = await asyncio.to_thread(S.machines.generate_key, str(b.get("name", "")))
        S.audit.log(u, "train.key.generate", {"key": k["name"]})
        return k

    @app.get("/api/train/machines/{mid}/server-id")
    async def machine_server_id(mid: str, u: str = Depends(user)):
        r = await S.machines.scan_hostkey(mid)
        return {k: v for k, v in r.items() if not k.startswith("_")}

    @app.post("/api/train/machines/{mid}/server-id")
    async def machine_trust(mid: str, b: dict = Depends(body), u: str = Depends(user)):
        r = await S.machines.trust_hostkey(mid, str(b.get("fingerprint", "")))
        S.training.reload()
        S.audit.log(u, "train.machine.trust", {"machine": mid, "fingerprint": r["trusted"]})
        return r

    @app.post("/api/train/machines/{mid}/test")
    async def machine_test(mid: str, u: str = Depends(user)):
        return await S.training.test(mid)

    @app.post("/api/train/machines/{mid}/prepare")
    async def machine_prepare(mid: str, b: dict = Depends(body), u: str = Depends(user)):
        datasets = b.get("datasets") or []
        if not isinstance(datasets, list):
            raise HTTPException(422, "datasets: a list")
        d = S.machines.details(mid)
        need(bool(b.get("confirm")), f"Copy the training script, riverops"
             + (f" and {len(datasets)} dataset(s) (several GB)" if datasets else "")
             + f" to '{d['label']}'? Uploads can take a while.")
        t = S.tasks.start("train.prepare", f"Prepare {d['label']}", u,
                          lambda logf: S.training.prepare(u, mid, datasets, logf), exclusive=f"prepare-{mid}")
        return {"task_id": t.id}

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def api_404(rest: str):
        raise HTTPException(404, "no such API endpoint")

    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
    return app
