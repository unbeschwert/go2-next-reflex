"""Unit and HTTP tests. They never contact the real board, robot or cluster (unreachable test addresses)."""
from __future__ import annotations

import json
import shlex
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from durance_dash import board as B
from durance_dash import models as M
from durance_dash.config import Config, ConfigError, ClusterCfg, load, project_file, sim_weights, validate
from durance_dash.dataset import NAME_RE, PATH_RE
from durance_dash.security import Users, check_new_credentials
from durance_dash.settings_svc import load_schema_module
from durance_dash.sshrun import q
from durance_dash.training import Training

GO2 = project_file(Config().paths, "bridge_settings.py").parent   # go2_viz/ or the repo's kria/
BS = load_schema_module(GO2)


@pytest.fixture()
def cfg(tmp_path) -> Config:
    c = Config()
    c.paths.state = str(tmp_path / "state")
    c.path = str(tmp_path / "config.toml")          # private clusters.toml goes next to it (tmp)
    c.paths.data = str(tmp_path / "data")
    c.board.host = "127.0.0.2"            # nothing listens there: board calls fail fast
    c.board.ssh_port = 9
    c.clusters = [ClusterCfg(id="fake", ssh_host="no-such-host.invalid", control_path=str(tmp_path / "cm-%C"),
                             workdir="~/x", slurm_script="t.slurm")]
    return c


@pytest.fixture()
def client(cfg):
    from durance_dash.app import create_app
    with TestClient(create_app(cfg), base_url="http://127.0.0.1:8800") as c:
        yield c


def login(c) -> dict:
    r = c.post("/api/auth/setup", json={"username": "op", "password": "a-long-pass-123"})
    assert r.status_code == 200, r.text
    return {"X-CSRF-Token": r.json()["csrf"]}


# ---------------------------------------------------------------- config
def test_config_rejects_lan_without_opt_in(cfg):
    cfg.server.host = "0.0.0.0"
    with pytest.raises(ConfigError):
        validate(cfg)
    cfg.server.allow_lan = True
    validate(cfg)


def test_config_robot_limits_and_domains(cfg):
    cfg.robot.max_vx = 5
    with pytest.raises(ConfigError):
        validate(cfg)
    cfg.robot.max_vx = 0.5
    cfg.sim.ros_domain = 0                 # the robot's domain: never for the simulator
    with pytest.raises(ConfigError):
        validate(cfg)


def test_config_file_unknown_key(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[server]\nprot = 1\n")
    with pytest.raises(ConfigError):
        load(str(p))


# ---------------------------------------------------------------- security
def test_password_policy():
    for user, pw in (("op", "short"), ("op", "aaaaaaaaaaaa"), ("op", "op")):
        with pytest.raises(Exception):
            check_new_credentials(user, pw)
    with pytest.raises(Exception):
        check_new_credentials("bad user!", "a-long-pass-123")
    check_new_credentials("op", "a-long-pass-123")


def test_users_hash_and_file_mode(tmp_path):
    u = Users(tmp_path / "users.json")
    u.create("op", "a-long-pass-123")
    assert u.verify("op", "a-long-pass-123")
    assert not u.verify("op", "wrong-password-1")
    assert not u.verify("nobody", "a-long-pass-123")
    assert (tmp_path / "users.json").stat().st_mode & 0o077 == 0
    assert "a-long-pass-123" not in (tmp_path / "users.json").read_text()


def test_auth_flow_csrf_and_headers(client):
    assert client.get("/api/auth/state").json()["setup_needed"] is True
    assert client.get("/api/overview").status_code == 401
    h = login(client)
    assert client.post("/api/auth/setup", json={"username": "x2", "password": "a-long-pass-123"}).status_code == 409
    assert client.post("/api/estop/clear").status_code == 403                 # no CSRF token
    assert client.post("/api/estop/clear", headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert client.post("/api/estop/clear", headers=h).status_code == 200
    r = client.get("/api/health")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    cookie = client.cookies.jar
    assert any(c.name == "dd_session" for c in cookie)


def test_cross_origin_and_host_checks(client):
    h = login(client)
    assert client.post("/api/estop/clear", headers={**h, "Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "evil.example:8800"}).status_code == 421


def test_login_rate_limit(client):
    login(client)
    client.post("/api/auth/logout", headers={"X-CSRF-Token": client.get("/api/auth/state").json()["csrf"]})
    codes = [client.post("/api/auth/login", json={"username": "op", "password": "wrong-password-x"}).status_code
             for _ in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429


def test_unknown_api_and_bad_json(client):
    h = login(client)
    assert client.get("/api/nope").status_code == 404
    r = client.put("/api/settings", content=b"{not json", headers={**h, "Content-Type": "application/json"})
    assert r.status_code == 422


def test_audit_log_written(client):
    h = login(client)
    client.post("/api/estop/clear", headers=h)
    entries = client.get("/api/audit").json()["entries"]
    assert entries[0]["action"] == "estop.clear" and entries[0]["user"] == "op"


# ---------------------------------------------------------------- settings schema (shared with the bridge)
def test_settings_defaults_match_bridge_rules():
    d = BS.defaults("own")
    assert d["min_rate_o0"] == 0.5 and d["margin"] == 0.3 and d["stable_s"] == 0.2 and d["min_frames"] == 40
    assert BS.defaults("own", robot=True)["stable_s"] == 0.3
    assert BS.defaults("dataset")["min_rate_o3"] == 0.3
    assert {r["key"] for r in BS.schema("own")} == set(d)


def test_settings_validation():
    clean, err = BS.validate({"window_s": 9, "paused": "yes", "margin": float("nan"), "min_frames": 40.4,
                              "x": 1, "turn_rate": True})
    assert set(err) == {"window_s", "paused", "margin", "x", "turn_rate"}
    assert clean == {"min_frames": 40}


def test_settings_file_fallback(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{broken")
    values, errors = BS.load(str(p), "own")
    assert values == BS.defaults("own") and "_file" in errors
    p.write_text(json.dumps({"window_s": 0.3, "margin": 5}))
    values, errors = BS.load(str(p), "own")
    assert values["window_s"] == 0.3 and values["margin"] == 0.3 and "margin" in errors


# ---------------------------------------------------------------- shell quoting / training arguments
def test_quote_is_injection_safe():
    evil = "x; rm -rf / $(id) `id` 'a'"
    assert shlex.split(q(evil)) == [evil]


def test_training_args_whitelist(cfg):
    t = Training(cfg, audit=None)
    name, args = t.build_args({"name": "r11-a", "dataset": "own", "arch": "nb40", "preset": "leak",
                               "args": {"spike-reg": 3, "qat-bits": "8h"}})
    assert name == "r11-a" and "--learnable-leak" in args and args[args.index("--spike-reg") + 1] == "3.0"
    bad = [{"name": "A;B", "dataset": "own", "arch": "nb40"},
           {"name": "ok-name", "dataset": "x", "arch": "nb40"},
           {"name": "ok-name", "dataset": "own", "arch": "nb40", "args": {"zip": "/etc/passwd"}},
           {"name": "ok-name", "dataset": "own", "arch": "nb40", "args": {"epochs": 10_000}},
           {"name": "ok-name", "dataset": "own", "arch": "nb40", "args": {"epochs": 2.5}},
           {"name": "ok-name", "dataset": "own", "arch": "nb40", "args": {"qat-bits": "8h; id"}}]
    for body in bad:
        with pytest.raises(Exception):
            t.build_args(body)


# ---------------------------------------------------------------- board helpers
def test_parse_lock():
    other = B.parse_lock("laptop-2 (training session): 80x80 own-gesture test, started 07:48 board time")
    assert other["owner"] == "laptop-2" and other["since"] == "07:48" and not other["mine"]
    assert "80x80" in other["purpose"]
    mine = B.parse_lock("dashboard:op (Durance dashboard): deploy x, started 10:01 board time")
    assert mine["mine"] and mine["owner"] == "dashboard:op"
    assert B.parse_lock("") is None and B.parse_lock(None) is None


def test_autostart_conf_roundtrip():
    c = B.validate_conf({"mode": "robot-dry", "classes": "own", "start_delay_s": 30})
    assert B.parse_conf(B.conf_text(c)) == c
    for bad in ({"mode": "robot; reboot"}, {"start_delay_s": -1}, {"keep_pause": "1"}, {"evil": 1}):
        with pytest.raises(Exception):
            B.validate_conf(bad)


def test_board_scripts_compile():
    for p in (Path(B.__file__).parent / "board_scripts").glob("*.py"):
        compile(p.read_text(), str(p), "exec")
    assert "base64" in B.py_cmd("board_status.py")


# ---------------------------------------------------------------- models / dataset
def test_arch_value_counts():
    assert M.value_count(M.ARCHS["nb40"][1]) == 175191          # measured on the board
    assert M.value_count(M.ARCHS["best80"][1]) == 638337
    assert M.value_count(M.ARCHS["v3"][1]) == 88799


def test_model_describe(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("\n".join(["1"] * 175191))
    assert M.describe(p)["arch"] == "nb40"
    p.write_text("not weights")
    assert M.describe(p) is None


def test_clip_name_patterns():
    for ok in ("gesture_04-follow-me-back-01.bin", "gesture_02-turn-left-360-fast-03.bin", "gesture_05-no-action-still-11.bin"):
        assert NAME_RE.match(ok)
    m = NAME_RE.match("gesture_02-turn-left-360-fast-03.bin")
    assert m.groups() == ("02", "turn-left-360", "fast", "03")
    for bad in ("../x/gesture_01-a-b-01.bin", "29/../gesture_01-a-b-01.bin", "29/gesture_01-a-b-01.bin/../x"):
        assert not PATH_RE.match(bad)


def _clip(path: Path, n: int = 50) -> None:
    ev = np.zeros(n, dtype=[("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
    ev["x"], ev["y"], ev["polarity"] = 100, 120, 1
    ev["timestamp"] = np.arange(n) * 20_000
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("!I", ev.nbytes) + ev.tobytes())


def test_dataset_play_relabel_reject(client, cfg):
    h = login(client)
    data = Path(cfg.paths.data)
    _clip(data / "29" / "gesture_01-heart-fast-01.bin")
    r = client.get("/api/dataset/clip", params={"source": "data", "path": "29/gesture_01-heart-fast-01.bin", "size": 40})
    assert r.status_code == 200 and r.json()["meta"]["events"] == 50 and r.json()["frames"] == 99
    assert client.get("/api/dataset/clip", params={"source": "data", "path": "../../etc/passwd"}).status_code == 422
    body = {"source": "data", "path": "29/gesture_01-heart-fast-01.bin", "cls": 2}
    r = client.post("/api/dataset/relabel", json=body, headers=h)
    assert r.status_code == 409 and "confirm" in r.json()
    r = client.post("/api/dataset/relabel", json={**body, "confirm": True}, headers=h)
    new = r.json()["path"]
    assert new.startswith("29/gesture_02-turn-left-360-fast-") and (data / new).exists()
    r = client.post("/api/dataset/reject", json={"source": "data", "path": new, "confirm": True}, headers=h)
    assert r.status_code == 200 and not (data / new).exists()


# ---------------------------------------------------------------- safety paths without a board
def test_board_unreachable_is_reported(client):
    login(client)
    ov = client.get("/api/overview").json()
    assert ov["board"]["reachable"] is False
    assert any(a["level"] == "danger" for a in ov["alerts"])


def test_estop_works_without_board(client):
    h = login(client)
    r = client.post("/api/estop", headers=h)
    assert r.status_code == 200 and r.json()["ok"]
    assert client.get("/api/overview").json()["safety"]["estop"] is True
    assert client.post("/api/robot/manual", json={"enabled": True}, headers=h).status_code == 409
    assert client.post("/api/robot/move", json={"vx": 1}, headers=h).status_code == 409


def test_manual_requires_link(client):
    h = login(client)
    r = client.post("/api/robot/manual", json={"enabled": True}, headers=h)
    assert r.status_code == 409                              # board unreachable: no robot link


def test_robot_move_rejects_non_numbers(client):
    h = login(client)
    assert client.post("/api/robot/move", json={"vx": "fast"}, headers=h).status_code == 422


# ---------------------------------------------------------------- RViz container
def test_rviz_vnc_websocket_requires_login_and_same_origin(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):                     # not logged in
        with client.websocket_connect("/api/rviz/vnc") as ws:
            ws.receive_bytes()
    login(client)
    with pytest.raises(WebSocketDisconnect):                     # logged in, but from another site
        with client.websocket_connect("/api/rviz/vnc", headers={"Origin": "http://evil.example"}) as ws:
            ws.receive_bytes()
    with pytest.raises(WebSocketDisconnect):                     # logged in, same origin, RViz not running
        with client.websocket_connect("/api/rviz/vnc", headers={"Origin": "http://127.0.0.1:8800"}) as ws:
            ws.receive_bytes()


def test_rviz_state_explains_missing_docker(client, monkeypatch):
    from durance_dash import rviz as R
    from durance_dash.sshrun import Result

    async def fake_docker(self, *args, **kw):
        return Result(1, "permission denied while trying to connect to the docker API", "", 0.0)
    monkeypatch.setattr(R.RViz, "docker", fake_docker)
    login(client)
    st = client.get("/api/rviz/state").json()
    assert st["docker"]["ok"] is False and "usermod" in st["docker"]["detail"] and st["vnc"] is None
    h = {"X-CSRF-Token": client.get("/api/auth/state").json()["csrf"]}
    assert client.post("/api/rviz/start", json={"source": "sim"}, headers=h).status_code == 503
    assert client.post("/api/rviz/start", json={"source": "x"}, headers=h).status_code == 422


# ---------------------------------------------------------------- clip + network spikes
OWN_MODEL = sim_weights(Config())          # models/board_ready/... or the repo's models/optimised_snn/...


@pytest.mark.skipif(not OWN_MODEL.exists(), reason="board model files not on this machine")
def test_clip_spikes_replay(client, cfg):
    cfg.paths.models = Config().paths.models
    login(client)
    _clip(Path(cfg.paths.data) / "29" / "gesture_02-turn-left-360-fast-01.bin", n=50)   # 1 s, like the dataset
    r = client.get("/api/dataset/spikes", params={"source": "data", "path": "29/gesture_02-turn-left-360-fast-01.bin"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"]["arch"] == "nb40" and j["model"]["classes"] == "own" and j["true"] == 1
    assert len(j["outputs"]) == j["frames"] == len(j["layer_spikes"]) == len(j["events_per_frame"])
    assert all(len(row) == 5 and set(row) <= {0, 1} for row in j["outputs"])
    assert sum(j["events_per_frame"]) == 50
    assert client.get("/api/dataset/spikes", params={"source": "data", "path": "../x.bin"}).status_code == 422
    assert client.get("/api/dataset/spikes", params={"source": "data", "path": "29/gesture_02-turn-left-360-fast-01.bin",
                                                     "model": "nope"}).status_code == 404


# ---------------------------------------------------------------- training machines (no built-in cluster)
def test_no_builtin_training_machine():
    assert Config().clusters == []


def test_no_machine_gives_clean_state(client, cfg):
    cfg.clusters.clear()
    client.app.state.services.training.reload()
    login(client)
    assert client.get("/api/train/hosts").json() == {"hosts": []}
    r = client.get("/api/train/jobs")
    assert r.status_code == 409 and "no training machine" in r.json()["detail"]
    assert client.get("/api/overview").json()["training"]["configured"] is False


def test_add_edit_remove_machine(client, cfg, tmp_path):
    h = login(client)
    m = {"id": "cloud-a100", "label": "Cloud A100", "kind": "direct", "host": "203.0.113.7", "user": "ubuntu",
         "port": 22, "workdir": "~/snn", "python": "python3"}
    r = client.post("/api/train/machines", json=m, headers=h)
    assert r.status_code == 200, r.text
    f = tmp_path / "clusters.toml"
    assert f.exists() and f.stat().st_mode & 0o077 == 0 and 'host = "203.0.113.7"' in f.read_text()
    hosts = client.get("/api/train/hosts").json()["hosts"]
    assert {"id": "cloud-a100", "label": "Cloud A100", "kind": "direct"}.items() <= next(x for x in hosts if x["id"] == "cloud-a100").items()
    assert "203.0.113" not in json.dumps(hosts)                       # the list never shows login details
    for bad in ({"host": "x; rm -rf /"}, {"user": "a b"}, {"workdir": "~/x; id"}, {"python": "python3 -c id"},
                {"port": 0}, {"kind": "slurm"}, {"identity_file": "/etc/shadow"}, {"evil": 1}):
        assert client.post("/api/train/machines", json={**m, **bad}, headers=h).status_code == 422, bad
    r = client.post("/api/train/machines/cloud-a100/delete", json={}, headers=h)
    assert r.status_code == 409 and "confirm" in r.json()
    assert client.post("/api/train/machines/cloud-a100/delete", json={"confirm": True}, headers=h).status_code == 200
    assert "203.0.113.7" not in f.read_text()


def test_generated_key_never_leaks_private_half(client):
    h = login(client)
    r = client.post("/api/train/keys", json={"name": "cloud-key"}, headers=h)
    assert r.status_code == 200 and r.json()["public"].startswith("ssh-ed25519 ")
    listing = json.dumps(client.get("/api/train/keys").json())
    assert "PRIVATE KEY" not in listing and "dashboard:cloud-key" in listing
    assert client.post("/api/train/keys", json={"name": "../x"}, headers=h).status_code == 422


def test_remote_errors_are_cleaned():
    from durance_dash.training import clean, reason
    t = clean("ssh: connect to host gpu7.example.org port 22: user@10.1.2.3 /home/me/.ssh/id_x failed")
    assert "example.org" not in t and "user@" not in t and "/home/me" not in t
    assert reason("kex_exchange_identification: Connection closed by remote host").startswith("connection refused")
    assert reason("Permission denied (publickey)") == "login refused"


# ---------------------------------------------------------------- repository layout
def test_project_file_repo_layout(tmp_path):
    from durance_dash.config import PathsCfg
    repo = tmp_path / "repo"
    for f in ("kria/bridge_settings.py", "ros2/rviz/go2.urdf", "training/optimised_snn/dataset/gestures.json"):
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text("x")
    p = PathsCfg(repo=str(repo), go2_viz="", project="")
    assert project_file(p, "bridge_settings.py") == repo / "kria" / "bridge_settings.py"
    assert project_file(p, "go2.urdf") == repo / "ros2" / "rviz" / "go2.urdf"
    assert project_file(p, "gestures.json") == repo / "training" / "optimised_snn" / "dataset" / "gestures.json"
    assert not project_file(p, "missing.py").exists()
    viz = tmp_path / "go2_viz"                      # the laptop layout's folder wins when it has the file
    viz.mkdir()
    (viz / "gestures.json").write_text("y")
    p.go2_viz = str(viz)
    assert project_file(p, "gestures.json") == viz / "gestures.json"
    assert project_file(p, "bridge_settings.py") == repo / "kria" / "bridge_settings.py"


def test_models_scan_repo_layout(tmp_path):
    n = M.value_count(M.ARCHS["nb40"][1])
    d = tmp_path / "optimised_snn" / "own_gestures"
    d.mkdir(parents=True)
    (d / "weights_r6b-reg3_056ee113.txt").write_text("1\n" * n)
    (d / "metrics_r6b-reg3.json").write_text(json.dumps({"classes": ["Heart", "Turn left", "Turn right", "none"],
                                                         "results": {"qat_q8h": {"val": 1.0, "test": 0.99}}}))
    (tmp_path / "optimised_snn" / "nb40_model.json").write_text("{}")
    ms = M.scan(tmp_path)
    assert len(ms) == 1
    m = ms[0]
    assert m["arch"] == "nb40" and m["classes"] == "own" and m["bits"] == "8h"
    assert m["path"] == "optimised_snn/own_gestures/weights_r6b-reg3_056ee113.txt"
    assert m["metrics"]["test"] == 0.99
