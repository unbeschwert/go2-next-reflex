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
                             workdir="~/x")]
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


def test_training_args_our_configs(cfg):
    """The New run form: a YAML config of training/configs + only the changed values as --set overrides."""
    t = Training(cfg, audit=None)
    presets = t.options()["presets"]
    default = presets[0]
    assert default["id"] == "training/configs/motion_buddy_80_v4.yaml"
    same = dict(default["args"])                     # the form sends the config's own values back
    name, conf, overrides = t.build_args({"name": "v4-a", "preset": default["id"], "args": same})
    assert name == "v4-a" and conf["id"] == default["id"] and overrides == []
    _, conf, overrides = t.build_args({"name": "v4-b", "preset": default["id"],
                                       "args": {**same, "train.epochs": 8, "seed": 43, "data.val": "11",
                                                "model.neuron.learnable_leak": not default["args"]["model.neuron.learnable_leak"]}})
    assert "train.epochs=8" in overrides and "seed=43" in overrides and 'data.val=["11"]' in overrides
    assert len(overrides) == 4 and "11" in conf["sessions"] and conf["data_root"]
    bad = [{"name": "A;B", "preset": default["id"]},
           {"name": "ok-name", "preset": "../../etc/passwd.yaml"},
           {"name": "ok-name", "preset": default["id"], "args": {"data.root": "/etc"}},          # not offered
           {"name": "ok-name", "preset": default["id"], "args": {"train.epochs": 10_000}},
           {"name": "ok-name", "preset": default["id"], "args": {"train.epochs": 2.5}},
           {"name": "ok-name", "preset": default["id"], "args": {"train.optimizer": "adam; id"}},
           {"name": "ok-name", "preset": default["id"], "args": {"data.test": "13; rm -rf /"}},
           {"name": "ok-name", "preset": default["id"], "args": {"augment.affine.p": 1.5}},
           {"name": "ok-name", "preset": default["id"], "args": {"train.epochs": 3}}]   # < average_last: the loader says no
    for body in bad:
        with pytest.raises(Exception):
            t.build_args(body)


def test_training_configs_resolve_base(cfg):
    """Configs with base: are resolved by the pipeline's loader, so the form shows their real values."""
    t = Training(cfg, audit=None)
    by_id = {c["id"]: c for c in t.configs()}
    v4 = by_id["training/configs/motion_buddy_80_v4.yaml"]
    assert v4["args"]["train.epochs"] == 20 and v4["args"]["data.test"] == "13,14"
    assert all(c["args"] for c in by_id.values())


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
    m = {"id": "cloud-a100", "label": "Cloud A100", "host": "203.0.113.7", "user": "ubuntu",
         "port": 22, "workdir": "~/snn", "python": "python3"}
    r = client.post("/api/train/machines", json=m, headers=h)
    assert r.status_code == 200, r.text
    f = tmp_path / "clusters.toml"
    assert f.exists() and f.stat().st_mode & 0o077 == 0 and 'host = "203.0.113.7"' in f.read_text()
    hosts = client.get("/api/train/hosts").json()["hosts"]
    assert {"id": "cloud-a100", "label": "Cloud A100"}.items() <= next(x for x in hosts if x["id"] == "cloud-a100").items()
    assert "203.0.113" not in json.dumps(hosts)                       # the list never shows login details
    for bad in ({"host": "x; rm -rf /"}, {"user": "a b"}, {"workdir": "~/x; id"}, {"python": "python3 -c id"},
                {"port": 0}, {"kind": "slurm"}, {"slurm_script": "x.sbatch"}, {"uv": "uv; id"},
                    {"identity_file": "/etc/shadow"}, {"evil": 1}):
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


# ---------------------------------------------------------------- runs of our pipeline (training/snn)
from durance_dash import snnruns as R                     # noqa: E402

TINY = {"input": {"channels": 2, "height": 8, "width": 8},
        "nodes": [{"node_index": "conv_lif", "node_type": "ConvLIF2d", "upscaling": 1, "channels": 4, "kernel_size": 3,
                   "stride": 2, "padding": [1, 1, 1, 1]},
                  {"node_index": "conv_lif_1", "node_type": "ConvLIF2d", "upscaling": 1, "channels": 5, "kernel_size": 4,
                   "stride": 1, "padding": [0, 0, 0, 0]}],
        "edges": [{"source": "input", "target": "conv_lif"}, {"source": "conv_lif", "target": "conv_lif_1"}],
        "outputs": ["conv_lif_1"]}
TINY_VALUES = (2 * 4 * 9 + 4 + 2) + (4 * 5 * 16 + 5 + 2)


def make_run(runs: Path, name="tiny-run", classes=None, scaling=2.56, weights="Q1.7", firmware=True) -> Path:
    import yaml
    run, dep = runs / name, runs / name / "deploy"
    dep.mkdir(parents=True)
    classes = classes or ["sit", "lay-down", "hello", "dance", "no-action"]
    (run / "config.yaml").write_text(yaml.safe_dump({"name": "tiny", "data": {
        "root": "data/motion_buddy_dataset", "classes": classes, "train": ["01"], "val": ["11"], "test": ["13"],
        "downsampling": 40, "encoding": "mean"}}))
    (run / "metrics.json").write_text(json.dumps({"fixed_point": {"val_acc": 0.7, "test_acc": 0.6}}))
    off = sum(name.encode())                                         # each run its own weights (md5)
    (dep / "weights.txt").write_text("\n".join(str((i + off) % 7 - 3) for i in range(TINY_VALUES)))
    (dep / "model.json").write_text(json.dumps(TINY))
    (dep / "export_summary.json").write_text(json.dumps({"formats": {"weight_int": 1, "weight_frac": 7,
                                                                     "potential_int": 4, "potential_frac": 12}}))
    (dep / "spleat-server-config.yaml").write_text(
        f"inputs:\n  channels: 2\n  height: 8\n  width: 8\n  scaling: {scaling}\noutputs:\n  channels: 5\n  height: 1\n  width: 1\n")
    (dep / "metavision_camera.json").write_text(json.dumps({"frequency": 100.0, "downsampling": 40,
                                                            "biases": {"bias_diff": 51}}))
    if firmware:
        fw = run / "firmware-job7"
        fw.mkdir()
        (fw / "spleat.bit.bin").write_bytes(b"bitstream-" + name.encode())
        (fw / "spleat.dtbo").write_bytes(b"overlay")
        (fw / "job.json").write_text(json.dumps({"id": 7, "weights": weights, "potentials": "Q4.12"}))
        (fw / "job7-model.json").write_text(json.dumps(TINY))
    return run


def test_snnruns_network_and_firmware_match(tmp_path):
    make_run(tmp_path)
    [m] = R.scan(tmp_path)
    assert m["kind"] == "setup" and m["classes"] == "dataset" and m["input"] == 8 and m["values"] == TINY_VALUES
    assert m["network"]["layers"] == [(2, 4, 3, 2, 1), (4, 5, 4, 1, 0)] and m["network"]["output"] == [5, 1, 1]
    assert m["formats"]["weights"] == "Q1.7" and m["firmware"]["job"] == 7 and m["problems"] == []
    assert m["metrics"] == {"val": 0.7, "test": 0.6, "kind": "fixed_point"}
    assert "path" not in R.public(m) and "firmware_dir" not in R.public(m)
    ms = M.scan(tmp_path / "no-models", tmp_path)                    # the catalogue lists runs too
    assert [x["name"] for x in ms] == ["tiny-run"]
    assert M.by_md5(tmp_path / "no-models", m["md5"], tmp_path)["kind"] == "setup"


def test_snnruns_problems(tmp_path):
    def problems(name, **kw):
        d = tmp_path / name
        d.mkdir()
        make_run(d, name, **kw)
        return " ".join(R.scan(d)[0]["problems"])
    assert "no firmware" in problems("wrong-format", weights="Q2.6")        # firmware for another weight format
    assert "scaling is 4096" in problems("old-scaling", scaling=4096.0)      # exported before the scaling fix
    assert "classes" in problems("odd-classes", classes=["a", "b", "c", "d", "e"])
    assert "no firmware" in problems("no-firmware", firmware=False)
    assert problems("fine") == ""


def test_snnruns_reuse_firmware_of_the_same_network(tmp_path):
    make_run(tmp_path, "has-firmware")
    make_run(tmp_path, "same-network", firmware=False)
    make_run(tmp_path, "other-format", weights="Q2.6")
    by = {m["name"]: m for m in R.scan(tmp_path)}
    assert by["same-network"]["firmware"]["rel"] == "has-firmware/firmware-job7"
    assert by["other-format"]["firmware"]["rel"] == "has-firmware/firmware-job7"      # its own one is Q2.6
    assert by["has-firmware"]["firmware"]["rel"] == "has-firmware/firmware-job7"


def test_snnruns_not_a_chain():
    g = json.loads(json.dumps(TINY))
    g["edges"].append({"source": "input", "target": "conv_lif_1"})   # a skip connection
    assert R.network(g) is None and R.network(TINY)["size"] == 8


def test_setup_files_keep_board_sensor_settings(tmp_path):
    make_run(tmp_path)
    [m] = R.scan(tmp_path)
    files = R.setup_files(m, {"frequency": 50.0, "downsampling": 8, "noise_filtering_factor": 0.3,
                              "biases": {"bias_diff": 60}})
    cam = json.loads(files["metavision_camera.json"])
    assert cam["downsampling"] == 40 and cam["frequency"] == 100.0                    # the network's
    assert cam["biases"] == {"bias_diff": 60} and cam["noise_filtering_factor"] == 0.3  # the board's
    assert files["model_classes"] == b"dataset\n" and set(files) == set(B.SETUP_FILES)


class FakeBoard:
    """The board's file system and the commands deploy_setup / restore send (see board.py)."""

    def __init__(self, fail_install_from=None):
        import hashlib
        self.md5 = lambda b: hashlib.md5(b).hexdigest()
        self.fs = {path: f"old {name}".encode() for name, path in B.SETUP_FILES.items()}
        self.fail = fail_install_from
        self.cmds = []

    async def put(self, data, path, mode="644", timeout=60):
        self.fs[path] = data
        return _ok("")

    async def run(self, cmd, input=None, timeout=20):
        import re as _re
        self.cmds.append(cmd)
        if cmd.startswith("set -e; mkdir -p"):                                  # backup
            bak = _re.search(r"mkdir -p (\S+);", cmd).group(1)
            for name, path in B.SETUP_FILES.items():
                self.fs[f"{bak}/{name}"] = self.fs[path]
            return _ok("")
        m = _re.match(r"cd (\S+) && md5sum", cmd)
        if m:
            d = m.group(1)
            return _ok("\n".join(f"{self.md5(v)}  {k[len(d) + 1:]}" for k, v in self.fs.items() if k.startswith(d + "/")))
        m = _re.match(r"bash -s -- (\S+)", cmd)
        if m:
            d = m.group(1)
            if self.fail and self.fail in d:
                return _fail("the SPLEAT app did not load")
            for name, path in B.SETUP_FILES.items():
                self.fs[path] = self.fs[f"{d}/{name}"]
            return _ok("\n".join(f"{self.md5(self.fs[p])}  {p}" for p in B.SETUP_FILES.values()) + "\nSETUP INSTALLED")
        m = _re.match(r"cat (\S+)$", cmd)
        if m:
            return _ok(self.fs[m.group(1)].decode()) if m.group(1) in self.fs else _fail("no such file")
        m = _re.match(r"echo (\S+) > (\S+)", cmd)
        if m:
            self.fs[m.group(2)] = m.group(1).encode()
        return _ok("")


def _ok(out):
    from durance_dash.sshrun import Result
    return Result(0, out, "", 0.0)


def _fail(err):
    from durance_dash.sshrun import Result
    return Result(1, "", err, 0.0)


class _Audit:
    def __init__(self):
        self.entries = []

    def log(self, *a, **k):
        self.entries.append(a)


def _board(cfg, tmp_path, fake, status):
    import asyncio as _a
    cfg.paths.runs = str(tmp_path)
    b = B.Board(cfg, _Audit(), broker=None)
    b.ssh = fake

    async def raw(max_age=2.0):
        return status
    b.raw_status = raw
    return b, _a


def test_deploy_setup_installs_and_backs_up(cfg, tmp_path):
    make_run(tmp_path)
    [m] = R.scan(tmp_path)
    fake = FakeBoard()
    status = {"ssh": True, "mode": "stopped", "camera_config": {"downsampling": 8, "biases": {"bias_diff": 60}}}
    board, aio = _board(cfg, tmp_path, fake, status)
    old = dict(fake.fs)
    res = aio.run(board.deploy_setup("op", m, log=lambda _: None))
    want = R.setup_files(m, status["camera_config"])
    for name, path in B.SETUP_FILES.items():
        assert fake.fs[path] == want[name], name
        assert fake.fs[f"{B.BACKUP_DIR}/{res['backup']}/{name}"] == old[path]            # backup of the old setup
    assert fake.fs[f"{B.BACKUP_DIR}/LAST"] == res["backup"].encode()
    assert any(c.startswith("rm -rf") and "stage_" in c for c in fake.cmds)              # staging folder cleaned


def test_restore_puts_the_previous_setup_back(cfg, tmp_path):
    make_run(tmp_path)
    [m] = R.scan(tmp_path)
    fake = FakeBoard()
    board, aio = _board(cfg, tmp_path, fake, {"ssh": True, "mode": "stopped", "camera_config": None})
    old = {p: fake.fs[p] for p in B.SETUP_FILES.values()}
    aio.run(board.deploy_setup("op", m, log=lambda _: None))
    assert {p: fake.fs[p] for p in B.SETUP_FILES.values()} != old
    res = aio.run(board.restore("op", log=lambda _: None))
    assert res["backup"].startswith("setup_") and {p: fake.fs[p] for p in B.SETUP_FILES.values()} == old
    assert any("md5sum -c" in c for c in fake.cmds)                                     # backup checked first


def test_deploy_setup_rolls_back_when_the_firmware_does_not_load(cfg, tmp_path):
    from fastapi import HTTPException
    make_run(tmp_path)
    [m] = R.scan(tmp_path)
    fake = FakeBoard(fail_install_from="stage_")
    board, aio = _board(cfg, tmp_path, fake, {"ssh": True, "mode": "test", "camera_config": None})
    old = {p: fake.fs[p] for p in B.SETUP_FILES.values()}
    with pytest.raises(HTTPException) as e:
        aio.run(board.deploy_setup("op", m, log=lambda _: None))
    assert "previous setup is back" in e.value.detail
    assert {p: fake.fs[p] for p in B.SETUP_FILES.values()} == old
    assert f"{B.BACKUP_DIR}/LAST" not in fake.fs


def test_deploy_setup_refuses_runs_with_problems(cfg, tmp_path):
    from fastapi import HTTPException
    make_run(tmp_path, firmware=False)
    [m] = R.scan(tmp_path)
    board, aio = _board(cfg, tmp_path, FakeBoard(), {"ssh": True})
    with pytest.raises(HTTPException) as e:
        aio.run(board.deploy_setup("op", m, log=lambda _: None))
    assert e.value.status_code == 409 and "no firmware" in e.value.detail


def test_weights_deploy_checks_the_firmware_format(cfg, tmp_path):
    """Same value count but another weight format: refused (Q2.6 weights on a Q1.7 firmware)."""
    from fastapi import HTTPException
    run = make_run(tmp_path)
    [m] = R.scan(tmp_path)
    legacy = {"kind": "weights", "name": "team-q26", "arch": "tiny", "values": TINY_VALUES, "md5": "x" * 32,
              "path": str(run / "deploy" / "weights.txt"), "formats": {"weights": "Q2.6"}}
    bit = R.md5_of(run / "firmware-job7" / "spleat.bit.bin")
    board, aio = _board(cfg, tmp_path, FakeBoard(), {"ssh": True, "model": {"values": TINY_VALUES},
                                                     "firmware": {"bit_md5": bit}})
    with pytest.raises(HTTPException) as e:
        aio.run(board.deploy("op", legacy, "dataset", log=lambda _: None))
    assert e.value.status_code == 409 and "Q1.7" in e.value.detail and "Q2.6" in e.value.detail


def test_board_summary_names_run_and_firmware(cfg, tmp_path):
    import time as _t
    run = make_run(tmp_path)
    other = make_run(tmp_path, "other-net", weights="Q2.6")
    [m] = [x for x in R.scan(tmp_path) if x["name"] == "tiny-run"]
    cfg.paths.runs = str(tmp_path)
    b = B.Board(cfg, _Audit(), broker=None)
    base = {"reachable": True, "ssh": True, "ping_ms": 1.0, "model": {"md5": m["md5"], "values": m["values"]},
            "server_config": {"input": 8, "scaling": 2.56, "output_shape": [5, 1, 1]}, "model_classes": "dataset",
            "helpers": {"robot_link": True, "bridge_settings_py": True}}
    for bit_of, warn in ((run, False), (other, True)):
        b._status = {**base, "firmware": {"bit_md5": R.md5_of(bit_of / "firmware-job7" / "spleat.bit.bin")}}
        b._status_t = _t.monotonic()
        s = __import__("asyncio").run(b.summary(max_age=60))
        assert s["model"]["name"] == "tiny-run" and s["model"]["arch"] == "tiny" and s["model"]["input"] == 8
        assert s["firmware"]["known"] and s["firmware"]["weights"] == ("Q1.7" if not warn else "Q2.6")
        assert any("firmware" in w and "meaningless" in w for w in s["warnings"]) == warn
        assert not any("unknown architecture" in w for w in s["warnings"])


def test_spike_simulator_for_a_run(cfg, tmp_path):
    from durance_dash.spikes import Spikes
    make_run(tmp_path)
    cfg.paths.runs = str(tmp_path)
    [m] = R.scan(tmp_path)
    sp = Spikes(cfg, dataset=None)
    assert any(x["id"] == m["id"] for x in sp.models())
    sim, size = sp.simulator(m)
    assert size == 8 and (sim.w_shift, sim.b_shift, sim.in_shift) == (5, 5, 7)          # Q1.7 weights, Q4.12 potentials
    assert sim.step(np.ones((2, 8, 8), np.float32)).shape == (5,)


def _class_spikes():
    """kria/live_gesture_bridge.py's class_spikes (the bridge itself needs ROS 2)."""
    import ast
    src = (Path(__file__).resolve().parents[2] / "kria" / "live_gesture_bridge.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "class_spikes")
    ns = {"struct": struct}
    exec(compile(ast.Module([fn], []), "class_spikes", "exec"), ns)
    return ns["class_spikes"]


def test_bridge_reads_spatial_output_maps():
    f = _class_spikes()
    assert f(struct.pack("<5f", 0, 1, 0, 0, 2), 5) == [0, 1, 0, 0, 1]                    # 5 x 1 x 1
    two_by_two = [0] * 20
    two_by_two[2 * 4 + 3] = 1.0                                                          # class 2, position (1, 1)
    assert f(struct.pack("<20f", *two_by_two), 5) == [0, 0, 1, 0, 0]
    assert f(b"abc", 5) is None and f(struct.pack("<6f", *[0] * 6), 5) is None


def test_direct_start_command_records_the_job(cfg, tmp_path):
    """The start command for a direct machine, run by a real bash: the job file gets "<pid> <run>" in the work
    folder and the run's log is written (a '&&' chain before '&' would lose both)."""
    import asyncio as _a
    import subprocess as _sp
    from durance_dash.sshrun import Result
    work = tmp_path / "work"
    work.mkdir()
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/bash\necho \"fake uv $*\"\n")
    fake_uv.chmod(0o755)
    cfg.clusters = [ClusterCfg(id="local", host="localhost", workdir=str(work), uv=str(fake_uv),
                               control_path=str(tmp_path / "cm-%C"))]
    t = Training(cfg, audit=_Audit())

    class LocalBash:
        async def run(self, cmd, input=None, timeout=20):
            p = _sp.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, cwd=tmp_path)
            return Result(p.returncode, p.stdout, p.stderr, 0.0)
    t.ssh["local"] = LocalBash()

    async def no_check(conf, overrides):
        return ""

    async def workdir(c, ssh):
        return str(work)

    async def no_sync(ssh, home, log):
        return None
    t._local_check, t._workdir, t._sync_code = no_check, workdir, no_sync
    res = _a.run(t.submit("op", {"host": "local", "name": "bash-check", "preset": "training/configs/smoke.yaml",
                                 "args": {"seed": 5}, "dry_run": False, "confirm": True}))
    import time as _t
    _t.sleep(0.5)
    pid, run = (work / ".dd_jobs").read_text().split()
    assert pid == res["job_id"] and run.startswith("bash-check-")
    log = (work / "logs" / f"{run}.out").read_text()
    assert f"--run-dir runs/{run}" in log and "--set seed=5" in log and (work / "runs" / run).is_dir()


def test_old_clusters_toml_with_kind_and_slurm(tmp_path):
    """Training machines are SSH servers: an older clusters.toml loads (kind / slurm_script dropped),
    a Slurm cluster in it is skipped instead of stopping the dashboard."""
    (tmp_path / "config.toml").write_text(f'[paths]\nstate = "{tmp_path / "state"}"\n')
    cf = tmp_path / "clusters.toml"
    cf.write_text('[[clusters]]\nid = "gpu"\nkind = "direct"\nhost = "203.0.113.7"\nworkdir = "~/w"\n\n'
                  '[[clusters]]\nid = "hpc"\nkind = "slurm"\nhost = "203.0.113.8"\nworkdir = "~/w"\n'
                  'slurm_script = "t.slurm"\n')
    cf.chmod(0o600)
    c = load(str(tmp_path / "config.toml"))
    assert [x.id for x in c.clusters] == ["gpu"] and not hasattr(c.clusters[0], "kind")
