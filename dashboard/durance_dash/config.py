"""Dashboard configuration.

Read from (first found): $DD_CONFIG, ~/.config/durance-dashboard/config.toml. Every key has a default,
so the file only needs what differs on this machine. See config.example.toml.

Two folder layouts work without configuration:
  * the team repository (this dashboard/ folder next to kria/, ros2/, training/): scripts are taken from
    those folders (REPO_DIRS), weights from models/;
  * the laptop layout ~/go2-hackathon with go2_viz/, models/board_ready/, laptop_ws/.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

HOME = Path.home()


def find_repo() -> Path | None:
    """The repository checkout this dashboard runs from ($DD_REPO, or dashboard/.. when it holds kria/ and
    ros2/); None in the laptop layout."""
    env = os.environ.get("DD_REPO")
    if env:
        return Path(env).expanduser()
    root = Path(__file__).resolve().parent.parent.parent
    return root if (root / "kria").is_dir() and (root / "ros2").is_dir() else None


REPO = find_repo()
# where the board, ROS and training files live in the repository (searched in this order, after go2_viz)
REPO_DIRS = ("kria", "ros2/rviz", "ros2", "training/optimised_snn", "training/optimised_snn/dataset")


class ConfigError(ValueError):
    pass


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"            # loopback only; anything else needs allow_lan = true
    port: int = 8800
    allow_lan: bool = False
    session_hours: float = 8.0
    secure_cookie: bool = False         # true when served over HTTPS (reverse proxy)


@dataclass
class PathsCfg:
    # Scripts and files the dashboard runs or sends to the board (live_gesture_bridge.py, gestures.json,
    # go2.urdf, ...) are found by name: first in go2_viz, then in the repository folders (REPO_DIRS).
    repo: str = str(REPO) if REPO else ""
    go2_viz: str = "" if REPO else str(HOME / "go2-hackathon" / "go2_viz")
    project: str = "" if REPO else str(HOME / "go2-hackathon")
    models: str = str(REPO / "models") if REPO else str(HOME / "go2-hackathon" / "models")
    data: str = str(HOME / "go2-hackathon" / "data")
    share: str = str(HOME / "go2-hackathon" / "share")
    captures: str = str(HOME / "river-captures")
    organisers_zip: str = str(HOME / "Downloads" / "durance-ai-handgesture-recognition (1).zip")
    state: str = str(HOME / ".local" / "share" / "durance-dashboard")
    laptop_env: str = str(REPO / "ros2" / "env.sh") if REPO else str(HOME / "go2-hackathon" / "laptop_ws" / "env.sh")
    board_link_env: str = (str(REPO / "ros2" / "env_board_link.sh") if REPO
                           else str(HOME / "go2-hackathon" / "laptop_ws" / "env_board_link.sh"))
    riverops: str = str(HOME / "go2-hackathon" / "riverops-src" / "riverops")   # the riverops package (training)
    recorder_python: str = str(HOME / ".local" / "share" / "uv" / "tools" / "river-tools" / "bin" / "python")
    ros_python: str = "/usr/bin/python3"


@dataclass
class BoardCfg:
    host: str = "192.168.77.1"
    user: str = "root"
    ssh_port: int = 22
    control_path: str = str(HOME / ".ssh" / "cm-durance-board-%C")
    connect_timeout_s: int = 4
    viewer_port: int = 8080             # snn_input_viewer.py on the board
    local_viewer_port: int = 18080      # SSH-forwarded to this laptop port
    robot_env: str = "/root/robot_link.sh"      # ROS 2 domain 0 = the Go2
    test_env: str = "/root/laptop_link.sh"      # ROS 2 domain 42 = laptop / RViz
    settings_file: str = "/root/bridge_settings.json"
    status_file: str = "/tmp/bridge_status.json"


@dataclass
class RobotCfg:
    max_vx: float = 0.5                 # m/s   (same indoor limits as go2_viz/go2_cmd.py)
    max_vy: float = 0.3
    max_wz: float = 1.0                 # rad/s
    deadman_s: float = 0.4


@dataclass
class SimCfg:
    ros_domain: int = 88                # isolated loopback domain for the local simulator
    board_test_domain: int = 42
    fake_port: int = 3390               # fake spleat-server output stream
    camera_port: int = 3391             # not exposed: frames go through the dashboard
    sim_session: str = "29"             # own-gesture clips replayed by the virtual person
    weights: str = ""                   # relative to paths.models; "" = the own-gesture model (SIM_WEIGHTS)


@dataclass
class ClusterCfg:
    """A GPU cluster for training. Its connection details live only in the private clusters.toml
    (mode 0600, never in the code, the docs or the dashboard's pages): the UI shows `label` and whether it
    is reachable, nothing else."""
    id: str = "gpu"
    label: str = "GPU machine"
    kind: str = "direct"                # "direct" = a GPU machine (cloud VM): training runs in the background;
                                        # "slurm" = a cluster: sbatch / squeue / scancel
    host: str = ""                      # host name or IP ...
    user: str = ""                      # ... login user ...
    port: int = 22                      # ... and SSH port
    identity_file: str = ""             # "~/.ssh/<key>" or "dashboard:<key>" (a key the dashboard generated)
    ssh_host: str = ""                  # instead of host/user/port: an alias from ~/.ssh/config
    control_path: str = ""              # "" = <state>/ssh/cm-<id>-%C
    workdir: str = ""                   # remote folder with the training script, data/ and runs/
    slurm_script: str = ""              # slurm only
    python: str = "python3"             # relative to workdir or on PATH (e.g. venv/bin/python)
    max_submits_per_hour: int = 20

    @property
    def dest(self) -> str:
        if self.ssh_host:
            return self.ssh_host
        return f"{self.user}@{self.host}" if self.user else self.host


@dataclass
class Config:
    server: ServerCfg = field(default_factory=ServerCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    board: BoardCfg = field(default_factory=BoardCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    clusters: list[ClusterCfg] = field(default_factory=list)      # none unless clusters.toml says so
    path: str = ""                                                   # the config file this came from

    @property
    def state_dir(self) -> Path:
        p = Path(self.paths.state).expanduser()
        p.mkdir(parents=True, exist_ok=True, mode=0o700)
        return p

    def cluster(self, cid: str) -> ClusterCfg:
        for c in self.clusters:
            if c.id == cid:
                return c
        raise KeyError(cid)


def _fill(cls, values: dict, where: str):
    if not isinstance(values, dict):
        raise ConfigError(f"{where}: expected a table")
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    obj = cls()
    for k, v in values.items():
        default = getattr(obj, k)
        if is_dataclass(default):
            v = _fill(type(default), v, f"{where}.{k}")
        elif isinstance(default, bool) and not isinstance(v, bool):
            raise ConfigError(f"{where}.{k}: expected true/false")
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ConfigError(f"{where}.{k}: expected a number")
            v = type(default)(v)
        elif isinstance(default, str) and not isinstance(v, str):
            raise ConfigError(f"{where}.{k}: expected a string")
        setattr(obj, k, v)
    return obj


def control_dir(state: str) -> Path:
    """Folder for SSH connection sockets. Unix socket paths are limited to 108 bytes and %C adds 40 characters,
    so prefer the short private runtime folder (/run/user/<uid>); the state folder is the fallback."""
    run = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(run) / "durance-ssh" if run and Path(run).is_dir() else Path(state).expanduser() / "ssh"
    if len(str(base)) > 50:                          # leave room for "/cm-<id>-<40 hex>"
        base = Path(f"/tmp/durance-ssh-{os.getuid()}")
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    return base


def project_dirs(paths: PathsCfg) -> list[Path]:
    dirs = [Path(paths.go2_viz).expanduser()] if paths.go2_viz else []
    if paths.repo:
        dirs += [Path(paths.repo).expanduser() / d for d in REPO_DIRS]
    if paths.project:                    # laptop layout: train_snn_v9.py lives in the team repo clone
        dirs.append(Path(paths.project).expanduser() / "go2-next-reflex" / "training")
    return dirs


def project_file(paths: PathsCfg, name: str) -> Path:
    """A project file by name (live_gesture_bridge.py, gestures.json, go2.urdf, ...) from the first folder of
    project_dirs() that has it; if none has it, where it is expected (so error messages show a path)."""
    dirs = project_dirs(paths)
    for d in dirs:
        if (d / name).is_file():
            return d / name
    return (dirs[0] if dirs else Path.cwd()) / name


# the simulator's own-gesture model (8-bit nb40, md5 056ee113): laptop layout, repository
SIM_WEIGHTS = ("board_ready/own_gestures_nb40_q8h_weights.txt", "optimised_snn/own_gestures/weights_r6b-reg3_056ee113.txt")


def sim_weights(cfg: "Config") -> Path:
    models = Path(cfg.paths.models).expanduser()
    if cfg.sim.weights:
        return models / cfg.sim.weights
    return next((models / w for w in SIM_WEIGHTS if (models / w).exists()), models / SIM_WEIGHTS[0])


def clusters_file(config_path: str) -> Path:
    """Private cluster list: $DD_CLUSTERS, else clusters.toml next to the config file."""
    return Path(os.environ.get("DD_CLUSTERS") or Path(config_path).expanduser().parent / "clusters.toml")


def load(path: str | None = None) -> Config:
    path = path or os.environ.get("DD_CONFIG") or str(HOME / ".config" / "durance-dashboard" / "config.toml")
    raw = {}
    if Path(path).exists():
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    cf = clusters_file(path)
    if "clusters" not in raw and cf.exists():
        if cf.stat().st_mode & 0o077:
            raise ConfigError(f"{cf} holds login details: make it private (chmod 600 {cf})")
        with open(cf, "rb") as f:
            raw["clusters"] = tomllib.load(f).get("clusters", [])
    cfg = Config()
    for key in ("server", "paths", "board", "robot", "sim"):
        if key in raw:
            setattr(cfg, key, _fill(type(getattr(cfg, key)), raw[key], key))
    if "clusters" in raw:
        if not isinstance(raw["clusters"], list):
            raise ConfigError("clusters: expected an array of tables ([[clusters]])")
        cfg.clusters = [_fill(ClusterCfg, c, f"clusters[{i}]") for i, c in enumerate(raw["clusters"])]
        for c in cfg.clusters:
            if not c.control_path:
                c.control_path = str(control_dir(cfg.paths.state) / f"cm-{c.id[:12]}-%C")
    cfg.path = str(Path(path).expanduser())
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    s = cfg.server
    if s.host not in ("127.0.0.1", "::1", "localhost") and not s.allow_lan:
        raise ConfigError(f"server.host = {s.host!r} exposes the dashboard to the network: set "
                          "server.allow_lan = true to confirm (and put it behind HTTPS)")
    if not 1 <= s.port <= 65535:
        raise ConfigError("server.port out of range")
    r = cfg.robot
    for name, v, hi in (("max_vx", r.max_vx, 1.5), ("max_vy", r.max_vy, 0.8), ("max_wz", r.max_wz, 2.5)):
        if not 0 < v <= hi:
            raise ConfigError(f"robot.{name} must be in (0, {hi}]")
    if not 0.1 <= r.deadman_s <= 1.0:
        raise ConfigError("robot.deadman_s must be 0.1..1.0")
    if not 1 <= cfg.sim.ros_domain <= 232 or cfg.sim.ros_domain in (0, 42):
        raise ConfigError("sim.ros_domain must be 1..232 and not 0 (robot) or 42 (board link)")
    ids = [c.id for c in cfg.clusters]
    if len(set(ids)) != len(ids):
        raise ConfigError("clusters: duplicate id")
    for c in cfg.clusters:
        if c.kind not in ("slurm", "direct"):
            raise ConfigError(f"clusters[{c.id}].kind: 'slurm' or 'direct'")
        if not ((c.ssh_host or c.host) and c.workdir):
            raise ConfigError(f"clusters[{c.id}]: host (or ssh_host) and workdir are required")
        if c.kind == "slurm" and not c.slurm_script:
            raise ConfigError(f"clusters[{c.id}]: slurm_script is required for a Slurm cluster")
        if not 1 <= c.port <= 65535:
            raise ConfigError(f"clusters[{c.id}].port out of range")
