"""Training of our SNN pipeline on GPU servers added in the dashboard, over SSH.

The pipeline is the repository's `python -m training.snn` with a YAML config (training/README.md); nothing
about networks or datasets is hard-coded here. The New run form lists the configs in training/configs/
(the presets), shows each one's values for a few common keys (the flags) and turns every changed value into a
`--set section.key=value` override, validated with the pipeline's own config loader before anything starts.

A machine's work folder is a copy of the repository, like remote/setup.sh and remote/train.sh make it:
- Prepare: installs uv if missing, copies the code (git's tracked and untracked-but-not-ignored files) and
  the proprietary riverops wheel (never committed), installs the training environment
  (`uv sync --group train`) and uploads the dataset sessions you choose (data/motion_buddy_dataset/NN/).
- Start: copies the code again (the run uses exactly this checkout), then
  `uv run --group train python -m training.snn train --config <yaml> --run-dir runs/<name> [--set ...]`
  in the background (setsid nohup); the dashboard tracks its process.
- Fetch: copies runs/<name>/ to this laptop's runs/, where the Board page can install it.
Replies never contain host names, user names or paths.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from fastapi import HTTPException

from .errors import need
from .sshrun import SSH, q, run_local

logger = logging.getLogger("durance.training")


def clean(text: str) -> str:
    """Remote errors for the UI: no user@host, host names or paths (the details go to the dashboard log)."""
    t = re.sub(r"\b[\w.-]+@[\w.-]+\b", "<host>", text or "")
    t = re.sub(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", "<host>", t, flags=re.I)
    t = re.sub(r"(?:~|/)[\w.@%~+-]*(?:/[\w.@%~+-]+)+", "<path>", t)
    return t.strip()


def reason(err: str) -> str:
    e = (err or "").lower()
    if "timed out" in e or "timeout" in e:
        return "connection timed out"
    if "permission denied" in e or "authentication" in e:
        return "login refused"
    if "host key" in e:
        return "host key not trusted"
    if "connection closed" in e or "refused" in e or "kex_exchange" in e:
        return "connection refused (start an SSH master connection first)"
    if "could not resolve" in e or "no such host" in e:
        return "unknown host"
    return "not reachable"


NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,80}$")
JOB_RE = re.compile(r"^\d{1,12}$")
SESSION_RE = re.compile(r"^\d{2}$")
DEFAULT_CONFIG = "training/configs/motion_buddy_80_v4.yaml"     # the default recipe (training/README.md)
# Dataset sessions Prepare can upload: the organisers' (Durance AI team) sessions and our own recordings.
DATASETS = {"organisers": ("Durance AI team sessions 01-14 (sit, lay down, hello, dance)", lambda n: n < 20),
            "own": ("Our sessions 20-30 (heart, turn left, turn right)", lambda n: n >= 20)}

# Config keys the form offers (any other key can still be changed in the YAML file).
# key: (type, min or choices, max, help)
FLAGS = {
    "seed": ("int", 0, 1_000_000, "random seed; runs differ by ~7 test points between seeds, so compare 2+"),
    "train.epochs": ("int", 1, 500, "float training epochs"),
    "train.batch_size": ("int", 1, 512, ""),
    "train.lr": ("float", 1e-6, 0.1, "learning rate"),
    "train.optimizer": ("choice", ["adam", "adamw"], None, "Adam (L2 in the gradient) or AdamW (decoupled decay)"),
    "train.weight_decay": ("float", 0.0, 1.0, ""),
    "train.loss": ("choice", ["mse", "ce", "ce+mse"], None, "spike-count targets (mse) and/or cross-entropy"),
    "train.average_last": ("int", 0, 100, "average the weights of the last N epochs (BatchNorm statistics recomputed)"),
    "quant.qat_epochs": ("int", 0, 200, "quantization-aware epochs after float training"),
    "quant.qat_lr": ("float", 1e-7, 0.01, ""),
    "model.neuron.learnable_leak": ("bool", None, None, "every layer learns its own leak"),
    "data.train": ("sessions", None, None, "training sessions (folders of data/motion_buddy_dataset), e.g. 01,02,03"),
    "data.val": ("sessions", None, None, "validation sessions (model selection)"),
    "data.test": ("sessions", None, None, "test sessions (reported only)"),
    **{f"augment.{a}.p": ("float", 0.0, 1.0, f"probability of the {a.replace('_', ' ')} augmentation")
       for a in ("speed", "time_shift", "affine", "flip", "dropout", "rate", "polarity", "noise", "hot_pixels",
                 "cutout", "warm_start", "mixup")},
}

RUNS_SCRIPT = r'''
import glob, json, os, re
out = []
for m in sorted(glob.glob("runs/*/metrics.json"), key=os.path.getmtime, reverse=True)[:200]:
    try:
        j = json.load(open(m))
    except Exception:
        continue
    d = os.path.dirname(m)
    r = j.get("fixed_point") or j.get("qat") or j.get("float") or {}
    try:
        name = re.search(r"^name:\s*(\S+)", open(d + "/config.yaml").read(), re.M).group(1)
    except Exception:
        name = None
    out.append({"name": os.path.basename(d), "config": name, "val": r.get("val_acc"), "test": r.get("test_acc"),
                "kind": "fixed point" if j.get("fixed_point") else None, "finished": os.path.getmtime(m),
                "deployable": os.path.isfile(d + "/deploy/weights.txt"), "classes": j.get("classes")})
print(json.dumps(out))
'''

JOBS_FILE = ".dd_jobs"          # "<pid> <run name>" per started run, in the work folder


def pipeline(repo: Path):
    """The repository's training.snn package (config loader and --set overrides; numpy + pyyaml only)."""
    if not (repo / "training" / "snn" / "__main__.py").is_file():
        raise HTTPException(503, "training/snn not found: training needs the repository checkout")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from training.snn import __main__ as cli
    from training.snn import config as conf
    return cli, conf


def get_key(cfg_dict: dict, key: str):
    node = cfg_dict
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def form_value(kind: str, v):
    return ",".join(str(x) for x in v) if kind == "sessions" and isinstance(v, list) else v


class Training:
    """Our pipeline on GPU servers added in the dashboard (SSH): the dashboard starts each run in the
    background (setsid nohup) and tracks its process."""

    def __init__(self, cfg, audit, machines=None):
        self.cfg, self.audit, self.machines = cfg, audit, machines
        self.repo = Path(cfg.paths.repo).expanduser() if cfg.paths.repo else None
        self.ssh: dict[str, SSH] = {}
        self._health: dict[str, tuple[float, dict]] = {}
        self._runs: dict[str, tuple[float, list]] = {}
        self._configs: tuple[tuple, list] | None = None
        self.submits: deque = deque()
        self._jobs_n: dict[str, int] = {}
        self._refreshing = False
        self._refresh_t = 0.0
        self.reload()

    def reload(self) -> None:
        """(Re)build the SSH settings after machines were added, changed or removed."""
        self.ssh = {}
        for c in self.cfg.clusters:
            ident = self.machines.identity_path(c) if self.machines else None
            kh = self.machines.known_hosts_files() if self.machines else None
            self.ssh[c.id] = SSH(c.dest, c.control_path, port=c.port, login_shell=True, connect_timeout=8,
                                 identity=ident, known_hosts=kh)
        self._health.clear()
        self._runs.clear()
        self._jobs_n.clear()
        self._refresh_t = 0.0

    def status(self) -> dict:
        """Cached health + running jobs of the first machine for the overview; refreshes in the background."""
        if self.cfg.clusters and not self._refreshing and time.monotonic() - self._refresh_t > 30:
            import asyncio
            self._refreshing = True
            self._task = asyncio.get_running_loop().create_task(self._refresh())
        if not self.cfg.clusters:
            return {"configured": False, "host": None, "ok": None, "detail": "no training machine configured",
                    "running_jobs": None}
        c = self.cfg.clusters[0]
        h = self._health.get(c.id)
        return {"configured": True, "host": c.label, "ok": h[1]["ok"] if h else None,
                "detail": h[1]["detail"] if h else None, "running_jobs": self._jobs_n.get(c.id)}

    async def shutdown(self) -> None:
        t = getattr(self, "_task", None)
        if t and not t.done():
            t.cancel()
            try:
                await t
            except BaseException:  # noqa: BLE001 - cancelled on purpose
                pass

    async def _refresh(self) -> None:
        try:
            await self.hosts()
            c = self.cfg.clusters[0]
            if self._health.get(c.id, (0, {}))[1].get("ok"):
                jobs = (await self.jobs(c.id))["jobs"]
                self._jobs_n[c.id] = sum(j["state"] == "RUNNING" for j in jobs)
        except Exception:  # noqa: BLE001 - overview status only
            pass
        finally:
            self._refresh_t = time.monotonic()
            self._refreshing = False

    def _host(self, hid: str | None):
        if not self.cfg.clusters:
            raise HTTPException(409, "no training machine configured")
        try:
            hid = hid or self.cfg.clusters[0].id
            return self.cfg.cluster(hid), self.ssh[hid]
        except KeyError:
            raise HTTPException(404, "unknown training machine")

    async def _run(self, ssh, cmd: str, what: str, timeout: float = 30.0):
        r = await ssh.run(cmd, timeout=timeout)
        if not r.ok:
            logger.warning("%s failed (exit %s): %s", what, r.rc, (r.err or r.out)[-1000:])
            if r.rc in (124, 255):                     # timeout / ssh itself failed: a connection problem
                raise HTTPException(502, f"{what} failed: {reason(r.err)}")
            detail = clean(r.err or r.out)[-300:] or f"the command ended with exit code {r.rc}"
            raise HTTPException(502, f"{what} failed: {detail}")
        return r

    @staticmethod
    def uv(c) -> str:
        """Shell snippet: the machine's uv (configured, on PATH, or where Prepare puts it)."""
        return q(c.uv) if c.uv else '"$(command -v uv || echo $HOME/.local/bin/uv)"'

    def _repo(self) -> Path:
        if not self.repo:
            raise HTTPException(503, "training needs the repository checkout (paths.repo)")
        return self.repo

    # ---------------------------------------------------------------- configs (the form)
    def configs(self) -> list[dict]:
        """Every YAML config under training/configs/, resolved with the pipeline's loader (base: included)."""
        repo = self._repo()
        root = Path(self.cfg.paths.training_configs).expanduser() if self.cfg.paths.training_configs else repo / "training" / "configs"
        files = sorted(root.rglob("*.yaml"))
        key = tuple((str(f), f.stat().st_mtime_ns) for f in files)
        if self._configs and self._configs[0] == key:
            return self._configs[1]
        _, conf = pipeline(repo)
        out = []
        for f in files:
            rel = str(f.relative_to(repo)) if f.is_relative_to(repo) else str(f)
            try:
                c = conf.load_config(f)
            except Exception as e:  # noqa: BLE001 - a broken config is listed as such, not fatal
                logger.info("config %s does not load: %s", rel, e)
                continue
            d = c.to_dict()
            size, layers = c.data.size, len(c.model.layers)
            label = (f"{rel.removeprefix('training/configs/')} · {size}×{size}, {layers} layers, {c.train.epochs} epochs, "
                     f"val {','.join(c.data.val)} / test {','.join(c.data.test)}")
            out.append({"id": rel, "label": label, "name": c.name, "classes": list(c.data.classes), "data_root": c.data.root,
                        "sessions": sorted(set(c.data.train) | set(c.data.val) | set(c.data.test)),
                        "args": {k: form_value(FLAGS[k][0], get_key(d, k)) for k in FLAGS if get_key(d, k) is not None}})
        out.sort(key=lambda x: (x["id"] != DEFAULT_CONFIG, x["id"].count("/"), x["id"]))
        self._configs = (key, out)
        return out

    def options(self) -> dict:
        presets = self.configs()
        flags = []
        for k, (t, lo, hi, h) in FLAGS.items():
            flags.append({"key": k, "type": "str" if t == "sessions" else t,
                          "min": lo if t in ("int", "float") else None, "max": hi,
                          "choices": lo if t == "choice" else None,
                          "default": presets[0]["args"].get(k) if presets else None, "help": h})
        return {"pipeline": "training.snn", "presets": presets, "datasets": [], "archs": [], "flags": flags,
                "prepare_datasets": [{"id": k, "label": v[0]} for k, v in DATASETS.items()]}

    def build_args(self, body: dict) -> tuple[str, dict, list[str]]:
        """(run name, config entry, --set overrides) from the form; validated with the pipeline's loader."""
        name = str(body.get("name", ""))
        if not NAME_RE.match(name):
            raise HTTPException(422, "name: 2-41 characters, lowercase letters, digits and -")
        conf = next((c for c in self.configs() if c["id"] == body.get("preset")), None)
        if conf is None:
            raise HTTPException(422, "preset: choose one of the configs in training/configs/")
        values = body.get("args") or {}
        if not isinstance(values, dict):
            raise HTTPException(422, "args: expected an object")
        overrides = []
        for key, v in values.items():
            spec = FLAGS.get(key)
            if spec is None:
                raise HTTPException(422, f"{key}: not a key the form can change (edit the YAML file)")
            t, lo, hi, _ = spec
            if t == "choice":
                if v not in lo:
                    raise HTTPException(422, f"{key}: one of {lo}")
            elif t == "bool":
                if v not in (True, False, 0, 1):
                    raise HTTPException(422, f"{key}: true or false")
                v = bool(v)
            elif t == "sessions":
                v = [x.strip() for x in str(v).split(",") if x.strip()]
                if not v or not all(SESSION_RE.match(x) for x in v):
                    raise HTTPException(422, f"{key}: session numbers like 01,02,13")
            elif t == "int":
                if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not lo <= v <= hi:
                    raise HTTPException(422, f"{key}: integer {lo}..{hi}")
                v = int(v)
            else:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                    raise HTTPException(422, f"{key}: number {lo}..{hi}")
                v = float(v)
            current = conf["args"].get(key)
            if (v if t != "sessions" else ",".join(v)) == current:
                continue                                  # unchanged: the config's own value
            overrides.append(f"{key}={json.dumps(v)}")    # JSON is valid YAML
        cli, conf_mod = pipeline(self._repo())
        try:
            cfg = cli.apply_overrides(conf_mod.load_config(self._repo() / conf["id"]), overrides)
        except Exception as e:  # noqa: BLE001 - ConfigError and friends: show the loader's message
            raise HTTPException(422, f"config: {e}")
        conf = dict(conf, sessions=sorted(set(cfg.data.train) | set(cfg.data.val) | set(cfg.data.test)),
                    data_root=cfg.data.root)
        return name, conf, overrides

    # ---------------------------------------------------------------- code and data
    async def _workdir(self, c, ssh) -> str:
        r = await self._run(ssh, f"mkdir -p {c.workdir}/data {c.workdir}/runs {c.workdir}/logs && cd {c.workdir} && pwd",
                            "creating the work folder", timeout=30)
        return r.out.strip().splitlines()[-1]

    def _code_files(self) -> list[str]:
        repo = self._repo()
        r = subprocess.run(["git", "-C", str(repo), "ls-files", "-co", "--exclude-standard"], capture_output=True,
                           text=True, timeout=60)
        if r.returncode != 0:
            raise HTTPException(500, "git ls-files failed: is the repository a git checkout?")
        files = [f for f in r.stdout.splitlines() if f and (repo / f).is_file()]
        wheel = Path(self.cfg.paths.riverops).expanduser() if self.cfg.paths.riverops else None
        if not wheel or not wheel.is_file():
            raise HTTPException(500, "the riverops wheel (paths.riverops, data/riverops-*.whl) is missing on the laptop")
        if not wheel.is_relative_to(repo):
            raise HTTPException(500, "the riverops wheel must be inside the repository (data/)")
        return files + [str(wheel.relative_to(repo))]

    async def _sync_code(self, ssh, home: str, log) -> None:
        files = self._code_files()
        r = await ssh.rsync(str(self._repo()) + "/", home + "/", files_from=files, timeout=900)
        if not r.ok:
            logger.warning("code sync failed: %s", r.err[-500:])
            raise HTTPException(502, f"copying the code failed: {clean(r.err)[-200:] or reason(r.err)}")
        log(f"code synced ({len(files)} files)")

    def _sessions(self, which: list[str], data_root: str) -> list[str]:
        root = self._repo() / data_root
        out = []
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if d.is_dir() and SESSION_RE.match(d.name) and any(DATASETS[w][1](int(d.name)) for w in which):
                out.append(d.name)
        return out

    # ---------------------------------------------------------------- machines
    async def hosts(self) -> dict:
        out = []
        for c in self.cfg.clusters:
            hit = self._health.get(c.id)
            if not hit or time.monotonic() - hit[0] > 30:
                ok, ms, err = await self.ssh[c.id].ping()
                if not ok:
                    logger.warning("training machine %s not reachable: %s", c.id, err)
                detail = f"reachable, {ms:.0f} ms" if ok else reason(err)
                hit = (time.monotonic(), {"id": c.id, "label": c.label, "ok": ok, "detail": detail})
                self._health[c.id] = hit
            out.append(hit[1])
        return {"hosts": out}

    async def test(self, hid: str) -> dict:
        """Connection, work folder, GPU, uv, code, training environment (torch + riverops) and data."""
        c, ssh = self._host(hid)
        ok, ms, err = await ssh.ping()
        if not ok:
            logger.warning("test of %s: %s", c.id, err)
            return {"ok": False, "steps": [{"name": "SSH login", "ok": False, "detail": reason(err)}]}
        code = ("import json\nr={}\ntry:\n import torch; r['torch']=torch.__version__; r['cuda']=torch.cuda.is_available(); "
                "r['gpu']=torch.cuda.get_device_name(0) if r['cuda'] else None\nexcept Exception as e: r['torch']=None\n"
                "try:\n import riverops; r['riverops']=True\nexcept Exception: r['riverops']=False\nprint(json.dumps(r))")
        b64 = base64.b64encode(code.encode()).decode()
        uv = self.uv(c)
        cmd = (f"mkdir -p {c.workdir} && cd {c.workdir} && echo WORKDIR_OK; "
               f"(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 4); "
               f"UV={uv}; [ -x \"$UV\" ] && echo UV_OK; [ -f training/snn/__main__.py ] && echo CODE_OK; "
               f"[ -x \"$UV\" ] && [ -f pyproject.toml ] && \"$UV\" run --group train --no-sync python -c "
               f"{q(f'import base64;exec(base64.b64decode({b64!r}))')} 2>/dev/null; "
               f"echo SESSIONS=$(ls -d data/motion_buddy_dataset/[0-9][0-9] 2>/dev/null | wc -l)")
        r = await ssh.run(cmd, timeout=120)
        out = r.out
        py = next((json.loads(ln) for ln in out.splitlines() if ln.startswith("{")), {})
        gpus = [ln.strip() for ln in out.splitlines() if re.match(r"^[A-Za-z].*,\s*\d+ MiB$", ln.strip())]
        m = re.search(r"SESSIONS=(\d+)", out)
        n_sessions = int(m.group(1)) if m else 0
        steps = [{"name": "SSH login", "ok": True, "detail": f"{ms:.0f} ms"},
                 {"name": "Work folder", "ok": "WORKDIR_OK" in out, "detail": "ready" if "WORKDIR_OK" in out else "cannot create it"},
                 {"name": "GPU", "ok": bool(gpus) or bool(py.get("cuda")), "detail": "; ".join(gpus) or py.get("gpu") or "no NVIDIA GPU found"},
                 {"name": "uv", "ok": "UV_OK" in out, "detail": "found" if "UV_OK" in out else "missing: run Prepare machine"},
                 {"name": "Code", "ok": "CODE_OK" in out, "detail": "training/snn found" if "CODE_OK" in out else "missing: run Prepare machine"},
                 {"name": "PyTorch + riverops", "ok": bool(py.get("torch")) and bool(py.get("riverops")),
                  "detail": (f"torch {py['torch']}, CUDA {'yes' if py.get('cuda') else 'no'}, riverops "
                             f"{'found' if py.get('riverops') else 'missing'}") if py.get("torch")
                  else "training environment not installed: run Prepare machine"},
                 {"name": "Data", "ok": n_sessions > 0,
                  "detail": f"{n_sessions} session folder(s)" if n_sessions else "no sessions uploaded yet (Prepare machine)"}]
        self._health.pop(c.id, None)
        return {"ok": all(s["ok"] for s in steps), "steps": steps}

    async def prepare(self, user: str, hid: str, datasets: list[str], log) -> dict:
        """uv, code + riverops wheel, the training environment and the chosen dataset sessions."""
        c, ssh = self._host(hid)
        bad = [d for d in datasets if d not in DATASETS]
        if bad:
            raise HTTPException(422, f"unknown datasets {bad}")
        repo = self._repo()
        home = await self._workdir(c, ssh)
        log("work folder ready")
        if not c.uv and not (await ssh.run('command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ]', timeout=20)).ok:
            local = shutil.which("uv")
            if not local:
                raise HTTPException(500, "uv is not installed on this laptop (needed to copy it to the machine)")
            await self._run(ssh, "mkdir -p $HOME/.local/bin", "creating ~/.local/bin", timeout=20)
            r = await ssh.rsync(local, ".local/bin/uv", timeout=300)
            if not r.ok:
                raise HTTPException(502, f"copying uv failed: {reason(r.err)}")
            log("copied uv to ~/.local/bin/uv")
        await self._sync_code(ssh, home, log)
        log("installing the training environment (uv sync --group train; the first time downloads torch, ~3 GB)")
        r = await ssh.run(f"cd {q(home)} && UV={self.uv(c)} && \"$UV\" sync --group train --quiet 2>&1 | tail -n 5 && "
                          "\"$UV\" run --group train --no-sync python -c "
                          + q("import torch, riverops; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"),
                          timeout=3600)
        log(clean(r.out + r.err)[-600:])
        if not r.ok:
            raise HTTPException(502, "installing the training environment failed: " + clean(r.err or r.out)[-300:])
        data_root = (next((x for x in self.configs() if x["id"] == DEFAULT_CONFIG), None) or {}).get("data_root") \
            or "data/motion_buddy_dataset"
        sessions = self._sessions(datasets, data_root)
        if datasets and not sessions:
            log(f"no matching session folders in {data_root} on this laptop")
        if sessions:
            log(f"uploading {len(sessions)} session folder(s) of {data_root}: {', '.join(sessions)} (only changed files move)")
            r = await ssh.rsync(str(repo / data_root) + "/", f"{home}/{data_root}/", files_from=sessions,
                                extra=["-r"], timeout=6 * 3600)
            if not r.ok:
                logger.warning("data upload failed: %s", r.err[-500:])
                raise HTTPException(502, f"uploading the data failed: {clean(r.err)[-200:] or reason(r.err)}")
            log("data uploaded")
        self.audit.log(user, "train.prepare", {"machine": c.id, "datasets": datasets, "sessions": sessions})
        return {"machine": c.id, "sessions": sessions}

    # ---------------------------------------------------------------- jobs
    async def _local_check(self, conf: dict, overrides: list[str]) -> str:
        """`python -m training.snn check`: the network, its FPGA memory estimate and the config's validity."""
        args = [sys.executable, "-m", "training.snn", "check", "--config", conf["id"]]
        for o in overrides:
            args += ["--set", o]
        r = await run_local(args, timeout=120, cwd=str(self._repo()))
        if not r.ok:
            raise HTTPException(422, "config check failed: " + r.out.strip()[-600:])
        return r.out

    async def submit(self, user: str, body: dict) -> dict:
        c, ssh = self._host(body.get("host"))
        name, conf, overrides = self.build_args(body)
        dry = bool(body.get("dry_run", True))
        sets = " ".join(f"--set {q(o)}" for o in overrides)
        train = (f"\"$UV\" run --group train python -m training.snn train --config {q(conf['id'])} "
                 f"--run-dir runs/$n {sets}").strip()
        shown = (f"uv run --group train python -m training.snn train --config {conf['id']} --run-dir runs/{name}-<time> "
                 + " ".join(f"--set {o}" for o in overrides)).strip()
        report = await self._local_check(conf, overrides)
        if dry:     # check everything the run needs, start nothing; list what is missing
            data_root = conf["data_root"]
            missing_sessions = " ".join(f"[ -d {q(data_root + '/' + s)} ] || echo 'MISSING session {s}';"
                                        for s in conf["sessions"])
            cmd = (f"cd {c.workdir} 2>/dev/null || {{ echo 'MISSING work folder'; exit 0; }}; UV={self.uv(c)}; "
                   f"[ -x \"$UV\" ] || echo 'MISSING uv'; [ -f pyproject.toml ] || echo 'MISSING code'; "
                   f"{missing_sessions} "
                   f"[ -x \"$UV\" ] && \"$UV\" run --group train --no-sync python -c 'import torch, riverops' 2>/dev/null "
                   f"|| echo 'MISSING the training environment'; "
                   f"[ -x \"$UV\" ] && \"$UV\" run --group train --no-sync python -c "
                   f"'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null || echo 'MISSING a CUDA GPU'; "
                   "echo CHECKED")
            r = await self._run(ssh, cmd, "checking the machine", timeout=90)
            missing = [ln[8:] for ln in r.out.splitlines() if ln.startswith("MISSING ")]
            if missing:
                raise HTTPException(409, "the machine is not ready: missing " + ", ".join(missing)
                                    + ". Use Prepare machine (with the datasets this config uses).")
            self.audit.log(user, "train.submit", {"machine": c.id, "name": name, "dry_run": True, "config": conf["id"],
                                                  "overrides": overrides}, "ready")
            return {"job_id": None, "command": shown + "\n\n" + report.strip(), "dry_run": True,
                    "output": f"ready: {name} would start now"}
        now = time.time()
        while self.submits and now - self.submits[0] > 3600:
            self.submits.popleft()
        if len(self.submits) >= c.max_submits_per_hour:
            raise HTTPException(429, f"at most {c.max_submits_per_hour} submissions per hour")
        need(body.get("confirm", False), f"Start '{name}' on {c.label}? It uses the machine's GPU until it finishes.\n{shown}")
        home = await self._workdir(c, ssh)
        await self._sync_code(ssh, home, lambda _: None)
        # Plain ';' before the '&': with 'a && b && c &' the whole chain would run in a background
        # subshell, and n, UV and the cd would not exist for the lines after it.
        cmd = (f"cd {q(home)} || exit 1; mkdir -p logs runs; UV={self.uv(c)}; n={q(name)}-$(date +%Y%m%d-%H%M%S); "
               f"mkdir -p runs/$n; setsid nohup {train} > logs/$n.out 2>&1 < /dev/null & pid=$!; "
               f"echo \"$pid $n\" >> {JOBS_FILE}; echo \"$pid\"")
        r = await self._run(ssh, cmd, "starting the run", timeout=60)
        job = (re.search(r"(\d+)", r.out) or [None, None])[1]
        self.submits.append(time.time())
        self.audit.log(user, "train.submit", {"machine": c.id, "name": name, "dry_run": False, "config": conf["id"],
                                              "overrides": overrides}, job)
        return {"job_id": job, "command": shown, "dry_run": False, "output": clean(r.out + r.err)[-1000:]}

    async def jobs(self, hid: str | None) -> dict:
        c, ssh = self._host(hid)
        r = await self._run(ssh, f"cd {c.workdir} 2>/dev/null && [ -f {JOBS_FILE} ] && tail -n 50 {JOBS_FILE} | "
                                 "while read pid n; do kill -0 \"$pid\" 2>/dev/null && "
                                 "echo \"$pid|$n|RUNNING|$(ps -o etime= -p $pid | tr -d ' ')|this machine|\"; done; true",
                            "listing the runs", timeout=20)
        jobs = []
        for line in r.out.splitlines():
            p = line.split("|")
            if len(p) == 6:
                jobs.append({"id": p[0], "name": p[1], "state": p[2], "time": p[3], "node": p[4], "reason": p[5]})
        return {"jobs": jobs}

    async def cancel(self, user: str, hid: str | None, job: str, confirm: bool) -> dict:
        c, ssh = self._host(hid)
        if not JOB_RE.match(str(job)):
            raise HTTPException(422, "job id")
        mine = {j["id"]: j for j in (await self.jobs(c.id))["jobs"]}
        if job not in mine:
            raise HTTPException(404, "not one of your running jobs")
        need(confirm, f"Cancel job {job} ({mine[job]['name']}, {mine[job]['state']} for {mine[job]['time']})?")
        cmd = f"kill -TERM -- -{int(job)} 2>/dev/null || kill -TERM {int(job)}"    # setsid: the run's process group
        await self._run(ssh, cmd, "cancelling the job", timeout=20)
        self.audit.log(user, "train.cancel", {"machine": c.id, "job": job, "name": mine[job]["name"]})
        return {"ok": True}

    async def log(self, hid: str | None, job: str, lines: int) -> dict:
        c, ssh = self._host(hid)
        if not JOB_RE.match(str(job)):
            raise HTTPException(422, "job id")
        lines = max(10, min(int(lines), 2000))
        find = f"f=logs/$(awk '$1 == \"{int(job)}\" {{print $2}}' {JOBS_FILE} 2>/dev/null | tail -n 1).out"
        r = await self._run(ssh, f'cd {c.workdir} && {find}; [ -f "$f" ] && tail -n {lines} "$f" || echo "no log for job {job} yet"',
                            "reading the log", timeout=20)
        return {"lines": r.out.splitlines()}

    async def runs(self, hid: str | None, max_age: float = 60.0) -> dict:
        c, ssh = self._host(hid)
        hit = self._runs.get(c.id)
        if hit and time.monotonic() - hit[0] < max_age:
            return {"runs": hit[1]}
        code = base64.b64encode(RUNS_SCRIPT.encode()).decode()
        r = await self._run(ssh, f"cd {c.workdir} && {c.python} -c {q(f'import base64;exec(base64.b64decode({code!r}))')}",
                            "collecting run results", timeout=60)
        try:
            runs = json.loads(r.out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise HTTPException(502, "the machine returned no run list")
        self._runs[c.id] = (time.monotonic(), runs)
        return {"runs": runs}

    async def fetch(self, user: str, hid: str | None, run: str, log) -> dict:
        """Copy runs/<run>/ (config, metrics, checkpoints, deploy/) to this laptop's runs/<run>/."""
        c, ssh = self._host(hid)
        if not RUN_RE.match(run or ""):
            raise HTTPException(422, "run name")
        if not self.cfg.paths.runs:
            raise HTTPException(503, "no runs folder configured (paths.runs)")
        dest = Path(self.cfg.paths.runs).expanduser() / run
        home = await self._workdir(c, ssh)
        r = await ssh.run(f"[ -f {q(home + '/runs/' + run + '/metrics.json')} ]", timeout=20)
        if not r.ok:
            raise HTTPException(404, f"run {run} has no results yet")
        dest.mkdir(parents=True, exist_ok=True)
        r = await ssh.rsync(f"{home}/runs/{run}/", str(dest) + "/", download=True, timeout=1800)
        if not r.ok:
            logger.warning("fetch of %s failed: %s", run, r.err[-500:])
            raise HTTPException(502, f"copying the run failed: {clean(r.err)[-200:] or reason(r.err)}")
        files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
        log(f"copied {len(files)} files to runs/{run}/")
        self.audit.log(user, "train.fetch", {"machine": c.id, "run": run, "files": len(files)})
        return {"run": run, "files": len(files)}
