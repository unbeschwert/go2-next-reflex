"""Model training on a GPU cluster over SSH (Slurm): submit, watch, cancel, collect results."""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from collections import deque
from pathlib import Path

from fastapi import HTTPException

from .config import project_file
from .errors import need
from .sshrun import SSH, q

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

DATASETS = {
    "organisers": {"label": "Organisers' gestures (sit / lay down / hello / dance), val 13, test 04-06",
                   "args": ["--zip", "data/motion_buddy_full.zip", "--val", "13", "--test", "04,05,06"]},
    "own": {"label": "Our gestures (heart / turn left / turn right), sessions 20-30, val 28, test 29+30",
            "args": ["--zip", "data/own_gestures_s20-30.zip", "--val", "28", "--test", "29,30",
                     "--class-names", "Heart,Turn left 360,Turn right 360,Follow me,No gesture",
                     "--unused", "3", "--mirror", "1:2"]},
}
ARCHS = {"nb40": "40x40, runs on the board's current 8-bit firmware", "v3": "40x40 team v3 net (needs a new firmware)",
         "fast80": "80x80 input, nb40 cost (needs a new firmware)", "best80": "80x80, firmware (3), 16-bit",
         "c40": "compact 40x40 (needs a new firmware)", "slim80": "compact 80x80 (needs a new firmware)"}
# flag: (type, min, max, default, help)
FLAGS = {
    "epochs": ("int", 1, 500, 5, "float training epochs"),
    "qat-epochs": ("int", 0, 300, 50, "quantisation-aware epochs (8-bit, as the FPGA runs it)"),
    "qat-lr-scale": ("float", 0.01, 1.0, 1.0, "QAT learning rate = lr x this"),
    "qat-bits": ("choice", ["8h", "16", "8h,16"], None, "8h", "weight format: 8h = the 8-bit firmware, 16 = firmware (3)"),
    "batch-size": ("int", 4, 128, 16, ""),
    "lr": ("float", 1e-5, 0.1, 0.003, "learning rate"),
    "weight-decay": ("float", 0.0, 0.1, 1e-4, ""),
    "ce-weight": ("float", 0.0, 5.0, 0.5, "cross-entropy term next to the spike-count loss"),
    "spike-reg": ("float", 0.0, 20.0, 0.0, "SynOps penalty: fewer spikes = lower FPGA latency"),
    "learnable-leak": ("int", 0, 1, 0, "1 = every layer learns its own leak"),
    "initial-tau": ("float", 1.01, 50.0, 14.29, "starting neuron time constant (2 = fast leak)"),
    "layer-dropout": ("float", 0.0, 0.5, 0.0, "dropout in the middle layers (training only)"),
    "binom-drop": ("float", 0.0, 0.9, 0.0, "event dropout (binomial thinning)"),
    "time-shift": ("int", 0, 50, 0, "max frames the clip is moved in time"),
    "polarity-p": ("float", 0.0, 1.0, 0.0, "probability of swapping ON/OFF"),
    "mixup-p": ("float", 0.0, 1.0, 0.0, "probability per batch of mixup"),
    "max-shift": ("int", -1, 16, -1, "pixel shift (-1 = size/16)"),
    "flip": ("int", 0, 1, 0, "left-right flip (own gestures: swaps left/right labels)"),
    "aug": ("choice", ["basic", "strong"], None, "basic", "strong = zoom/rotation/shift warp"),
    "warmup-steps": ("int", 0, 100, 40, "context frames before each clip (the FPGA never resets)"),
    "quiet-prob": ("float", 0.0, 0.5, 0.0, "share of synthetic quiet scenes labelled no-gesture"),
    "noaction-weight": ("float", 0.1, 10.0, 1.0, "loss weight of no-action clips"),
    "seed": ("int", 0, 1_000_000, 42, ""),
}
PRESETS = [
    {"id": "base", "label": "Our recipe (r9-base)", "args": {"epochs": 5, "qat-epochs": 50, "qat-lr-scale": 1.0, "flip": 0}},
    {"id": "leak", "label": "+ fast learnable leak (r9-leak)", "args": {"epochs": 5, "qat-epochs": 50, "qat-lr-scale": 1.0,
                                                                         "flip": 0, "learnable-leak": 1, "initial-tau": 2.0}},
    {"id": "aug", "label": "+ team augmentation (r9-aug)", "args": {"epochs": 5, "qat-epochs": 50, "qat-lr-scale": 1.0, "flip": 0,
                                                                     "max-shift": 4, "binom-drop": 0.4, "time-shift": 20,
                                                                     "polarity-p": 0.3, "mixup-p": 0.5, "layer-dropout": 0.2}},
    {"id": "sparse", "label": "Sparse / fast on the FPGA (spike-reg 3, like the live model)",
     "args": {"epochs": 5, "qat-epochs": 50, "qat-lr-scale": 1.0, "spike-reg": 3.0}},
]

RUNS_SCRIPT = r'''
import glob, json, os, sys
out = []
for m in sorted(glob.glob("runs/*/metrics.json"), key=os.path.getmtime, reverse=True)[:200]:
    try:
        j = json.load(open(m))
    except Exception:
        continue
    res = j.get("results", {}); pick = j.get("deployable_pick")
    r = res.get(pick) or {}
    a = j.get("args", {})
    d = os.path.basename(os.path.dirname(m))
    out.append({"name": d, "job": d.rsplit("-", 1)[-1] if d.rsplit("-", 1)[-1].isdigit() else None,
                "arch": (j.get("arch") or {}).get("name"), "dataset": os.path.basename(a.get("zip", "")),
                "val": r.get("val"), "test": r.get("test"), "synops": r.get("synops"),
                "trig_correct": (r.get("trig") or {}).get("correct"), "trig_false": (r.get("trig") or {}).get("false"),
                "pick": pick, "finished": os.path.getmtime(m),
                "files": sorted(os.path.basename(f) for f in glob.glob(os.path.dirname(m) + "/weights_q*.txt"))})
print(json.dumps(out))
'''


DIRECT_JOBS = ".dd_jobs"        # direct machines: "<pid> <run name>" per started run, in the work folder


class Training:
    """Training on GPU machines added in the dashboard. kind "slurm": sbatch / squeue / scancel; kind "direct"
    (any GPU machine, e.g. a cloud VM): the dashboard starts train_snn_v9.py in the background (setsid nohup)
    and tracks its process. Replies never contain host names, user names or paths."""

    def __init__(self, cfg, audit, machines=None):
        self.cfg, self.audit, self.machines = cfg, audit, machines
        self.ssh: dict[str, SSH] = {}
        self._health: dict[str, tuple[float, dict]] = {}
        self._runs: dict[str, tuple[float, list]] = {}
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
                hit = (time.monotonic(), {"id": c.id, "label": c.label, "kind": c.kind, "ok": ok, "detail": detail})
                self._health[c.id] = hit
            out.append(hit[1])
        return {"hosts": out}

    async def test(self, hid: str) -> dict:
        """Connection, GPU, Python, torch/CUDA, riverops and work folder of one machine."""
        c, ssh = self._host(hid)
        ok, ms, err = await ssh.ping()
        if not ok:
            logger.warning("test of %s: %s", c.id, err)
            return {"ok": False, "steps": [{"name": "SSH login", "ok": False, "detail": reason(err)}]}
        code = ("import json,sys\nr={}\ntry:\n import torch; r['torch']=torch.__version__; r['cuda']=torch.cuda.is_available(); "
                "r['gpu']=torch.cuda.get_device_name(0) if r['cuda'] else None\nexcept Exception as e: r['torch']=None; r['err']=type(e).__name__\n"
                "try:\n import riverops; r['riverops']=True\nexcept Exception: r['riverops']=False\nprint(json.dumps(r))")
        b64 = base64.b64encode(code.encode()).decode()
        cmd = (f"mkdir -p {c.workdir} && cd {c.workdir} && echo WORKDIR_OK; "
               f"(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 4); "
               f"{c.python} -c {q(f'import base64;exec(base64.b64decode({b64!r}))')} 2>/dev/null || echo PYTHON_FAIL; "
               f"[ -d riverops ] && echo RIVEROPS_DIR; [ -f train_snn_v9.py ] && echo SCRIPT_OK; "
               f"echo DATA_N=$(ls data/*.zip 2>/dev/null | wc -l); "
               + ("command -v sbatch >/dev/null && echo SLURM_OK" if c.kind == "slurm" else "true"))
        r = await ssh.run(cmd, timeout=90)
        out = r.out
        py = next((json.loads(ln) for ln in out.splitlines() if ln.startswith("{")), {})
        gpus = [ln.strip() for ln in out.splitlines() if re.match(r"^[A-Za-z].*,\s*\d+ MiB$", ln.strip())]
        m = re.search(r"DATA_N=(\d+)", out)
        n_data = int(m.group(1)) if m else 0
        steps = [{"name": "SSH login", "ok": True, "detail": f"{ms:.0f} ms"},
                 {"name": "Work folder", "ok": "WORKDIR_OK" in out, "detail": "ready" if "WORKDIR_OK" in out else "cannot create it"},
                 {"name": "GPU", "ok": bool(gpus) or bool(py.get("cuda")), "detail": "; ".join(gpus) or py.get("gpu") or "no NVIDIA GPU found"},
                 {"name": "Python + PyTorch", "ok": bool(py.get("torch")), "detail": f"torch {py['torch']}, CUDA {'yes' if py.get('cuda') else 'no'}" if py.get("torch") else "PyTorch not found for the configured Python"},
                 {"name": "riverops", "ok": bool(py.get("riverops")),
                  "detail": "found" if py.get("riverops") else ("uploaded, but it needs PyTorch" if "RIVEROPS_DIR" in out
                                                                else "missing: run Prepare machine")},
                 {"name": "Training script + data", "ok": "SCRIPT_OK" in out and n_data > 0,
                  "detail": (f"script + {n_data} dataset(s)" if "SCRIPT_OK" in out and n_data else
                             "script found, no dataset uploaded yet (Prepare machine)" if "SCRIPT_OK" in out else
                             "missing: run Prepare machine")}]
        if c.kind == "slurm":
            steps.append({"name": "Slurm", "ok": "SLURM_OK" in out, "detail": "sbatch found" if "SLURM_OK" in out else "sbatch not found"})
        self._health.pop(c.id, None)
        return {"ok": all(s["ok"] for s in steps), "steps": steps}

    async def prepare(self, user: str, hid: str, datasets: list[str], log) -> dict:
        """Copy the training script, riverops and the chosen datasets to the machine's work folder."""
        c, ssh = self._host(hid)
        paths = self.cfg.paths
        script = project_file(paths, "train_snn_v9.py")
        riverops = Path(paths.riverops).expanduser()
        sources = {"organisers": (Path(paths.organisers_zip).expanduser(), "motion_buddy_full.zip"),
                   "own": (Path(paths.share).expanduser() / "own_gestures_s20-30.zip", "own_gestures_s20-30.zip")}
        bad = [d for d in datasets if d not in sources]
        if bad:
            raise HTTPException(422, f"unknown datasets {bad}")
        for f in [script, riverops] + [sources[d][0] for d in datasets]:
            if not f.exists():
                raise HTTPException(500, f"missing on the laptop: {f.name}")
        home = (await self._run(ssh, f"mkdir -p {c.workdir}/data {c.workdir}/logs {c.workdir}/runs && cd {c.workdir} && pwd",
                                "creating the work folder", timeout=30)).out.strip().splitlines()[-1]
        log("work folder ready")
        for local, remote in ((script, "train_snn_v9.py"), (riverops, "riverops")):
            res = await ssh.scp(str(local), f"{home}/{remote}", download=False, timeout=300)
            if not res.ok:
                logger.warning("upload of %s failed: %s", remote, res.err[-500:])
                raise HTTPException(502, f"upload of {remote} failed: {reason(res.err)}")
            log(f"uploaded {remote}")
        for d in datasets:
            local, name = sources[d]
            log(f"uploading {name} ({local.stat().st_size / 1e6:.0f} MB): this can take a while")
            res = await ssh.scp(str(local), f"{home}/data/{name}", download=False, timeout=6 * 3600)
            if not res.ok:
                logger.warning("upload of %s failed: %s", name, res.err[-500:])
                raise HTTPException(502, f"upload of {name} failed: {reason(res.err)}")
            log(f"uploaded data/{name}")
        r = await ssh.run(f"cd {c.workdir} && {c.python} -c 'import pydantic' 2>/dev/null || "
                          f"{c.python} -m pip install --user -q pydantic 2>&1 | tail -n 2", timeout=300)
        log("checked Python packages" + ("" if r.ok else f": {clean(r.out + r.err)[-200:]}"))
        self.audit.log(user, "train.prepare", {"machine": c.id, "datasets": datasets})
        return {"machine": c.id}

    def options(self) -> dict:
        return {"datasets": [{"id": k, "label": v["label"]} for k, v in DATASETS.items()],
                "archs": [{"id": k, "label": v} for k, v in ARCHS.items()], "presets": PRESETS,
                "flags": [{"key": k, "type": t, "min": lo if t != "choice" else None, "max": hi,
                           "choices": lo if t == "choice" else None, "default": d, "help": h}
                          for k, (t, lo, hi, d, h) in FLAGS.items()]}

    def build_args(self, body: dict) -> tuple[str, list[str]]:
        name = str(body.get("name", ""))
        if not NAME_RE.match(name):
            raise HTTPException(422, "name: 2-41 characters, lowercase letters, digits and -")
        ds = DATASETS.get(body.get("dataset"))
        if ds is None:
            raise HTTPException(422, f"dataset must be one of {sorted(DATASETS)}")
        arch = body.get("arch")
        if arch not in ARCHS:
            raise HTTPException(422, f"arch must be one of {sorted(ARCHS)}")
        args = list(ds["args"]) + ["--arch", arch]
        values = dict(next((p["args"] for p in PRESETS if p["id"] == body.get("preset")), {}))
        user_args = body.get("args") or {}
        if not isinstance(user_args, dict):
            raise HTTPException(422, "args: expected an object")
        values.update(user_args)
        for key, v in values.items():
            spec = FLAGS.get(key)
            if spec is None:
                raise HTTPException(422, f"flag --{key} is not allowed")
            t, lo, hi, _, _ = spec
            if t == "choice":
                if v not in lo:
                    raise HTTPException(422, f"--{key} must be one of {lo}")
                sval = v
            elif t == "int":
                if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not lo <= v <= hi:
                    raise HTTPException(422, f"--{key}: integer {lo}..{hi}")
                sval = str(int(v))
            else:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                    raise HTTPException(422, f"--{key}: number {lo}..{hi}")
                sval = repr(float(v))
            args += [f"--{key}", sval]
        return name, args

    # ---------------------------------------------------------------- jobs
    async def submit(self, user: str, body: dict) -> dict:
        c, ssh = self._host(body.get("host"))
        name, args = self.build_args(body)
        dry = bool(body.get("dry_run", True))
        if c.kind == "slurm":
            cmd = f"cd {c.workdir} && sbatch {'--test-only ' if dry else '--parsable '}--job-name={q(name)} {q(c.slurm_script)} {q(*args)}"
            shown = f"sbatch --job-name={name} {c.slurm_script} " + " ".join(args)
        else:
            run = f"{name}-$(date +%Y%m%d-%H%M%S)"
            train = f"{c.python} -u train_snn_v9.py --out runs/$n {q(*args)}"
            shown = f"{c.python} train_snn_v9.py --out runs/{name}-<time> " + " ".join(args)
            if dry:     # check everything the run needs, start nothing; list what is missing
                need_data = q(next(a for a in args if a.startswith("data/")))
                cmd = (f"cd {c.workdir} 2>/dev/null || {{ echo 'MISSING work folder'; exit 0; }}; "
                       f"[ -f train_snn_v9.py ] || echo 'MISSING training script'; [ -d riverops ] || echo 'MISSING riverops'; "
                       f"[ -f {need_data} ] || echo 'MISSING dataset {need_data}'; "
                       f"{c.python} -c 'import torch' 2>/dev/null || echo 'MISSING PyTorch for {c.python}'; "
                       f"{c.python} -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null "
                       f"|| echo 'MISSING a CUDA GPU'; echo CHECKED")
            else:
                cmd = (f"cd {c.workdir} && mkdir -p logs runs && n={run} && "
                       f"setsid nohup {train} > logs/$n.out 2>&1 < /dev/null & pid=$! && "
                       f"echo \"$pid $n\" >> {DIRECT_JOBS} && echo \"$pid\"")
        if not dry:
            now = time.time()
            while self.submits and now - self.submits[0] > 3600:
                self.submits.popleft()
            if len(self.submits) >= c.max_submits_per_hour:
                raise HTTPException(429, f"at most {c.max_submits_per_hour} submissions per hour")
            need(body.get("confirm", False), f"Start '{name}' on {c.label}? It uses the machine's GPU until it "
                                             f"finishes.\n{shown}")
        r = await self._run(ssh, cmd, "starting the run" if c.kind == "direct" else "sbatch", timeout=60)
        text = clean((r.out + r.err).strip())
        if c.kind == "direct" and dry:
            missing = [ln[8:] for ln in r.out.splitlines() if ln.startswith("MISSING ")]
            if missing:
                raise HTTPException(409, "the machine is not ready: missing " + ", ".join(missing)
                                    + ". Use Prepare machine (and install PyTorch with CUDA if it is missing).")
            text = f"ready: {name} would start now"
        job = None if dry else (re.search(r"(\d+)", r.out) or [None, None])[1]
        if not dry:
            self.submits.append(time.time())
        self.audit.log(user, "train.submit", {"machine": c.id, "name": name, "dry_run": dry, "args": args}, job or text[-200:])
        return {"job_id": job, "command": shown, "dry_run": dry, "output": text[-1000:]}

    async def jobs(self, hid: str | None) -> dict:
        c, ssh = self._host(hid)
        if c.kind == "slurm":
            r = await self._run(ssh, 'squeue -u "$USER" -h -o "%i|%j|%T|%M|%N|%r"', "listing the jobs", timeout=20)
        else:
            r = await self._run(ssh, f"cd {c.workdir} 2>/dev/null && [ -f {DIRECT_JOBS} ] && tail -n 50 {DIRECT_JOBS} | "
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
        cmd = f"scancel {q(job)}" if c.kind == "slurm" else f"kill -TERM -- -{int(job)} 2>/dev/null || kill -TERM {int(job)}"
        await self._run(ssh, cmd, "cancelling the job", timeout=20)
        self.audit.log(user, "train.cancel", {"machine": c.id, "job": job, "name": mine[job]["name"]})
        return {"ok": True}

    async def log(self, hid: str | None, job: str, lines: int) -> dict:
        c, ssh = self._host(hid)
        if not JOB_RE.match(str(job)):
            raise HTTPException(422, "job id")
        lines = max(10, min(int(lines), 2000))
        if c.kind == "slurm":
            find = f"f=$(ls -t logs/*-{job}.out 2>/dev/null | head -n 1)"
        else:
            find = f"f=logs/$(awk '$1 == \"{int(job)}\" {{print $2}}' {DIRECT_JOBS} 2>/dev/null | tail -n 1).out"
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
        c, ssh = self._host(hid)
        if not RUN_RE.match(run or ""):
            raise HTTPException(422, "run name")
        dest = Path(self.cfg.paths.models).expanduser() / "fetched" / run
        dest.mkdir(parents=True, exist_ok=True)
        r = await ssh.run(f"cd {c.workdir}/runs/{q(run)} && ls weights_q*.txt model.json metrics.json 2>/dev/null", timeout=20)
        files = [f for f in r.out.split() if re.fullmatch(r"(weights_q\w+\.txt|model\.json|metrics\.json)", f)]
        if not files:
            raise HTTPException(404, f"run {run} has no weights / metrics yet")
        home = (await self._run(ssh, f"cd {c.workdir} && pwd", "resolving the work folder", timeout=15)).out.strip()
        for f in files:
            res = await ssh.scp(f"{home}/runs/{run}/{f}", str(dest / f), download=True, timeout=300)
            if not res.ok:
                logger.warning("copy of %s failed: %s", f, res.err[-500:])
                raise HTTPException(502, f"copy of {f} failed: {clean(res.err)[-200:] or reason(res.err)}")
            log(f"copied {f}")
        self.audit.log(user, "train.fetch", {"machine": c.id, "run": run, "files": files})
        return {"dir": str(dest), "files": files}
