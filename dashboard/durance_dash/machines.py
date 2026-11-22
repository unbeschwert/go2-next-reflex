"""Training machines added in the dashboard: any GPU machine or cluster reachable with an SSH key.

Stored in the private clusters.toml (mode 0600). Passwords are never stored: logins use SSH keys, either one of
the user's keys in ~/.ssh or a key the dashboard generates (its public half is shown to paste into the cloud
provider). A new machine's host key is trusted only after the operator compared its fingerprint.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

from fastapi import HTTPException

from .config import ClusterCfg, ConfigError, clusters_file, control_dir, validate as validate_cfg

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,30}$")
HOST_RE = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$|^\[?[0-9a-fA-F:]{2,39}\]?$")
USER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,31}$")
DIR_RE = re.compile(r"^~?[A-Za-z0-9_./-]{1,200}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
PY_RE = re.compile(r"^[A-Za-z0-9_./-]{1,100}$")
KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
STANDARD_KEYS = {"id_ed25519", "id_ecdsa", "id_rsa", "id_ed25519_sk", "id_ecdsa_sk"}
EDITABLE = ("id", "label", "kind", "host", "user", "port", "identity_file", "ssh_host", "workdir", "slurm_script",
            "python", "max_submits_per_hour")


def _toml(clusters: list[ClusterCfg]) -> str:
    out = ["# Training machines of the Durance dashboard (edited from its Training page). Private: mode 0600.\n"]
    for c in clusters:
        out.append("[[clusters]]")
        for k in EDITABLE:
            v = getattr(c, k)
            if v in ("", None):
                continue
            out.append(f"{k} = {json.dumps(v) if isinstance(v, str) else v}")
        out.append("")
    return "\n".join(out)


class Machines:
    def __init__(self, cfg, config_path: str, on_change):
        self.cfg, self.on_change = cfg, on_change
        self.path = clusters_file(config_path)
        self.keys_dir = cfg.state_dir / "keys"
        self.keys_dir.mkdir(mode=0o700, exist_ok=True)
        self.known_hosts = cfg.state_dir / "known_hosts"

    # ---------------------------------------------------------------- identity / known hosts for SSH
    def identity_path(self, c: ClusterCfg) -> str | None:
        v = c.identity_file
        if not v:
            return None
        if v.startswith("dashboard:"):
            return str(self.keys_dir / v.split(":", 1)[1])
        return os.path.expanduser(v)

    def known_hosts_files(self) -> list[str]:
        return [str(self.known_hosts), os.path.expanduser("~/.ssh/known_hosts")]

    # ---------------------------------------------------------------- list / edit
    def summary(self) -> list[dict]:
        return [{"id": c.id, "label": c.label, "kind": c.kind} for c in self.cfg.clusters]

    def details(self, mid: str) -> dict:
        c = self._get(mid)
        return {k: getattr(c, k) for k in EDITABLE}

    def _get(self, mid: str) -> ClusterCfg:
        for c in self.cfg.clusters:
            if c.id == mid:
                return c
        raise HTTPException(404, "unknown training machine")

    def save(self, data: dict) -> ClusterCfg:
        unknown = set(data) - set(EDITABLE) - {"confirm"}
        if unknown:
            raise HTTPException(422, f"unknown fields {sorted(unknown)}")
        c = ClusterCfg()
        for k in EDITABLE:
            if k in data and data[k] is not None:
                setattr(c, k, data[k])
        c.label = re.sub(r"[\x00-\x1f<>]", "", str(c.label))[:40].strip() or "GPU machine"
        checks = [("id", ID_RE, c.id), ("workdir", DIR_RE, c.workdir), ("python", PY_RE, c.python)]
        if c.ssh_host:
            checks.append(("ssh_host", re.compile(r"^[A-Za-z0-9_.-]{1,64}$"), c.ssh_host))
        else:
            checks.append(("host", HOST_RE, c.host))
            if c.user:
                checks.append(("user", USER_RE, c.user))
        if c.kind == "slurm":
            checks.append(("slurm_script", FILE_RE, c.slurm_script))
        for name, rx, v in checks:
            if not isinstance(v, str) or not rx.match(v):
                raise HTTPException(422, f"{name}: invalid value")
        if not isinstance(c.port, int) or isinstance(c.port, bool) or not 1 <= c.port <= 65535:
            raise HTTPException(422, "port: 1..65535")
        if c.identity_file:
            names = {k["ref"] for k in self.keys()}
            if c.identity_file not in names:
                raise HTTPException(422, "identity_file: choose one of the listed keys")
        c.control_path = str(control_dir(self.cfg.paths.state) / f"cm-{c.id[:12]}-%C")
        others = [x for x in self.cfg.clusters if x.id != c.id]
        trial = others + [c]
        try:
            saved, self.cfg.clusters = self.cfg.clusters, trial
            validate_cfg(self.cfg)
        except ConfigError as e:
            self.cfg.clusters = saved
            raise HTTPException(422, str(e))
        self._write()
        self.on_change()
        return c

    def delete(self, mid: str) -> None:
        self._get(mid)
        self.cfg.clusters = [c for c in self.cfg.clusters if c.id != mid]
        self._write()
        self.on_change()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(_toml(self.cfg.clusters))
        os.replace(tmp, self.path)

    # ---------------------------------------------------------------- SSH keys
    def keys(self) -> list[dict]:
        """Private keys usable for logins: the user's ~/.ssh keys and dashboard-generated ones (no key content)."""
        out = []
        ssh = Path.home() / ".ssh"
        for p in sorted(ssh.glob("*")) if ssh.is_dir() else []:
            # only the standard key names: other keys in ~/.ssh may belong to unrelated accounts, and their file
            # names can reveal them (use an ~/.ssh/config alias for those)
            if p.name not in STANDARD_KEYS:
                continue
            if p.is_file() and not p.name.endswith(".pub") and KEY_RE.match(p.name):
                try:
                    head = p.open("rb").read(40)
                except OSError:
                    continue
                if b"PRIVATE KEY" in head:
                    out.append({"ref": f"~/.ssh/{p.name}", "name": p.name, "source": "~/.ssh",
                                "public": (p.with_name(p.name + ".pub").read_text().strip()
                                           if p.with_name(p.name + ".pub").exists() else None)})
        for p in sorted(self.keys_dir.glob("*")):
            if p.is_file() and not p.name.endswith(".pub"):
                pub = p.with_name(p.name + ".pub")
                out.append({"ref": f"dashboard:{p.name}", "name": p.name, "source": "dashboard",
                            "public": pub.read_text().strip() if pub.exists() else None})
        return out

    def generate_key(self, name: str) -> dict:
        if not KEY_RE.match(name or ""):
            raise HTTPException(422, "key name: letters, digits, . _ -")
        path = self.keys_dir / name
        if path.exists():
            raise HTTPException(409, "a key with that name exists")
        r = subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"durance-dashboard-{name}",
                            "-f", str(path)], capture_output=True, timeout=30)
        if r.returncode != 0:
            raise HTTPException(500, "ssh-keygen failed")
        os.chmod(path, 0o600)
        return {"ref": f"dashboard:{name}", "name": name, "source": "dashboard",
                "public": path.with_name(name + ".pub").read_text().strip()}

    # ---------------------------------------------------------------- host keys
    async def scan_hostkey(self, mid: str) -> dict:
        c = self._get(mid)
        if c.ssh_host:
            return {"keys": [], "note": "this machine uses an ~/.ssh/config alias: its host key is managed there"}
        p = await asyncio.create_subprocess_exec("ssh-keyscan", "-T", "6", "-p", str(c.port), c.host,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, _ = await asyncio.wait_for(p.communicate(), 20)
        lines = [ln for ln in out.decode(errors="replace").splitlines() if ln and not ln.startswith("#")]
        if not lines:
            raise HTTPException(502, "no host key received (host down, wrong port, or firewall)")
        keys = []
        for ln in lines:
            r = subprocess.run(["ssh-keygen", "-lf", "-"], input=ln.encode(), capture_output=True, timeout=10)
            if r.returncode == 0:
                bits, fp, *_rest = r.stdout.decode().split()
                kind = ln.split()[1] if len(ln.split()) > 1 else "?"
                keys.append({"type": kind, "fingerprint": fp, "line": ln})
        trusted = self.known_hosts.read_text().splitlines() if self.known_hosts.exists() else []
        for k in keys:
            k["trusted"] = k["line"] in trusted
        return {"keys": [{"type": k["type"], "fingerprint": k["fingerprint"], "trusted": k["trusted"]} for k in keys],
                "_lines": [k["line"] for k in keys], "_fps": [k["fingerprint"] for k in keys]}

    async def trust_hostkey(self, mid: str, fingerprint: str) -> dict:
        scan = await self.scan_hostkey(mid)
        if fingerprint not in scan.get("_fps", []):
            raise HTTPException(409, "the host now presents a different key: compare the fingerprint again")
        line = scan["_lines"][scan["_fps"].index(fingerprint)]
        existing = self.known_hosts.read_text().splitlines() if self.known_hosts.exists() else []
        if line not in existing:
            fd = os.open(self.known_hosts, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(line + "\n")
        return {"trusted": fingerprint}
