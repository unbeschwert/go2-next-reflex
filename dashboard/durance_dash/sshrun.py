"""SSH to the Kria board and to the GPU cluster.

Only fixed command templates are run; every variable part is quoted with shlex.quote (q()). One
multiplexed connection per host (ControlMaster), so commands start fast and the cluster gateway does not
see bursts of new logins. BatchMode: never prompts; strict host keys (a re-flashed board must be
re-added to ~/.ssh/known_hosts by hand, on purpose).
"""
from __future__ import annotations

import asyncio
import os
import shlex
import time
from dataclasses import dataclass


def q(*args) -> str:
    """Quote arguments for a POSIX shell."""
    return " ".join(shlex.quote(str(a)) for a in args)


@dataclass
class Result:
    rc: int
    out: str
    err: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def check(self, what: str) -> "Result":
        if not self.ok:
            from fastapi import HTTPException
            raise HTTPException(502, f"{what} failed (exit {self.rc}): {self.err.strip()[-400:] or self.out.strip()[-400:]}")
        return self


class SSH:
    def __init__(self, dest: str, control_path: str, port: int = 22, connect_timeout: int = 4,
                 login_shell: bool = False, identity: str | None = None, known_hosts: list[str] | None = None):
        self.dest, self.port, self.timeout = dest, port, connect_timeout
        self.control_path = os.path.expanduser(control_path)
        os.makedirs(os.path.dirname(self.control_path), mode=0o700, exist_ok=True)
        self.login_shell = login_shell          # wrap commands in `bash -lc` (clusters: profile variables, environment modules)
        self.identity = identity                # private key file (else ssh's own defaults / ~/.ssh/config)
        self.known_hosts = known_hosts          # extra known_hosts files (host keys trusted in the dashboard)

    def common(self) -> list[str]:
        """Options shared by ssh and scp."""
        args = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={self.timeout}", "-o", "StrictHostKeyChecking=yes",
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3", "-o", "LogLevel=ERROR"]
        if self.identity:
            args += ["-i", self.identity, "-o", "IdentitiesOnly=yes"]
        if self.known_hosts:
            args += ["-o", "UserKnownHostsFile=" + " ".join(self.known_hosts)]
        return args

    def base(self, master: bool = True) -> list[str]:
        args = ["ssh", *self.common(), "-p", str(self.port)]
        if master:
            args += ["-o", "ControlMaster=auto", "-o", f"ControlPath={self.control_path}",
                     "-o", "ControlPersist=600"]
        return args + [self.dest]

    def wrap(self, cmd: str) -> str:
        return f"bash -lc {shlex.quote(cmd)}" if self.login_shell else cmd

    async def run(self, cmd: str, input: bytes | None = None, timeout: float = 20.0) -> Result:
        t0 = time.monotonic()
        try:
            p = await asyncio.create_subprocess_exec(
                *self.base(), self.wrap(cmd), stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError:
            return Result(127, "", "ssh client not installed", 0.0)
        try:
            out, err = await asyncio.wait_for(p.communicate(input), timeout)
        except asyncio.TimeoutError:
            p.kill()
            await p.wait()
            return Result(124, "", f"timed out after {timeout:.0f} s", time.monotonic() - t0)
        except asyncio.CancelledError:             # never leave an orphaned ssh behind
            if p.returncode is None:
                p.kill()
                await asyncio.shield(p.wait())
            raise
        return Result(p.returncode, out.decode(errors="replace"), err.decode(errors="replace"), time.monotonic() - t0)

    async def spawn(self, cmd: str) -> asyncio.subprocess.Process:
        """Long-running remote process with stdin/stdout pipes (teleop, status tap)."""
        return await asyncio.create_subprocess_exec(
            *self.base(), self.wrap(cmd), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

    async def ping(self) -> tuple[bool, float, str]:
        r = await self.run("true", timeout=self.timeout + 4)
        return r.ok, r.seconds * 1000, (r.err.strip() or r.out.strip())[-300:]

    async def forward(self, local_port: int, remote_port: int) -> bool:
        """Local port forward 127.0.0.1:local -> board 127.0.0.1:remote on the shared master."""
        if not (await self.ping())[0]:
            return False
        p = await asyncio.create_subprocess_exec(
            "ssh", "-o", f"ControlPath={self.control_path}", "-O", "forward",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", self.dest,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await p.communicate()
        return p.returncode == 0 or b"already" in err.lower()

    async def put(self, data: bytes, remote_path: str, mode: str = "644", timeout: float = 60.0) -> Result:
        """Write bytes to a remote file atomically (tmp + mv)."""
        tmp = remote_path + ".dd-tmp"
        cmd = f"cat > {q(tmp)} && chmod {q(mode)} {q(tmp)} && mv -f {q(tmp)} {q(remote_path)}"
        return await self.run(cmd, input=data, timeout=timeout)

    async def scp(self, src: str, dst: str, download: bool, timeout: float = 600.0) -> Result:
        """Copy a file or directory; `dst`/`src` on the remote side is a path (no host part)."""
        t0 = time.monotonic()
        remote = f"{self.dest}:{dst if not download else src}"
        args = ["scp", "-q", "-r", *self.common(), "-o", f"ControlPath={self.control_path}",
                "-o", "ControlMaster=auto", "-o", "ControlPersist=600", "-P", str(self.port)]
        args += [remote, dst] if download else [src, remote]
        p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(p.communicate(), timeout)
        except asyncio.TimeoutError:
            p.kill()
            await p.wait()
            return Result(124, "", "scp timed out", time.monotonic() - t0)
        return Result(p.returncode, out.decode(errors="replace"), err.decode(errors="replace"), time.monotonic() - t0)

    async def rsync(self, src: str, dst: str, download: bool = False, files_from: list[str] | None = None,
                    extra: list[str] | None = None, timeout: float = 600.0) -> Result:
        """rsync over the shared SSH connection (incremental: only changed files move). The remote side is a
        path (no host part). files_from: copy only these paths, relative to src (a folder)."""
        t0 = time.monotonic()
        rsh = shlex.join(["ssh", *self.common(), "-p", str(self.port), "-o", "ControlMaster=auto",
                          "-o", f"ControlPath={self.control_path}", "-o", "ControlPersist=600"])
        remote = f"{self.dest}:{src if download else dst}"
        args = ["rsync", "-az", "--partial", "-e", rsh, *(extra or [])]
        if files_from is not None:
            args.append("--files-from=-")
        args += [remote, dst] if download else [src, remote]
        data = ("\n".join(files_from) + "\n").encode() if files_from is not None else None
        try:
            p = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE if data else asyncio.subprocess.DEVNULL,
                                                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError:
            return Result(127, "", "rsync not installed on this laptop", 0.0)
        try:
            out, err = await asyncio.wait_for(p.communicate(data), timeout)
        except asyncio.TimeoutError:
            p.kill()
            await p.wait()
            return Result(124, "", "rsync timed out", time.monotonic() - t0)
        except asyncio.CancelledError:
            if p.returncode is None:
                p.kill()
                await asyncio.shield(p.wait())
            raise
        return Result(p.returncode, out.decode(errors="replace"), err.decode(errors="replace"), time.monotonic() - t0)


async def run_local(args: list[str], timeout: float = 600.0, cwd: str | None = None, env: dict | None = None,
                    log=None) -> Result:
    """Run a local program (no shell), streaming its output lines to `log`."""
    t0 = time.monotonic()
    p = await asyncio.create_subprocess_exec(*args, cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.STDOUT)
    lines: list[str] = []

    async def pump():
        assert p.stdout
        async for raw in p.stdout:
            line = raw.decode(errors="replace").rstrip()
            lines.append(line)
            if log:
                log(line)

    try:
        await asyncio.wait_for(asyncio.gather(pump(), p.wait()), timeout)
    except asyncio.TimeoutError:
        p.kill()
        await p.wait()
        return Result(124, "\n".join(lines), f"timed out after {timeout:.0f} s", time.monotonic() - t0)
    return Result(p.returncode, "\n".join(lines), "", time.monotonic() - t0)
