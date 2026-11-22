"""Users, sessions, CSRF, login rate limiting, audit log and HTTP security headers."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import HTTPException, Request

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")
MIN_PASSWORD = 10
SCRYPT = dict(n=2 ** 15, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=32)
COOKIE = "dd_session"

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
       "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
       "frame-ancestors 'none'")
HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Cache-Control": "no-store",
}


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)


def _write_private(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class Users:
    """users.json: {username: {salt, hash, created}} with mode 0600."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def empty(self) -> bool:
        return not self._load()

    def create(self, username: str, password: str) -> None:
        check_new_credentials(username, password)
        with self.lock:
            users = self._load()
            if username in users:
                raise HTTPException(409, "user exists")
            salt = secrets.token_bytes(16)
            users[username] = {"salt": salt.hex(), "hash": _hash(password, salt).hex(), "created": time.time()}
            _write_private(self.path, json.dumps(users, indent=1))

    def verify(self, username: str, password: str) -> bool:
        u = self._load().get(username)
        if not u:
            _hash(password, b"0" * 16)          # same time as a real check (no user enumeration)
            return False
        return hmac.compare_digest(_hash(password, bytes.fromhex(u["salt"])).hex(), u["hash"])

    def set_password(self, username: str, password: str) -> None:
        check_new_credentials(username, password)
        with self.lock:
            users = self._load()
            salt = secrets.token_bytes(16)
            users[username].update(salt=salt.hex(), hash=_hash(password, salt).hex(), changed=time.time())
            _write_private(self.path, json.dumps(users, indent=1))


def check_new_credentials(username: str, password: str) -> None:
    if not USERNAME_RE.match(username or ""):
        raise HTTPException(422, "username: 2-32 characters, letters, digits, . _ -")
    if len(password or "") < MIN_PASSWORD:
        raise HTTPException(422, f"password: at least {MIN_PASSWORD} characters")
    if password.lower() == username.lower() or len(set(password)) < 5:
        raise HTTPException(422, "password too weak")


class Sessions:
    """Server-side sessions (in memory: a restart logs everybody out, which is the safe default)."""

    def __init__(self, hours: float):
        self.ttl = hours * 3600
        self.items: dict[str, dict] = {}
        self.lock = threading.Lock()

    def create(self, user: str) -> tuple[str, str]:
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
        with self.lock:
            self._expire()
            self.items[token] = {"user": user, "csrf": csrf, "expires": time.time() + self.ttl}
        return token, csrf

    def get(self, token: str | None) -> dict | None:
        if not token:
            return None
        with self.lock:
            s = self.items.get(token)
            if s and s["expires"] < time.time():
                self.items.pop(token, None)
                return None
            return s

    def drop(self, token: str | None) -> None:
        with self.lock:
            self.items.pop(token or "", None)

    def drop_user(self, user: str, keep: str | None = None) -> None:
        with self.lock:
            for t in [t for t, s in self.items.items() if s["user"] == user and t != keep]:
                self.items.pop(t)

    def _expire(self) -> None:
        now = time.time()
        for t in [t for t, s in self.items.items() if s["expires"] < now]:
            self.items.pop(t)


class RateLimit:
    """At most `n` events per `window` seconds per key (login attempts)."""

    def __init__(self, n: int = 5, window: float = 60.0):
        self.n, self.window = n, window
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.time()
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.n:
                raise HTTPException(429, f"too many attempts, wait {int(self.window - (now - q[0])) + 1} s")
            q.append(now)

    def reset(self, key: str) -> None:
        with self.lock:
            self.hits.pop(key, None)


class Audit:
    """Append-only JSON lines: who did what, when, with which parameters and result."""

    MAX_BYTES = 20 * 1024 * 1024

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def log(self, user: str, action: str, params: dict | None = None, result: str = "ok") -> None:
        entry = {"t": time.time(), "user": user, "action": action, "params": _redact(params or {}),
                 "result": str(result)[:500]}
        with self.lock:
            if self.path.exists() and self.path.stat().st_size > self.MAX_BYTES:
                os.replace(self.path, self.path.with_suffix(".1.jsonl"))
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def tail(self, limit: int = 200) -> list[dict]:
        if not self.path.exists():
            return []
        with self.lock:
            lines = self.path.read_text().splitlines()[-limit:]
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


def _redact(params: dict) -> dict:
    return {k: ("***" if "pass" in k.lower() or "token" in k.lower() else v) for k, v in params.items()}


class Auth:
    """FastAPI dependencies: `auth.user(request)` for any logged-in request; CSRF on state changes."""

    def __init__(self, users: Users, sessions: Sessions, audit: Audit):
        self.users, self.sessions, self.audit = users, sessions, audit

    def session(self, request: Request) -> dict:
        s = self.sessions.get(request.cookies.get(COOKIE))
        if s is None:
            raise HTTPException(401, "login required")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            sent = request.headers.get("x-csrf-token", "")
            if not hmac.compare_digest(sent, s["csrf"]):
                raise HTTPException(403, "missing or invalid CSRF token")
        return s

    def user(self, request: Request) -> str:
        return self.session(request)["user"]


def client_key(request: Request) -> str:
    return request.client.host if request.client else "?"
