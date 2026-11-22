"""Gesture-control settings: schema/defaults from go2_viz/bridge_settings.py (shared with the bridge),
values in the bridge's settings file on the board or in the simulator."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from fastapi import HTTPException

from .config import project_file


def load_schema_module(go2_viz: Path):
    spec = importlib.util.spec_from_file_location("bridge_settings", go2_viz / "bridge_settings.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{go2_viz}/bridge_settings.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Settings:
    def __init__(self, cfg, board, sim, live, audit, broker):
        self.cfg, self.board, self.sim, self.live, self.audit, self.broker = cfg, board, sim, live, audit, broker
        self.BS = load_schema_module(project_file(cfg.paths, "bridge_settings.py").parent)
        self.written_t = 0.0

    @property
    def target(self) -> str:
        return self.live.source

    async def context(self) -> tuple[str, bool]:
        """(classes, robot_mode) of the bridge that will read the settings."""
        if self.target == "sim":
            return "own", False
        st = await self.board.raw_status()
        classes = st.get("model_classes") if st.get("model_classes") in ("own", "dataset") else "own"
        return classes, str(st.get("mode", "")).startswith("robot")

    async def read_file(self) -> dict:
        if self.target == "sim":
            p = self.sim.dir / "bridge_settings.json"
            try:
                v = json.loads(p.read_text()) if p.exists() else {}
                return v if isinstance(v, dict) else {}
            except ValueError:
                return {}
        return await self.board.read_settings()

    async def write_file(self, values: dict) -> None:
        if self.target == "sim":
            self.sim.dir.mkdir(parents=True, exist_ok=True)
            self.BS.save(str(self.sim.dir / "bridge_settings.json"), values)
        else:
            await self.board.write_settings(values)
        self.written_t = time.time()

    async def get(self) -> dict:
        classes, robot = await self.context()
        defaults = self.BS.defaults(classes, robot)
        raw = await self.read_file()
        clean, errors = self.BS.validate({k: v for k, v in raw.items() if not k.startswith("_")}, classes)
        values = dict(defaults, **clean)
        live = self.live.last or {}
        applied = bool(live) and live.get("t", 0) > self.written_t + 0.3 and not live.get("settings_errors")
        return {"schema": self.BS.schema(classes), "defaults": defaults, "values": values,
                "classes": self.BS.LABELS.get(classes, self.BS.LABELS["own"]), "mode": "robot" if robot else "test",
                "target": self.target, "applied": applied, "file_errors": errors}

    async def put(self, user: str, values: dict, confirm: bool = False) -> dict:
        if not isinstance(values, dict) or not values:
            raise HTTPException(422, "values: expected a non-empty object")
        classes, _ = await self.context()
        clean, errors = self.BS.validate(values, classes)
        if errors:
            raise HTTPException(422, "; ".join(f"{k}: {v}" for k, v in errors.items()))
        if self.target == "board":
            await self.board.ensure_lock(user, "gesture-control settings", confirm=confirm)
        current = {k: v for k, v in (await self.read_file()).items() if not k.startswith("_")}
        current, _ = self.BS.validate(current, classes)
        current.update(clean)
        await self.write_file(current)
        self.audit.log(user, "settings.put", clean)
        return await self.get()

    async def reset(self, user: str, key: str | None, confirm: bool = False) -> dict:
        classes, _ = await self.context()
        known = {r["key"] for r in self.BS.schema(classes)}
        if key is not None and key not in known:
            raise HTTPException(422, f"unknown setting {key}")
        if self.target == "board":
            await self.board.ensure_lock(user, "gesture-control settings", confirm=confirm)
        current, _ = self.BS.validate(await self.read_file(), classes)
        if key is None:                            # reset all, but a pause (E-STOP / manual) stays as it is
            current = {"paused": current["paused"]} if "paused" in current else {}
        else:
            current.pop(key, None)
        await self.write_file(current)
        self.audit.log(user, "settings.reset", {"key": key or "all"})
        return await self.get()

    async def set_paused(self, user: str, paused: bool, reason: str, bypass_lock: bool = False) -> None:
        """Pause / resume gesture control. E-STOP uses bypass_lock: safety never waits for a lock."""
        classes, _ = await self.context()
        if self.target == "board" and not bypass_lock:
            await self.board.ensure_lock(user, reason, confirm=False)
        current, _ = self.BS.validate(await self.read_file(), classes)
        current["paused"] = bool(paused)
        await self.write_file(current)
        self.audit.log(user, "settings.paused", {"paused": paused, "reason": reason})
