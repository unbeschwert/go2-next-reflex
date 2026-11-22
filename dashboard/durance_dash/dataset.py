"""Gesture datasets: browse, play clips, relabel / reject clips (folder sources only)."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import threading
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from .config import project_file
from .errors import need

EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
NAME_RE = re.compile(r"^gesture_(\d{2})-([a-z0-9-]+?)-([a-z]+)-(\d+)\.bin$")
PATH_RE = re.compile(r"^(\d{1,3})/(gesture_\d{2}-[a-z0-9-]+-[a-z]+-\d+\.bin)$")
ORGANISERS = {1: "sit", 2: "lay-down", 3: "hello", 4: "dance", 5: "no-action"}
SIZES = (40, 80, 160)
MAX_FRAMES = 300


class Dataset:
    def __init__(self, cfg, audit):
        self.cfg, self.audit = cfg, audit
        p = cfg.paths
        self.data = Path(p.data).expanduser()
        self._zip_index: dict[str, tuple[int, list]] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- sources
    def own_classes(self) -> dict[int, str]:
        g = json.loads((project_file(self.cfg.paths, "gestures.json")).read_text())
        return {c["id"]: c["name"] for c in g["classes"]}

    def sources(self) -> dict:
        out = [{"id": "data", "name": f"Dataset folder ({self.data})", "kind": "folder", "writable": True,
                "path": str(self.data)}]
        share = Path(self.cfg.paths.share).expanduser()
        for z in sorted(share.glob("*.zip")):
            out.append({"id": "share:" + z.name, "name": f"share/{z.name}", "kind": "zip", "writable": False, "path": str(z)})
        org = Path(self.cfg.paths.organisers_zip).expanduser()
        if org.exists():
            out.append({"id": "organisers", "name": "Organisers' Motion Buddy dataset (zip)", "kind": "zip",
                        "writable": False, "path": str(org)})
        for s in out:
            items = self._items(s)
            s["sessions"] = sorted({i[0] for i in items}, key=lambda x: int(x))
            names: dict[str, Counter] = {}
            for _, f, _ in items:
                m = NAME_RE.match(f)
                if m:
                    names.setdefault(str(int(m.group(1))), Counter())[m.group(2)] += 1
            # the folder mixes the organisers' sessions (01-19: sit, ...) with ours (20+: heart, ...)
            s["classes"] = {k: " / ".join(n for n, _ in v.most_common()) for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))}
            s.pop("path")
        return {"sources": out}

    def _source(self, sid: str) -> dict:
        if sid == "data":
            return {"id": sid, "kind": "folder", "writable": True, "path": self.data}
        if sid == "organisers":
            p = Path(self.cfg.paths.organisers_zip).expanduser()
        elif sid.startswith("share:") and re.fullmatch(r"share:[\w.-]+\.zip", sid):
            p = Path(self.cfg.paths.share).expanduser() / sid[6:]
        else:
            raise HTTPException(404, "unknown source")
        if not p.exists():
            raise HTTPException(404, "source not found")
        return {"id": sid, "kind": "zip", "writable": False, "path": p}

    def _items(self, s: dict) -> list[tuple[str, str, int]]:
        """(session, file name, bytes) of every clip of a source."""
        path = Path(s["path"])
        if s["kind"] == "folder":
            out = []
            for d in path.iterdir() if path.is_dir() else []:
                if d.is_dir() and d.name.isdigit():
                    out += [(d.name, f.name, f.stat().st_size) for f in d.glob("gesture_*.bin") if NAME_RE.match(f.name)]
            return out
        st = path.stat()
        with self._lock:
            hit = self._zip_index.get(str(path))
            if hit and hit[0] == st.st_mtime_ns:
                return hit[1]
            items = []
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    m = re.match(r"^motion_buddy_dataset/(\d+)/(gesture_[^/]+\.bin)$", info.filename)
                    if m and NAME_RE.match(m.group(2)):
                        items.append((m.group(1), m.group(2), info.file_size))
            self._zip_index[str(path)] = (st.st_mtime_ns, items)
            return items

    # ---------------------------------------------------------------- listing
    def summary(self, sid: str) -> dict:
        s = self._source(sid)
        counts: dict[str, Counter] = {}
        names: dict[str, str] = {}
        seen: dict[str, Counter] = {}
        for sess, f, _ in self._items(s):
            m = NAME_RE.match(f)
            cls = str(int(m.group(1)))
            seen.setdefault(cls, Counter())[m.group(2)] += 1
            counts.setdefault(sess, Counter())[cls] += 1
        names = {k: " / ".join(n for n, _ in v.most_common()) for k, v in seen.items()}
        return {"classes": dict(sorted(names.items(), key=lambda kv: int(kv[0]))),
                "sessions": [{"session": k, "counts": dict(v), "total": sum(v.values())}
                             for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))]}

    def clips(self, sid: str, session: str | None, cls: str | None, offset: int, limit: int) -> dict:
        s = self._source(sid)
        limit = max(1, min(int(limit), 500))
        rows = []
        for sess, f, size in sorted(self._items(s), key=lambda i: (int(i[0]), i[1])):
            if session and sess != session:
                continue
            m = NAME_RE.match(f)
            if cls and str(int(m.group(1))) != str(cls):
                continue
            rows.append({"path": f"{sess}/{f}", "session": sess, "cls": int(m.group(1)), "label": m.group(2),
                         "style": m.group(3), "take": int(m.group(4)), "bytes": size})
        return {"total": len(rows), "clips": rows[int(offset):int(offset) + limit]}

    # ---------------------------------------------------------------- one clip
    def _read(self, s: dict, rel: str) -> bytes:
        m = PATH_RE.match(rel or "")
        if not m:
            raise HTTPException(422, "invalid clip path")
        if s["kind"] == "folder":
            p = (s["path"] / m.group(1) / m.group(2)).resolve()
            if s["path"].resolve() not in p.parents or not p.is_file():
                raise HTTPException(404, "clip not found")
            return p.read_bytes()
        with zipfile.ZipFile(s["path"]) as z:
            try:
                return z.read(f"motion_buddy_dataset/{rel}")
            except KeyError:
                raise HTTPException(404, "clip not found")

    def clip(self, sid: str, rel: str, size: int) -> dict:
        if size not in SIZES:
            raise HTTPException(422, f"size must be one of {SIZES}")
        s = self._source(sid)
        data = self._read(s, rel)
        n = int.from_bytes(data[:4], "big") if len(data) >= 4 else -1
        if n != len(data) - 4 or n % EV.itemsize:
            raise HTTPException(422, "not a valid event recording")
        ev = np.frombuffer(data, EV, offset=4)
        sc = 320 // size
        if len(ev):
            t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])) // 10_000
            frames = int(min(MAX_FRAMES, 8_000_000 // (2 * size * size), t.max() + 1))   # answer <= ~11 MB
            ok = (t >= 0) & (t < frames) & (ev["x"] < 320) & (ev["y"] < 320) & (ev["polarity"] < 2)
            idx = ((t[ok] * 2 + ev["polarity"][ok].astype(np.int64)) * size + ev["y"][ok].astype(np.int64) // sc) * size \
                + ev["x"][ok].astype(np.int64) // sc
            counts = np.bincount(idx, minlength=frames * 2 * size * size)
            duration = float((int(ev["timestamp"][-1]) - int(ev["timestamp"][0])) / 1e6)
        else:
            frames, counts, duration = 1, np.zeros(2 * size * size, np.int64), 0.0
        arr = np.minimum(counts, 255).astype(np.uint8)
        m = NAME_RE.match(rel.split("/")[1])
        return {"meta": {"path": rel, "session": rel.split("/")[0], "cls": int(m.group(1)), "label": m.group(2),
                         "style": m.group(3), "events": int(len(ev)), "duration_s": round(duration, 3),
                         "md5": hashlib.md5(data).hexdigest()[:8]},
                "size": size, "frames": frames, "fps": 100, "data": base64.b64encode(arr.tobytes()).decode()}

    # ---------------------------------------------------------------- edits (folder only)
    def _writable(self, sid: str, rel: str) -> tuple[Path, re.Match]:
        s = self._source(sid)
        if not s["writable"]:
            raise HTTPException(409, "this source is read-only (zip): relabel in the dataset folder")
        m = PATH_RE.match(rel or "")
        if not m:
            raise HTTPException(422, "invalid clip path")
        p = (s["path"] / m.group(1) / m.group(2)).resolve()
        if s["path"].resolve() not in p.parents or not p.is_file():
            raise HTTPException(404, "clip not found")
        return p, NAME_RE.match(m.group(2))

    def relabel(self, user: str, sid: str, rel: str, cls: int, confirm: bool) -> dict:
        p, m = self._writable(sid, rel)
        session = int(p.parent.name)
        names = self.own_classes() if session >= 20 else ORGANISERS
        if cls not in names:
            raise HTTPException(422, f"class must be one of {names}")
        if int(m.group(1)) == cls:
            return {"path": rel}
        need(confirm, f"Relabel {p.name} from '{m.group(2)}' to '{names[cls]}'? (The file is renamed; every "
                      "training run after this uses the new label.)")
        style, take = m.group(3), int(m.group(4))
        while True:
            new = p.parent / f"gesture_{cls:02d}-{names[cls]}-{style}-{take:02d}.bin"
            if not new.exists():
                break
            take += 1
        p.rename(new)
        self.audit.log(user, "dataset.relabel", {"from": rel, "to": f"{p.parent.name}/{new.name}"})
        return {"path": f"{p.parent.name}/{new.name}"}

    def reject(self, user: str, sid: str, rel: str, confirm: bool) -> dict:
        p, _ = self._writable(sid, rel)
        need(confirm, f"Move {p.name} out of the dataset (to data/_rejected/{p.parent.name}/)? It can be moved back by hand.")
        dest_dir = self.data / "_rejected" / p.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        if dest.exists():
            dest = dest_dir / f"{p.stem}-dup{len(list(dest_dir.glob(p.stem + '*')))}.bin"
        shutil.move(str(p), str(dest))
        self.audit.log(user, "dataset.reject", {"path": rel, "moved_to": str(dest)})
        return {"moved_to": str(dest.relative_to(self.data))}
