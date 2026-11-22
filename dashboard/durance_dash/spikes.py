"""A clip's network spikes: replay a dataset clip through the FPGA integer model (go2_viz/spleat_sim.py,
same winner as the FPGA on 98 % of clips) and return the output spikes and each layer's activity per 10 ms frame."""
from __future__ import annotations

import hashlib
import importlib.util
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from . import models as M
from .config import project_file

LABELS = {"own": ["Heart", "Turn left 360", "Turn right 360", "Follow me", "none"],
          "dataset": ["Sit", "Lay down", "Hello", "Dance", "none"]}
DEFAULTS = {"own": ("board_ready/own_gestures_nb40_q8h_weights.txt",                       # laptop layout
                    "optimised_snn/own_gestures/weights_r6b-reg3_056ee113.txt"),              # repository
            "dataset": ("board_ready/dataset_gestures_nb40_q8h_weights.txt",
                        "optimised_snn/dataset_gestures/weights_r4-nb40-qatfull_e9ec792c.txt")}
SETTLE_FRAMES = 60          # blank frames first: the FPGA never resets, its neurons are already running
EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])


def counts(data: bytes, size: int, max_frames: int = 300) -> np.ndarray:
    """Event counts per 10 ms frame, polarity and size x size block: int [T, 2, size, size]."""
    n = int.from_bytes(data[:4], "big") if len(data) >= 4 else -1
    if n != len(data) - 4 or n % EV.itemsize:
        raise HTTPException(422, "not a valid event recording")
    ev = np.frombuffer(data, EV, offset=4)
    if not len(ev):
        return np.zeros((1, 2, size, size), np.int64)
    sc = 320 // size
    t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])) // 10_000
    frames = int(min(max_frames, t.max() + 1))
    ok = (t >= 0) & (t < frames) & (ev["x"] < 320) & (ev["y"] < 320) & (ev["polarity"] < 2)
    idx = ((t[ok] * 2 + ev["polarity"][ok].astype(np.int64)) * size + ev["y"][ok].astype(np.int64) // sc) * size \
        + ev["x"][ok].astype(np.int64) // sc
    return np.bincount(idx, minlength=frames * 2 * size * size).reshape(frames, 2, size, size)


class Spikes:
    def __init__(self, cfg, dataset):
        self.cfg, self.dataset = cfg, dataset
        self.models_dir = Path(cfg.paths.models).expanduser()
        self._sim = None
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def sim_module(self):
        if self._sim is None:
            path = project_file(self.cfg.paths, "spleat_sim.py")
            spec = importlib.util.spec_from_file_location("spleat_sim", path)
            if spec is None or spec.loader is None:
                raise HTTPException(503, f"{path} not found")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._sim = mod
        return self._sim

    def models(self) -> list[dict]:
        archs = self.sim_module().ARCHS
        return [{"id": m["id"], "name": m["name"], "arch": m["arch"], "classes": m["classes"], "bits": m["bits"]}
                for m in M.scan(self.models_dir) if m["arch"] in archs and m["classes"] in LABELS]

    def pick(self, model_id: str | None, session: str) -> dict:
        ms = M.scan(self.models_dir)
        if model_id:
            m = next((x for x in ms if x["id"] == model_id), None)
            if m is None:
                raise HTTPException(404, "unknown model")
        else:
            want = DEFAULTS["own" if int(session) >= 20 else "dataset"]
            m = next((x for x in ms if x["path"] in want), None)
            if m is None:
                raise HTTPException(404, f"default model {want[0]} not found: choose one")
        if m["arch"] not in self.sim_module().ARCHS:
            raise HTTPException(422, f"the FPGA simulator has no {m['arch']} network")
        return m

    def run(self, source: str, rel: str, model_id: str | None) -> dict:
        s = self.dataset._source(source)
        data = self.dataset._read(s, rel)
        m = self.pick(model_id, rel.split("/")[0])
        key = (hashlib.md5(data).hexdigest(), m["md5"])
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        sim_mod = self.sim_module()
        size = sim_mod.ARCHS[m["arch"]][0]
        frames = counts(data, size).astype(np.float32) / ((320 // size) ** 2)
        shifts = dict(w_shift=0, b_shift=0, in_shift=12) if m["bits"] == "16" else dict(w_shift=6, b_shift=6, in_shift=6)
        sim = sim_mod.Spleat(str(self.models_dir / m["path"]), m["arch"], leak_mode="retain", **shifts)
        blank = np.zeros((2, size, size), np.float32)
        for _ in range(SETTLE_FRAMES):
            sim.step(blank)
        outputs, layers = [], []
        for f in frames:
            outputs.append(sim.step(f).astype(np.uint8).tolist())
            layers.append([int(p.sum()) for p in sim.prev])
        neurons = [int(p.size) for p in sim.P]
        out = np.asarray(outputs)
        totals = out.sum(0).tolist()
        labels = LABELS[m["classes"]]
        name = rel.split("/")[1]
        true_cls = int(name.split("-")[0].split("_")[1]) - 1
        res = {"model": {"id": m["id"], "name": m["name"], "arch": m["arch"], "classes": m["classes"],
                         "md5": m["md5"][:8], "bits": m["bits"]},
               "labels": labels, "frames": len(outputs), "fps": 100, "input_size": size,
               "outputs": outputs, "layer_spikes": layers, "neurons": neurons, "totals": totals,
               "rates": [round(t / max(1, len(outputs)), 3) for t in totals],
               "predicted": int(np.argmax(totals)), "true": true_cls if 0 <= true_cls < 5 else None,
               "events_per_frame": [int(v) for v in (frames.reshape(len(frames), -1).sum(1) * ((320 // size) ** 2)).round()],
               "settle_frames": SETTLE_FRAMES, "simulator": "go2_viz/spleat_sim.py"}
        with self._lock:
            self._cache[key] = res
            while len(self._cache) > 64:
                self._cache.popitem(last=False)
        return res
