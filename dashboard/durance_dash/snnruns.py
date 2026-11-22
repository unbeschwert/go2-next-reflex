"""Runs of our training pipeline (python -m training.snn with a YAML config, see training/README.md).

Everything the dashboard needs to know about a run's network comes from the files the pipeline writes, so a
new YAML config (other layers, input size, classes, formats) needs no change here:

    runs/<name>/config.yaml                    resolved config: classes, input size, splits
    runs/<name>/metrics.json                   accuracies (fixed-point = what the FPGA computes)
    runs/<name>/deploy/weights.txt             the SNN parameters for /root/weights.txt
    runs/<name>/deploy/model.json              the network graph the firmware is built from
    runs/<name>/deploy/export_summary.json     weight and potential formats (Q1.7, Q4.12, ...)
    runs/<name>/deploy/spleat-server-config.yaml, metavision_camera.json   board configs for this network

Firmware is any folder below runs/ with spleat.bit.bin + spleat.dtbo + the model.json it was built from
(model.json or job<ID>-model.json) + job.json with its formats, as scripts/firmware_job.py downloads it
(runs/<name>/firmware-job<ID>/) or as a board setup folder holds it. A firmware fits a run when both
model.json graphs are the same and the formats match.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

# Class lists the board's gesture bridge knows (kria/live_gesture_bridge.py --classes own|dataset), by output index.
CLASS_SETS = {"dataset": ["sit", "lay-down", "hello", "dance", "no-action"],
              "own": ["heart", "turn-left-360", "turn-right-360", None, "no-action"]}   # output 3: follow me / unused
DEPLOY_FILES = ("weights.txt", "model.json", "export_summary.json", "spleat-server-config.yaml",
                "metavision_camera.json")

_md5: dict[str, tuple[int, int, str]] = {}


def md5_of(path: Path) -> str:
    """md5 of a file, cached by size and mtime."""
    st = path.stat()
    hit = _md5.get(str(path))
    if hit and hit[:2] == (st.st_size, st.st_mtime_ns):
        return hit[2]
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    _md5[str(path)] = (st.st_size, st.st_mtime_ns, h.hexdigest())
    return h.hexdigest()


def class_set(names: list[str]) -> str | None:
    """'dataset' or 'own' when the run's classes are in the bridge's output order, else None."""
    n = [str(x).lower() for x in names]
    for key, want in CLASS_SETS.items():
        if len(n) == len(want) and all(w is None or w == c for w, c in zip(want, n)):
            return key
    return None


def graph_key(model_json: dict) -> str:
    """Canonical form of a model.json graph: two firmwares are interchangeable when these are equal."""
    keep = {k: model_json.get(k) for k in ("input", "nodes", "edges", "outputs")}
    return json.dumps(keep, sort_keys=True)


def network(model_json: dict) -> dict | None:
    """Input size, (in, out, kernel, stride, padding) per layer and the output shape of a plain ConvLIF2d chain
    (what our pipeline exports); None for other graphs (skips, several outputs, upscaling)."""
    try:
        inp, nodes, edges = model_json["input"], model_json["nodes"], model_json["edges"]
        names = [n["node_index"] for n in nodes]
        chain = [{"source": a, "target": b} for a, b in zip(["input"] + names[:-1], names)]
        if edges != chain or model_json.get("outputs") != names[-1:]:
            return None
        ci, h = int(inp["channels"]), int(inp["height"])
        if int(inp["width"]) != h:
            return None
        layers = []
        for n in nodes:
            pad = n.get("padding", [0, 0, 0, 0])
            if n.get("node_type") != "ConvLIF2d" or n.get("upscaling", 1) != 1 or len(set(pad)) != 1:
                return None
            k, s, p, co = int(n["kernel_size"]), int(n["stride"]), int(pad[0]), int(n["channels"])
            layers.append((ci, co, k, s, p))
            h = (h + 2 * p - k) // s + 1
            ci = co
        return {"size": int(inp["height"]), "layers": layers, "output": [ci, h, h]}
    except (KeyError, TypeError, ValueError):
        return None


def q_name(int_bits: int, frac_bits: int) -> str:
    return f"Q{int_bits}.{frac_bits}"


def parse_q(text: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"Q(\d+)\.(\d+)", str(text or "").strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def server_config(text: str) -> dict:
    """inputs/outputs/scaling of a spleat-server config.yaml."""
    try:
        y = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    i, o = y.get("inputs") or {}, y.get("outputs") or {}
    return {"input": i.get("height"), "scaling": i.get("scaling"),
            "outputs": [o.get("channels"), o.get("height", 1), o.get("width", 1)]}


def _first(d: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        hit = sorted(d.glob(pat))
        if hit:
            return hit[0]
    return None


def firmwares(runs_dir: Path) -> list[dict]:
    """Every firmware folder below runs/ (two levels deep): what network and formats it was built for."""
    out = []
    if not runs_dir or not runs_dir.is_dir():
        return out
    for bit in sorted(list(runs_dir.glob("*/spleat.bit.bin")) + list(runs_dir.glob("*/*/spleat.bit.bin"))):
        d = bit.parent
        dtbo, job, mj = d / "spleat.dtbo", d / "job.json", _first(d, "model.json", "job*-model.json")
        if not (dtbo.is_file() and job.is_file() and mj):
            continue
        try:
            j = json.loads(job.read_text())
            graph = json.loads(mj.read_text())
        except (OSError, ValueError):
            continue
        if not (parse_q(j.get("weights")) and parse_q(j.get("potentials"))):
            continue
        rel = d.relative_to(runs_dir)
        out.append({"dir": str(d), "rel": str(rel), "job": j.get("id"), "name": j.get("name"),
                    "label": (f"firmware job {j['id']}" if j.get("id") is not None else "firmware") + f" ({rel})",
                    "weights": j["weights"], "potentials": j["potentials"], "strategy": j.get("strategy"),
                    "bit_md5": md5_of(bit), "dtbo_md5": md5_of(dtbo), "graph": graph_key(graph),
                    "network": network(graph), "mtime": bit.stat().st_mtime})
    return out


def firmware_public(fw: dict | None) -> dict | None:
    if not fw:
        return None
    return {k: fw[k] for k in ("rel", "job", "label", "weights", "potentials", "strategy")} | {"bit_md5": fw["bit_md5"][:8]}


def _metrics(run: Path) -> dict | None:
    try:
        j = json.loads((run / "metrics.json").read_text())
    except (OSError, ValueError):
        return None
    for key in ("fixed_point", "qat", "float"):           # fixed point = what the FPGA computes
        r = j.get(key) or {}
        if r.get("test_acc") is not None:
            return {"val": r.get("val_acc"), "test": r.get("test_acc"), "kind": key}
    return None


def scan(runs_dir: Path | None) -> list[dict]:
    """Deployable runs under runs/: one entry per runs/<name>/deploy/ with all the files of DEPLOY_FILES."""
    if not runs_dir or not runs_dir.is_dir():
        return []
    fws = firmwares(runs_dir)
    out = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        dep = run / "deploy"
        if not all((dep / f).is_file() for f in DEPLOY_FILES) or not (run / "config.yaml").is_file():
            continue
        try:
            cfg = yaml.safe_load((run / "config.yaml").read_text()) or {}
            graph = json.loads((dep / "model.json").read_text())
            summary = json.loads((dep / "export_summary.json").read_text())
            camera = json.loads((dep / "metavision_camera.json").read_text())
            server_text = (dep / "spleat-server-config.yaml").read_text()
        except (OSError, ValueError, yaml.YAMLError):
            continue
        f = summary.get("formats") or {}
        try:
            formats = {"weights": q_name(f["weight_int"], f["weight_frac"]),
                       "potentials": q_name(f["potential_int"], f["potential_frac"]),
                       "weight_frac": int(f["weight_frac"]), "potential_frac": int(f["potential_frac"]),
                       "weight_bits": int(f["weight_int"]) + int(f["weight_frac"])}
        except (KeyError, TypeError, ValueError):
            continue
        net = network(graph)
        key = graph_key(graph)
        fits = [fw for fw in fws if fw["graph"] == key and fw["weights"] == formats["weights"]
                and fw["potentials"] == formats["potentials"]]
        # prefer the run's own firmware, then the newest one built for the same network
        fits.sort(key=lambda fw: (Path(fw["dir"]).is_relative_to(run), fw["mtime"]), reverse=True)
        data = cfg.get("data") or {}
        names = [str(c) for c in data.get("classes") or []]
        weights = dep / "weights.txt"
        st = weights.stat()
        sc = server_config(server_text)
        problems = deploy_problems(names, net, sc, formats, data, camera, bool(fits))
        out.append({
            "id": "run-" + hashlib.sha1(run.name.encode()).hexdigest()[:10], "name": run.name,
            "path": str(weights), "source": "runs", "kind": "setup",
            "arch": str(cfg.get("name") or run.name), "input": net["size"] if net else sc.get("input"),
            "classes": class_set(names), "class_names": names,
            "metrics": _metrics(run), "bits": str(formats["weight_bits"]), "formats": formats,
            "md5": md5_of(weights), "values": len(weights.read_bytes().split()), "bytes": st.st_size, "mtime": st.st_mtime,
            "network": net, "graph": key, "splits": {k: data.get(k) for k in ("train", "val", "test")},
            "downsampling": camera.get("downsampling"), "frequency": camera.get("frequency"),
            "server": sc, "firmware": firmware_public(fits[0]) if fits else None,
            "firmware_dir": fits[0]["dir"] if fits else None, "run_dir": str(run), "problems": problems,
        })
    return out


def expected_scaling(data: dict, potential_frac: int) -> float | None:
    """spleat-server inputs.scaling for despot's block sums (doc/deploy-model.md): 2^frac / downsampling² for
    'mean' frames, 2^frac for 'sum' frames."""
    ds, enc = data.get("downsampling"), data.get("encoding", "mean")
    if not ds:
        return None
    return (1 << potential_frac) / ds ** 2 if enc == "mean" else float(1 << potential_frac) if enc == "sum" else None


def deploy_problems(names, net, sc, formats, data, camera, has_firmware) -> list[str]:
    """Why a run cannot be installed on the board as it is (empty: it can)."""
    p = []
    if class_set(names) is None:
        p.append(f"classes {names} are not in an order the board's gesture bridge knows "
                 f"({CLASS_SETS['dataset']} or heart, turn-left-360, turn-right-360, <any>, no-action)")
    if net is None:
        p.append("model.json is not a plain chain of ConvLIF2d layers")
    elif sc.get("input") != net["size"] or sc.get("outputs") != net["output"]:
        p.append(f"spleat-server-config.yaml (input {sc.get('input')}, outputs {sc.get('outputs')}) does not match "
                 f"model.json (input {net['size']}, outputs {net['output']})")
    want = expected_scaling(data, formats["potential_frac"])
    if want is not None and sc.get("scaling") is not None and abs(float(sc["scaling"]) - want) > 1e-6:
        p.append(f"server config scaling is {sc['scaling']:g}, but despot's block sums need {want:g}: "
                 "re-export the run (python -m training.snn export --run-dir runs/<name>)")
    if net and camera.get("downsampling") and 320 // int(camera["downsampling"]) != net["size"]:
        p.append(f"camera downsampling {camera['downsampling']} does not give the {net['size']}x{net['size']} input")
    if not has_firmware:
        p.append(f"no firmware for this network with weights {formats['weights']} / potentials "
                 f"{formats['potentials']} below runs/ (build one: uv run scripts/firmware_job.py submit --run "
                 "runs/<name> --download)")
    return p


def public(m: dict) -> dict:
    """A run entry without local paths (for the browser)."""
    return {k: v for k, v in m.items() if k not in ("path", "firmware_dir", "run_dir", "graph")}


def find_firmware(runs_dir: Path | None, bit_md5: str | None) -> dict | None:
    if not bit_md5:
        return None
    return next((fw for fw in firmwares(runs_dir) if fw["bit_md5"] == bit_md5), None) if runs_dir else None


def setup_files(m: dict, board_camera: dict | None) -> dict[str, bytes]:
    """The six files of a board setup for a run (names as in firmware_performance_testing/switch_setup.sh).
    The camera config keeps the board's sensor settings (biases, noise filter) and takes the run's
    downsampling and frame rate, which the network depends on."""
    if not m.get("firmware_dir"):
        raise ValueError("no firmware for this network")
    dep, fw = Path(m["path"]).parent, Path(m["firmware_dir"])
    camera = dict(board_camera) if isinstance(board_camera, dict) and board_camera else \
        json.loads((dep / "metavision_camera.json").read_text())
    run_cam = json.loads((dep / "metavision_camera.json").read_text())
    for k in ("downsampling", "frequency"):
        camera[k] = run_cam[k]
    return {"spleat.bit.bin": (fw / "spleat.bit.bin").read_bytes(), "spleat.dtbo": (fw / "spleat.dtbo").read_bytes(),
            "config.yaml": (dep / "spleat-server-config.yaml").read_bytes(),
            "metavision_camera.json": (json.dumps(camera, indent=2) + "\n").encode(),
            "weights.txt": (dep / "weights.txt").read_bytes(), "model_classes": f"{m['classes']}\n".encode()}
