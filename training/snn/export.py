"""Everything the Kria needs, in run_dir/deploy/:

    model.json                 layer graph → riversdk.dev "New firmware"
    weights.txt                integers → /root/weights.txt
    spleat-server-config.yaml  → /root/.config/spleat-server/config.yaml
    metavision_camera.json     → /root/metavision_camera.json
    parameters.json            the same integers, per layer (for inspection)
    memory_report.md/.json     BRAM/URAM estimate for the chosen formats
    DEPLOY.md                  the firmware-form values and copy commands, filled in
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from riverops.export import export_for_river

from . import fpga
from .config import Config
from .model import build_model, layer_shapes
from .quant import Formats, saturation

CAMERA_BIASES = {"bias_diff": 51, "bias_diff_off": 40, "bias_diff_on": 40, "bias_fo": 29, "bias_hpf": 0,
                 "bias_refr": 82}


def export_deployment(qat_model, formats: Formats, cfg: Config, run_dir: Path, metrics: dict | None = None) -> dict:
    out = run_dir / "deploy"
    out.mkdir(parents=True, exist_ok=True)
    size, n = cfg.data.size, cfg.n_classes
    qat_model = qat_model.cpu().eval()

    # Layer graph. Only the structure matters, so export a fresh copy of the architecture.
    example = torch.zeros(1, 2, 2, size, size)
    (out / "model.json").write_text(export_for_river(build_model(cfg).net.eval(), example))

    # Integer parameters: all weights, all biases, then threshold/leak per layer.
    layers = [layer.export_params() for layer in qat_model.net]
    weights = [v for p in layers for v in p["weights"]]
    biases = [v for p in layers for v in p["bias"]]
    neurons = [v for p in layers for v in (p["threshold"], p["leak"])]
    values = weights + biases + neurons
    expected = sum(s.weights + s.out_c + 2 for s in layer_shapes(cfg))
    if len(values) != expected:
        raise RuntimeError(f"exported {len(values)} parameters, expected {expected}")
    (out / "weights.txt").write_text("\n".join(str(int(v)) for v in values))   # no trailing newline
    (out / "parameters.json").write_text(json.dumps(
        {f"layer_{i}": p for i, p in enumerate(layers)}, indent=1) + "\n")

    sat = {
        "weights": saturation(weights, formats.weight),
        "biases": saturation(biases, formats.weight),
        "per_layer_weights": [saturation(p["weights"], formats.weight) for p in layers],
    }

    # Kria-side configuration, derived from the same config as training.
    server = {
        "application": {"host": "0.0.0.0", "port": 3333, "weights_file": "/root/weights.txt", "max_concurrency": 40},
        "inputs": {"channels": 2, "height": size, "width": size, "scaling": float(2 ** formats.potential_frac)},
        "outputs": {"channels": n, "height": 1, "width": 1},
    }
    (out / "spleat-server-config.yaml").write_text(yaml.safe_dump(server, sort_keys=False))
    camera = {"frequency": cfg.data.frequency_hz, "downsampling": cfg.data.downsampling,
              "noise_filtering_factor": 0.5, "biases": CAMERA_BIASES}
    (out / "metavision_camera.json").write_text(json.dumps(camera, indent=2) + "\n")

    memory = fpga.estimate(cfg, weight_bits=formats.weight_int + formats.weight_frac,
                           potential_bits=formats.potential_int + formats.potential_frac)
    (out / "memory_report.json").write_text(json.dumps(memory, indent=1) + "\n")
    (out / "memory_report.md").write_text(fpga.report_markdown(memory, cfg))

    summary = {"parameters": len(values), "saturation": sat, "memory_verdict": memory["verdict"],
               "formats": formats.to_dict(),
               # per layer, in potential-format integers; != 2^potential_frac where a layer is scaled
               "thresholds": [int(p["threshold"]) for p in layers]}
    (out / "DEPLOY.md").write_text(deploy_markdown(cfg, formats, memory, sat, metrics))
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    per_layer = ", ".join(f"{v:.1%}" for v in sat["per_layer_weights"])
    print(f"exported to {out}: {len(values):,} parameters, weights clipped {sat['weights']:.2%} "
          f"(per layer: {per_layer}), biases clipped {sat['biases']:.2%}", flush=True)
    print(fpga.report_text(memory), flush=True)
    return summary


def deploy_markdown(cfg: Config, f: Formats, memory: dict, sat: dict, metrics: dict | None) -> str:
    acc = ""
    if metrics:
        fp = metrics["fixed_point"]
        acc = (f"Fixed-point accuracy (closest to the FPGA): validation {fp['val_acc']:.1%}, "
               f"test {fp['test_acc']:.1%}.\n\n")
    warn = ""
    worst = max(range(len(sat["per_layer_weights"])), key=lambda i: sat["per_layer_weights"][i])
    if sat["per_layer_weights"][worst] > 0.01 or sat["biases"] > 0.01:
        # Check per layer: a small layer can be badly clipped while the overall figure looks fine.
        per_layer = ", ".join(f"layer {i}: {v:.1%}" for i, v in enumerate(sat["per_layer_weights"]))
        warn = (f"> **Warning:** weights clipped at the format limit ({per_layer}); biases {sat['biases']:.1%}. "
                f"Layer {worst} needs more range than the shared weight format gives it.\n\n")
    if memory["verdict"] == "too_big":
        warn += "> **Warning:** the memory estimate says this model does not fit the KV260.\n\n"
    classes = ", ".join(f"{i} = {c}" for i, c in enumerate(cfg.data.classes))
    return f"""# Deploy `{cfg.name}` to the Kria KV260

{acc}{warn}Output neurons: {classes}.

## 1. Generate the firmware: riversdk.dev → Open firmware jobs → New firmware

| Field | Value |
|---|---|
| Model file | `model.json` |
| Board | **KV260** |
| Prophesee IP | **GenX320** |
| Weights quantization | **Q{f.weight_int}.{f.weight_frac}** |
| Potentials quantization | **Q{f.potential_int}.{f.potential_frac}** |
| Implementation strategy | **{cfg.fpga.strategy.replace('-', ' ').capitalize()}** |

The two formats **must** be exactly these: `weights.txt` holds integers in these formats, and the form
defaults (Q4.12 / Q4.12) are different. Download `firmware-kv260.tar.gz` when the job is done and unpack it
next to this file (it should contain the bitstream `spleat.bit.bin` and the overlay `spleat.dtbo`).
Or from the command line: `uv run scripts/firmware_job.py submit --run runs/<this run> --download`
(reuses an existing job when the model and options are identical; see doc/deploy-model.md).

Memory estimate: {memory['verdict']}: {memory['bram_only']:.1f} of {memory['bram_total']} BRAM36 if all in
BRAM; see `memory_report.md`.

## 2. Copy to the board

```bash
K=root@durance-kria.local
scp spleat.bit.bin spleat.dtbo $K:/usr/lib/firmware/xilinx/spleat/
scp weights.txt $K:/root/weights.txt
scp spleat-server-config.yaml $K:/root/.config/spleat-server/config.yaml
scp metavision_camera.json $K:/root/metavision_camera.json
```

## 3. Start it (on the Kria)

```bash
systemctl stop spleat-server.service
xmutil loadapp spleat
backward-server -c /root/.config/spleat-server/config.yaml     # leave running, shows logs
# second SSH session:
despot_test_camera                                             # once after each reboot
despot_send_and_forget --to_display 0
```

The server sends {cfg.n_classes} float32 scores per frame on TCP port 3333. The ROS side
(`go2-kria`) must be changed to expect {cfg.n_classes} scores (see doc/go2-kria-overview.md).

Input frames: {cfg.data.size}x{cfg.data.size}, {cfg.data.time_window_us / 1000:g} ms per frame
(despot downsampling {cfg.data.downsampling}, frequency {cfg.data.frequency_hz:g} Hz), training encoding
`{cfg.data.encoding}`. Check that despot's frames use the same encoding (training/README.md).
"""
