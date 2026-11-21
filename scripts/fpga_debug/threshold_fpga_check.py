"""Does the FPGA use the per-layer threshold from weights.txt? Layer 0 scaled by 1/4 (weights, bias, threshold).

Builds the scaled model from the run's float model (no fine-tuning), exports its weights.txt, loads it on the
Kria (backing up and afterwards restoring the board's weights.txt), and compares FPGA and fixed-point
simulation on N clips with a server restart (= potential reset) before each.
If the hardware ignored the threshold (e.g. fixed 1.0), layer 0 would fire ~4x less and nothing would match.
Usage: uv run --group train python scripts/fpga_debug/threshold_fpga_check.py [k=4] [clips=16]
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq, prepare_qat
from training.snn.config import load_config
from training.snn.data import discover
from training.snn.events import read_events, events_to_counts
from training.snn.model import build_model
from training.snn.train import load_qat_model, zero_tiny_gammas

K = float(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
KRIA, HOST = "root@durance-kria.local", "durance-kria.local"
run = Path("runs/motion-buddy-40-v1"); cfg = load_config(run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
_, formats = load_qat_model(run / "qat_best.pt", cfg)
model = build_model(cfg)
model.load_state_dict(torch.load(run / "float_best.pt", map_location="cpu", weights_only=True)["state_dict"])
zero_tiny_gammas(model); model.eval()
with torch.no_grad():
    model.net[0].bn.weight /= K; model.net[0].bn.bias /= K
model.net[0].lif.v_threshold = cfg.model.neuron.v_threshold / K
qat = prepare_qat(model, formats.riverops(), fused_bn=True).eval()
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()

layers = [layer.export_params() for layer in qat.net]
ints = [int(v) for p in layers for v in np.asarray(p["weights"]).ravel()]
ints += [int(v) for p in layers for v in np.asarray(p["bias"]).ravel()]
ints += [int(v) for p in layers for v in (p["threshold"], p["leak"])]
out = Path("runs/fpga_debug/threshold_k%g" % K); out.mkdir(parents=True, exist_ok=True)
(out / "weights.txt").write_text("\n".join(map(str, ints)))
print("thresholds per layer in weights.txt:", [p["threshold"] for p in layers], " leaks:", [p["leak"] for p in layers])
ref = (run / "deploy" / "weights.txt").read_text().split("\n")
print(f"weights.txt: {len(ints)} values (deployed v1 file: {len(ref)})")
assert len(ints) == len(ref)

def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", KRIA, cmd], check=True, capture_output=True, text=True).stdout

def restart():
    ssh("systemctl reset-failed spleat-server.service; systemctl restart spleat-server.service")
    for _ in range(100):
        try:
            c = FPGAClient(HOST, 3333, (2, S, S), (cfg.n_classes, 1, 1)); c.connect(); return c
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("spleat-server did not come back")

stamp = time.strftime("%Y%m%d-%H%M%S")
ssh(f"cp /root/weights.txt /root/weights.txt.bak-{stamp}")
subprocess.run(["scp", "-q", str(out / "weights.txt"), f"{KRIA}:/root/weights.txt"], check=True)
print(f"loaded scaled weights on the Kria (backup /root/weights.txt.bak-{stamp})")
results = []
try:
    clips = discover(cfg)["val"][::7][:N]
    for i, clip in enumerate(clips):
        x = events_to_counts(read_events(clip.path), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
        c = restart(); f = []
        for frame in x:
            c.send_tensor(frame); f.append(c.receive_tensor()[0].ravel())
        c.close()
        F = np.array(f).astype(int)
        with torch.no_grad():
            E = emu.net(torch.from_numpy(x)[None])[0, :, :, 0, 0].numpy().astype(int)
        results.append((clip.label, F.sum(0).argmax(), E.sum(0).argmax(), float((F == E).mean()),
                        F.sum(0).tolist(), E.sum(0).tolist()))
        print(f"  clip {i + 1:2d}: FPGA {F.sum(0).tolist()}  sim {E.sum(0).tolist()}  identical spikes {(F == E).mean():.1%}", flush=True)
finally:
    ssh(f"cp /root/weights.txt.bak-{stamp} /root/weights.txt; systemctl reset-failed spleat-server.service; systemctl restart spleat-server.service")
    print("restored the board's original weights.txt and restarted spleat-server")
same = np.mean([r[1] == r[2] for r in results]); ident = np.mean([r[3] for r in results])
acc_f = np.mean([r[0] == r[1] for r in results]); acc_e = np.mean([r[0] == r[2] for r in results])
print(f"\nk={K:g}: same prediction FPGA/sim {same:.0%}, identical spikes {ident:.1%}; accuracy on these {len(results)} "
      f"clips: FPGA {acc_f:.0%}, sim {acc_e:.0%}")
(out / "result.json").write_text(json.dumps({"k": K, "same_prediction": same, "identical_spikes": ident,
                                             "clips": [list(map(lambda v: v if isinstance(v, (list, float)) else int(v), r)) for r in results]}, indent=1))
