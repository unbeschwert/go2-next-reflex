"""Reset by restarting spleat-server before every sample; compare with emulation started from zero potentials."""
import random, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts, NAME_RE
from training.snn.train import load_qat_model

KRIA = "root@durance-kria.local"
run = Path("runs/motion-buddy-40-v1"); cfg = load_config(run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()

def restart_server():
    # systemd allows 5 starts per minute (StartLimitBurst); reset-failed clears that counter so we can
    # restart once per sample without the service ending up in 'failed (start-limit-hit)'.
    subprocess.run(["ssh", "-o", "BatchMode=yes", KRIA,
                    "systemctl reset-failed spleat-server.service; systemctl restart spleat-server.service"], check=True)
    for _ in range(50):                                   # wait until the port accepts connections
        try:
            c = FPGAClient("durance-kria.local", 3333, (2, S, S), (cfg.n_classes, 1, 1)); c.connect(); c.close()
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("spleat-server did not come back")

def stream(frames):
    c = FPGAClient("durance-kria.local", 3333, (2, S, S), (cfg.n_classes, 1, 1)); c.connect()
    out = []
    for f in frames:
        c.send_tensor(f); out.append(c.receive_tensor()[0].ravel())
    c.close(); return np.array(out).astype(int)

def emulate(frames):
    with torch.no_grad():
        return emu.net(torch.from_numpy(frames)[None])[0, :, :, 0, 0].numpy().astype(int)

zeros = np.zeros((T, 2, S, S), np.float32)
print("zeros after restart vs emulation from zero potentials:")
for i in range(3):
    restart_server(); fp = stream(zeros)
    em = emulate(zeros)
    print(f"  restart {i}: FPGA {fp.sum(0)}  emu {em.sum(0)}  identical: {np.array_equal(fp, em)} ({np.mean(fp == em):.1%})")

files = sorted(Path(cfg.data.root, "13").glob("gesture_*.bin")); random.Random(1).shuffle(files)
res = []
for f in files[:10]:
    clip = events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
    restart_server(); fp = stream(clip); em = emulate(clip)
    first = next((t for t in range(T) if not np.array_equal(fp[t], em[t])), None)
    res.append((np.mean(fp == em), fp.sum(0).argmax() == em.sum(0).argmax()))
    print(f"  {f.name:38s} identical {np.mean(fp == em):6.1%}  first diff at frame {first}  FPGA {fp.sum(0)}  emu {em.sum(0)}")
print(f"mean identical spikes {np.mean([r[0] for r in res]):.1%}, same prediction {sum(r[1] for r in res)}/{len(res)}")
