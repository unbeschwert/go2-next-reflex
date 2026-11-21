"""Same starting state on both sides: 300 zero frames, then the clip (one connection). Compare clip frames."""
import sys, random
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts, NAME_RE
from training.snn.train import load_qat_model

run = Path("runs/motion-buddy-40-v1"); cfg = load_config(run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()
WARM = 300
files = sorted(Path(cfg.data.root, "13").glob("gesture_*.bin")); random.Random(1).shuffle(files)
res = []
for f in files[:10]:
    clip = events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
    seq = np.concatenate([np.zeros((WARM, 2, S, S), np.float32), clip])
    c = FPGAClient("durance-kria.local", 3333, shape_in=(2, S, S), shape_out=(cfg.n_classes, 1, 1)); c.connect()
    out = []
    for fr in seq:
        c.send_tensor(fr); out.append(c.receive_tensor()[0].ravel())
    c.close()
    fp = np.array(out)[WARM:].astype(int)
    with torch.no_grad():
        em = emu.net(torch.from_numpy(seq)[None])[0, WARM:, :, 0, 0].numpy().astype(int)
    first = next((t for t in range(T) if not np.array_equal(fp[t], em[t])), None)
    label = NAME_RE.match(f.name).group(2)
    res.append((np.mean(fp == em), fp.sum(0).argmax() == em.sum(0).argmax(), first))
    print(f"{f.name:38s} identical spikes {np.mean(fp == em):6.1%}  first diff at frame {first}  "
          f"FPGA {fp.sum(0)}  emu {em.sum(0)}  ({label})")
print(f"mean identical spikes {np.mean([r[0] for r in res]):.1%}, same prediction {sum(r[1] for r in res)}/{len(res)}")
