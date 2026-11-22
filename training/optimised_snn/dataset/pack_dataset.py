#!/usr/bin/env python3
"""Add our recorded sessions to a copy of the Motion Buddy zip (the Durance AI team's dataset) for training.

The training script lists members matching motion_buddy_dataset/<session>/gesture_<NN>-*.bin,
so every folder data/<session>/ becomes one more session (= one more person).

    python3 pack_dataset.py --sessions 20,21 --out ~/go2-hackathon/data/motion_buddy_plus.zip
    # then:  python train_snn_v8.py --zip motion_buddy_plus.zip --val 13 --test 20
"""
import argparse
import re
import shutil
import struct
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(Path.home() / "Downloads" / "durance-ai-handgesture-recognition (1).zip"))
ap.add_argument("--dataset", default=str(Path.home() / "go2-hackathon" / "data"))
ap.add_argument("--sessions", required=True, help="comma list, e.g. 20,21")
ap.add_argument("--out", required=True)
a = ap.parse_args()

out = Path(a.out)
sessions = [f"{int(s):02d}" for s in a.sessions.split(",")]
with zipfile.ZipFile(a.base) as zf:
    have = {m.group(1) for n in zf.namelist() if (m := re.match(r"motion_buddy_dataset/(\d+)/", n))}
clash = sorted(set(sessions) & have)
if clash:
    raise SystemExit(f"sessions {clash} already exist in the base zip - use new numbers (e.g. 20+)")

print(f"copying {a.base} -> {out}")
shutil.copyfile(a.base, out)
with zipfile.ZipFile(out, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
    for s in sessions:
        files = sorted((Path(a.dataset) / s).glob("gesture_*.bin"))
        bad = 0
        for p in files:
            d = p.read_bytes()
            if len(d) < 17 or struct.unpack("!I", d[:4])[0] != len(d) - 4 or (len(d) - 4) % 13:
                bad += 1
                continue
            zf.writestr(f"motion_buddy_dataset/{s}/{p.name}", d)
        print(f"  session {s}: {len(files) - bad} clips added, {bad} invalid skipped")
print(f"done: {out} ({out.stat().st_size / 1e9:.2f} GB)")
