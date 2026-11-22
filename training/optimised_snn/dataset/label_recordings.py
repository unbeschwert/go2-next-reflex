#!/usr/bin/env python3
"""File new river-tools recordings into the Motion Buddy dataset layout.

river-tools saves clips as samples/sample_<date>_<time>.bin. Training expects
    <dataset>/<session>/gesture_<NN>-<label>-<style>-<k>.bin
where NN is the class number (01 = first class). This moves every not-yet-filed
sample into that layout, oldest first, numbering k after existing clips.

    python3 label_recordings.py --session 20 --cls 1 --label sit --style normal
    python3 label_recordings.py ... --dry-run          # only show what would happen

Classes of the gesture dataset: 1 sit, 2 lay-down, 3 hello, 4 dance, 5 no-action.
For a new task use your own numbering, e.g. follow: 1 left, 2 centre, 3 right, 4 none.
"""
import argparse
import re
import shutil
import struct
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default=str(Path.home() / "river-captures" / "samples"))
ap.add_argument("--dataset", default=str(Path.home() / "go2-hackathon" / "data"))
ap.add_argument("--session", required=True, help="session folder, e.g. 20 (use a new one per person/place)")
ap.add_argument("--cls", type=int, required=True, help="class number, 1-based")
ap.add_argument("--label", required=True, help="class name used in the file name, e.g. sit")
ap.add_argument("--style", default="normal", help="normal / fast / large ...")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

src = sorted(Path(a.samples).glob("sample_*.bin"))
if not src:
    raise SystemExit(f"no unfiled recordings in {a.samples}")
dst_dir = Path(a.dataset) / f"{int(a.session):02d}"
prefix = f"gesture_{a.cls:02d}-{a.label}-{a.style}-"
existing = [int(m.group(1)) for p in dst_dir.glob(prefix + "*.bin")
            if (m := re.search(r"-(\d+)\.bin$", p.name))]
k = max(existing, default=0)

for p in src:
    data = p.read_bytes()
    n = struct.unpack("!I", data[:4])[0] if len(data) >= 4 else -1
    events = (len(data) - 4) // 13
    if n != len(data) - 4 or n % 13 or events == 0:
        print(f"skip {p.name}: not a complete recording")
        continue
    k += 1
    target = dst_dir / f"{prefix}{k:02d}.bin"
    print(f"{p.name} ({events} events) -> {target}")
    if not a.dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), target)
print("dry run - nothing moved" if a.dry_run else f"filed into {dst_dir}")
