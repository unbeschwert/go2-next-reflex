#!/usr/bin/env python3
"""Export our recorded sessions as a shareable training zip.

    python3 export_sessions.py --sessions 20            # -> ~/go2-hackathon/share/go2_gestures_s20.zip
    python3 export_sessions.py --sessions 20,21,22 --raw  # also zip the raw takes (for re-cutting)

Zip layout (what train_snn_v3.py --zip reads):
    motion_buddy_dataset/<session>/gesture_<id>-<name>-<style>-<n>.bin
    labels.csv  gestures.json  README.md
"""
import argparse
import csv
import io
import json
import re
import struct
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--sessions", required=True, help="comma list, e.g. 20,21")
ap.add_argument("--dataset", default=str(Path.home() / "go2-hackathon" / "data"))
ap.add_argument("--captures", default=str(Path.home() / "river-captures"))
ap.add_argument("--out-dir", default=str(Path.home() / "go2-hackathon" / "share"))
ap.add_argument("--raw", action="store_true", help="also write a zip with the raw takes")
ap.add_argument("--time-split", default="",
                help="e.g. 70,15,15: split every take in time into train/val/test (for few people); "
                     "val/test go to sessions 1SS/2SS (e.g. 120/220 for session 20), 1 clip gap at each boundary")
ap.add_argument("--val", default="", help="validation sessions (comma list); default: the third-last session")
ap.add_argument("--test", default="", help="test sessions (comma list); default: the last two sessions")
a = ap.parse_args()

gest = json.loads((HERE / "gestures.json").read_text())
classes = {g["id"]: g for g in gest["classes"]}
active = {cid for cid, g in classes.items() if g.get("active", True)}
inactive = {cid: g for cid, g in classes.items() if cid not in active}
skipped = Counter()
sessions = [f"{int(s):02d}" for s in a.sessions.split(",")]
out_dir = Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
nums = [int(s) for s in sessions]
contiguous = len(nums) > 2 and nums == list(range(nums[0], nums[-1] + 1))
tag = (f"s{sessions[0]}-{sessions[-1]}" if contiguous else "s" + "-".join(sessions)) + ("_timesplit" if a.time_split else "")
out = out_dir / f"go2_gestures_{tag}.zip"

rows, bad = [], []
for s in sessions:
    for p in sorted((Path(a.dataset) / s).glob("gesture_*.bin")):
        m = re.match(r"gesture_(\d+)-(.+)-([a-z]+)-(\d+)\.bin$", p.name)
        d = p.read_bytes()
        n = struct.unpack("!I", d[:4])[0] if len(d) >= 4 else -1
        if not m or n != len(d) - 4 or n % 13 or int(m.group(1)) not in classes:
            bad.append(p.name)
            continue
        cid = int(m.group(1))
        if cid not in active:
            skipped[classes[cid]["name"]] += 1
            continue
        rows.append({"path": f"motion_buddy_dataset/{s}/{p.name}", "session": s, "class_id": cid,
                     "label_index": cid - 1, "class_name": classes[cid]["name"], "style": m.group(3),
                     "events": n // 13, "_src": p})
for r in rows:
    r["split"] = "train"
if a.time_split:
    fr = [float(x) for x in a.time_split.split(",")]
    groups = {}
    for r in rows:
        n = int(re.search(r"-(\d+)\.bin$", r["path"]).group(1))
        groups.setdefault((r["session"], r["class_id"], r["style"]), []).append((n, r))
    dropped = []
    for g in groups.values():
        g.sort(key=lambda x: x[0])
        m = len(g)
        if m < 10:                           # too few clips to split with gaps -> all training
            continue
        b1 = round(m * fr[0] / sum(fr)); b2 = round(m * (fr[0] + fr[1]) / sum(fr))
        for i, (n, r) in enumerate(g):
            if i in (b1, b2) and 0 < i < m - 1:
                dropped.append(r)            # gap so neighbouring seconds don't leak across splits
            elif i >= b2:
                r["split"] = "test"
            elif i >= b1:
                r["split"] = "val"
    for r in dropped:
        rows.remove(r)
    for r in rows:
        if r["split"] != "train":
            sid = f"{(1 if r['split'] == 'val' else 2)}{r['session']}"
            r["path"] = r["path"].replace(f"/{r['session']}/", f"/{sid}/", 1)
    tag_split = True
else:
    tag_split = False
    # who recorded each session: ~/river-captures/sNN/person.txt (record_session.py --person); default itself
    person = {}
    for s in sessions:
        f = Path(a.captures) / f"s{s}" / "person.txt"
        p = f.read_text().strip() if f.exists() else s
        person[s] = f"{int(p):02d}" if p.isdigit() else p
    VAL = [f"{int(s):02d}" for s in a.val.split(",") if s] or sessions[-3:-2]
    TEST = [f"{int(s):02d}" for s in a.test.split(",") if s] or sessions[-2:]
    if len(sessions) >= 3 or a.val or a.test:   # person split, recorded in labels.csv (files stay in place)
        split_of = {s: ("val" if s in VAL else "test" if s in TEST else "train") for s in sessions}
        clash = {}
        for s, p in person.items():
            clash.setdefault(p, set()).add(split_of[s])
        clash = {p: v for p, v in clash.items() if len(v) > 1}
        if clash:
            raise SystemExit(f"person(s) in more than one split: {clash} - one person's sessions must stay "
                             f"together; choose --val/--test accordingly (person per session: {person})")
        for r in rows:
            r["split"] = split_of[r["session"]]
        for r in rows:
            r["person"] = person[r["session"]]
if bad:
    raise SystemExit(f"invalid/unknown files, fix first: {bad[:10]}")
if not rows:
    raise SystemExit("no clips found")

count = Counter((r["class_id"], r["style"]) for r in rows)
styles = sorted({r["style"] for r in rows})
per_session = Counter(r["session"] for r in rows)

table = ["| label | id | class | " + " | ".join(styles) + " | total |",
         "|---|---|---|" + "---|" * len(styles) + "---|"]
for cid, g in classes.items():
    if cid not in active:
        table.append(f"| {cid - 1} | {cid} | {g['name']} | " + " | ".join("-" for _ in styles) + " | not included |")
        continue
    vals = [count[(cid, st)] for st in styles]
    table.append(f"| {cid - 1} | {cid} | {g['name']} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} |")
gest_table = ["| label | class | how it is performed | robot action |", "|---|---|---|---|"]
for cid, g in classes.items():
    if cid in active:
        gest_table.append(f"| {cid - 1} | {g['name']} | {g['how']} | {g['go2']} |")
    else:
        gest_table.append(f"| {cid - 1} | *(unused: {g['name']}, dropped for now)* | - | map to nothing |")
split_note = ""
if a.time_split:
    sc = Counter(r["split"] for r in rows)
    split_note = f"""
## Train / validation / test split (time-based)

With only {len(sessions)} person(s) a split by person is not possible yet, so **every take is split in time**:
the first {a.time_split.split(',')[0]} % of its seconds are training, the next {a.time_split.split(',')[1]} % validation, the last
{a.time_split.split(',')[2]} % test, with one second left out at each boundary so neighbouring seconds don't leak.
Validation/test clips are stored as extra session folders so the session-based training script works as is:

| split | folders | clips |
|---|---|---|
| train | {', '.join(sessions)} | {sc['train']} |
| validation | {', '.join('1' + s for s in sessions)} | {sc['val']} |
| test | {', '.join('2' + s for s in sessions)} | {sc['test']} |

`train_snn_v6.py --zip {out.name} --val {','.join('1' + s for s in sessions)} --test {','.join('2' + s for s in sessions)} --flip 0`

This measures how well the model works **for the recorded person(s)**, not for new people.
"""
elif len(sessions) >= 3:
    val_s, test_s = VAL, TEST
    have = {r["class_id"] for r in rows}
    same = {p: [s for s in sessions if person[s] == p] for p in set(person.values())}
    same = {p: v for p, v in same.items() if len(v) > 1}
    person_note = ("Some people recorded more than one session (kept in the same split): "
                   + "; ".join(f"person {p} = sessions {', '.join(v)}" for p, v in sorted(same.items())) + ".\n"
                   if same else "")
    names = ",".join(g["name"].replace("-360", "").replace("-", " ").capitalize() for g in classes.values())
    unused = [str(cid - 1) for cid in classes if cid not in have]
    unused_arg = f" --unused {','.join(unused)}" if unused else ""
    unused_note = (f"`--unused {','.join(unused)}` keeps the output(s) without clips in this export silent, "
                   if unused else "")
    split_note = f"""
## Train / validation / test split (by person)

Split by **person**, so the scores show how the model does on people it has never seen.
Suggested split: validation = sessions {', '.join(val_s)}, test = sessions {', '.join(test_s)},
training = all others ({', '.join(s for s in sessions if s not in val_s and s not in test_s)}).
{person_note}
```
python train_snn_v8.py --zip {out.name} --arch nb40 --val {','.join(val_s)} --test {','.join(test_s)} \\
  --class-names "{names}"{unused_arg} --mirror 1:2 --qat-bits 8h
```

{unused_note}`--mirror 1:2` makes the left/right flip augmentation swap the two turn labels,
`--qat-bits 8h` exports weights for the 8-bit board firmware.
"""

readme = f"""# Go2 gesture dataset ({', '.join('session ' + s for s in sessions)})

Event-camera recordings of our own gestures for the Go2 demo, recorded with the Prophesee GenX320
(320x320) on the Kria. **Each file is one 1-second clip of one person repeating one gesture.**
One session = one person. Total: **{len(rows)} clips** ({', '.join(f'session {s}: {n}' for s, n in sorted(per_session.items()))}).

## Labels - no manual labelling needed

The label is in the file name, and `labels.csv` lists every clip:

```
motion_buddy_dataset/<session>/gesture_<id>-<class>-<style>-<n>.bin
e.g. motion_buddy_dataset/20/gesture_02-turn-left-360-fast-17.bin  ->  class 2, label index 1
```

Training label = `id - 1` (5 outputs), the same rule as the reference training code, so
`train_snn_v8.py --zip {out.name}` reads it directly.

{chr(10).join(gest_table)}
{"" if not inactive else chr(10) + "Output " + ", ".join(str(c - 1) for c in inactive) + " has no clips in this export: the model keeps 5 outputs (same board config) and that output is simply never the target; the robot bridge should map it to nothing." + chr(10)}

Left/right is the **performer's own** left/right (the camera sees it mirrored).

## Clips per class and style

{chr(10).join(table)}

Styles: `normal` / `fast` / `large` = speed and size of the movement; no-action also has `still`
(standing, small fidgets) and `empty` (nobody in view).

{split_note}
## File format

Same as the Motion Buddy dataset and `river-tools` recordings:

```
4 bytes   big-endian uint32 = number of bytes that follow (13 x events)
13 bytes per event, big-endian:  x uint16 | y uint16 | polarity uint8 (1 = ON) | timestamp int64 (us)
```

```python
import numpy as np
EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
ev = np.fromfile(path, dtype=EV, offset=4)
t = ev["timestamp"] - ev["timestamp"][0]      # us since clip start, 0 .. ~1e6
```

## How it was recorded

- Camera fixed, performer ~2.5-3 m away, same biases as live inference (`/root/metavision_camera.json`).
- An on-screen prompter showed the gesture; each gesture was repeated non-stop in three styles
  (normal, fast, large): 1 min per style in session 20, 20 s per style from session 21 on.
  No-action: normal and fast fidgeting, plus standing still and an empty scene (these two have
  very few events and are mostly removed by the filter). Labels come from the prompter, not from
  hand labelling.
- Takes were cut into back-to-back 1 s windows (first/last second dropped) and filtered:
  too few events, whole-frame smear (camera moved), person at the frame edge, or a motion pattern
  unlike the rest of the take.

## Important for training

1. **Split by session (person)** for validation/test, never by random clip - clips of one person
   are near-duplicates.
2. **Horizontal flip augmentation swaps meaning**: turn-left-360 (label 1) <-> turn-right-360
   (label 2). Swap these two labels whenever a clip is flipped (`--mirror 1:2` in train_snn_v8.py),
   or use `--flip 0`.
3. **Do not mix with the Motion Buddy zip as-is**: its classes 01-04 are sit/lay-down/hello/dance and
   would collide with our 01-04. Its clips can be added as no-action (rename to `gesture_05-...`).
4. Activity differs a lot between classes: heart ~10-20k events per clip, the turns ~40-100k,
   no-action ~15-70k.
5. The deployed board model takes 40x40 input (downsampling 8) and 5 outputs; output 4 = no action.
"""

csv_buf = io.StringIO()
w = csv.DictWriter(csv_buf, fieldnames=[k for k in rows[0] if not k.startswith("_")])
w.writeheader()
for r in rows:
    w.writerow({k: v for k, v in r.items() if not k.startswith("_")})

with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for r in rows:
        zf.write(r["_src"], r["path"])
    zf.writestr("labels.csv", csv_buf.getvalue())
    zf.writestr("gestures.json", (HERE / "gestures.json").read_text())
    zf.writestr("README.md", readme)
(out_dir / f"README_{tag}.md").write_text(readme)
print(f"{out}: {len(rows)} clips, {out.stat().st_size / 1e6:.0f} MB"
      + (f" | not included (inactive): {dict(skipped)}" if skipped else "")
      + (f" | split {dict(Counter(r['split'] for r in rows))}" if a.time_split else ""))

if a.raw:
    raw_out = out_dir / f"go2_gestures_{tag}_raw_takes.zip"
    with zipfile.ZipFile(raw_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for s in sessions:
            for p in sorted((Path(a.captures) / f"s{s}" / "raw").glob("*.bin")):
                zf.write(p, f"raw_takes/{s}/{p.name}")
            for p in sorted((Path(a.captures) / f"s{s}" / "previews").glob("*.png")):
                zf.write(p, f"previews/{s}/{p.name}")
    print(f"{raw_out}: {raw_out.stat().st_size / 1e6:.0f} MB")
