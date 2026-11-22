#!/usr/bin/env python3
"""Cut long river-tools takes into 1 s training clips (Motion Buddy layout).

Dataset clips are 1 s slices of a gesture done over and over (activity is flat
across the second), so record a take as: Start -> repeat ONE gesture without
stopping for ~20 s -> Stop. This script then

  * skips the first/last --trim s (walking in/out, reaching for the mouse),
  * tiles the rest into back-to-back --clip s windows,
  * drops windows with too few events (pauses, person out of view),
  * writes  <dataset>/<session>/gesture_<NN>-<label>-<style>-<k>.bin
    (same 13-byte big-endian format as the dataset, original timestamps),
  * moves the raw take to <session dir>/raw/ so it can be re-cut later,
  * saves a preview PNG (event rate + kept windows + thumbnails).

    # cued session (record_session.py) - labels come from the file names:
    python3 cut_takes.py --session 20 --labeled
    # river-tools takes - label every new take in the session folder:
    python3 cut_takes.py --session 20 --cls 1 --label sit --style normal
    python3 cut_takes.py --session 20 --summary          # clips per class/style
    python3 cut_takes.py ... --dry-run                   # show only

Classes: see gestures.json (1-5 dataset classes, 6+ our own).
"""
import argparse
import re
import shutil
import struct
from collections import Counter
from pathlib import Path

import numpy as np

EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
import json
CLASSES = {g["id"]: g["name"] for g in
           json.loads((Path(__file__).resolve().parent / "gestures.json").read_text())["classes"]}
STYLES = ("normal", "fast", "large")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session", required=True, help="session number, one per person/place, e.g. 20")
ap.add_argument("--captures", default=str(Path.home() / "river-captures"),
                help="takes are read from <captures>/s<session>/samples/")
ap.add_argument("--dataset", default=str(Path.home() / "go2-hackathon" / "data"))
ap.add_argument("--cls", type=int, choices=sorted(CLASSES))
ap.add_argument("--label", help="defaults to the dataset name of --cls")
ap.add_argument("--style", default="normal")
ap.add_argument("--clip", type=float, default=1.0, help="clip length, s")
ap.add_argument("--trim", type=float, default=1.0, help="seconds dropped at start and end of a take")
ap.add_argument("--min-frac", type=float, default=0.3,
                help="drop a window with fewer events than this x the take's median window")
ap.add_argument("--min-events", type=int, default=2000,
                help="absolute minimum events per clip (dataset min 1.4k; heart/namaskaram from 2.5-3 m give 4-20k)")
ap.add_argument("--max-cover", type=float, default=0.75,
                help="drop if more than this fraction of the frame is active = camera moved / person too close "
                     "(dataset 99th pct 0.73)")
ap.add_argument("--max-border", type=float, default=0.6,
                help="drop if more than this fraction of events is in the outer 40 px band = person at the edge "
                     "(dataset 99th pct 0.55)")
ap.add_argument("--outlier", type=float, default=4.0,
                help="drop clips whose motion pattern is this many MADs from the take's typical clip (0 = off)")
ap.add_argument("--best", type=int, default=0, help="keep at most N most typical clips per take (0 = all that pass)")
ap.add_argument("--recut", action="store_true",
                help="delete this session's clips and cut again from <session dir>/raw/ (e.g. with new thresholds)")
ap.add_argument("--max-clips", type=int, default=0, help="max clips per take (0 = all)")
ap.add_argument("--no-preview", action="store_true")
ap.add_argument("--labeled", action="store_true",
                help="cut the cue-labelled takes from record_session.py (<session dir>/labeled/)")
ap.add_argument("--summary", action="store_true", help="only print clip counts of the session")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

session = f"{int(a.session):02d}"
sess_dir = Path(a.captures) / f"s{session}"
dst_dir = Path(a.dataset) / session


def summary():
    c = Counter()
    for p in dst_dir.glob("gesture_*.bin"):
        m = re.match(r"gesture_(\d+)-(.+)-([a-z]+)-\d+\.bin$", p.name)
        if m:
            c[(int(m.group(1)), m.group(3))] += 1
    styles = sorted({s for _, s in c} | set(STYLES))
    print(f"session {session} ({dst_dir}):")
    print("  class".ljust(20) + "".join(s.rjust(8) for s in styles) + "   total")
    for k, name in CLASSES.items():
        row = [c[(k, s)] for s in styles]
        if not any(row) and k > 5:
            continue
        print(f"  {k:2d} {name}".ljust(20) + "".join(str(v).rjust(8) for v in row) + str(sum(row)).rjust(8))
    print(f"  total {sum(c.values())} clips")


if a.summary:
    summary()
    raise SystemExit

REASON_COLOR = {"ok": "tab:green", "low": "tab:red", "smear": "tab:orange", "edge": "tab:purple",
                "odd": "tab:gray", "extra": "tab:cyan"}


def window_quality(ev, t, starts):
    """Per 1 s window: events, active-frame fraction, edge fraction, coarse motion pattern."""
    n_ev, cover, border, hist = [], [], [], []
    for s in starts:
        lo, hi = np.searchsorted(t, [s, s + clip_us])
        x = ev["x"][lo:hi].astype(np.int64); y = ev["y"][lo:hi].astype(np.int64)
        n = hi - lo
        g = np.bincount((y // 10) * 32 + x // 10, minlength=1024)
        n_ev.append(n)
        cover.append(float((g >= max(3, n / 1024 * 0.2)).mean()) if n else 0.0)
        border.append(float(((x < 40) | (x >= 280) | (y < 40) | (y >= 280)).mean()) if n else 1.0)
        h = np.bincount((y // 20) * 16 + x // 20, minlength=256).astype(float)
        hist.append(np.sqrt(h / max(1, n)))
    return np.array(n_ev), np.array(cover), np.array(border), np.array(hist)


def choose(n_ev, cover, border, hist):
    """Reason per window: ok / low / smear / edge / odd / extra."""
    med = float(np.median(n_ev))
    why = np.array(["ok"] * len(n_ev), dtype=object)
    why[(n_ev < a.min_events) | (n_ev < a.min_frac * med)] = "low"
    why[(why == "ok") & (cover > a.max_cover)] = "smear"
    why[(why == "ok") & (border > a.max_border)] = "edge"
    ok = np.where(why == "ok")[0]
    dist = np.zeros(len(n_ev))
    if len(ok) >= 8:
        typical = np.median(hist[ok], axis=0)
        dist = np.linalg.norm(hist - typical, axis=1)
        d = dist[ok]
        mad = np.median(np.abs(d - np.median(d))) + 1e-9
        if a.outlier > 0:
            why[ok[d > np.median(d) + a.outlier * mad]] = "odd"
    if a.best:
        ok = np.where(why == "ok")[0]
        if len(ok) > a.best:
            why[ok[np.argsort(dist[ok])[a.best:]]] = "extra"
    return why


def preview(path, t, starts, kept, ev, png, why=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    why = why if why is not None else np.where(kept, "ok", "low")
    ok_starts = [s for s, ok in zip(starts, kept) if ok]
    thumbs = [ok_starts[int(i)] for i in np.linspace(0, len(ok_starts) - 1, min(8, len(ok_starts)))] if ok_starts else []
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_axes([0.05, 0.58, 0.93, 0.36])
    rate = np.bincount(t // 50_000) / 0.05 / 1e3
    ax.plot(np.arange(len(rate)) * 0.05, rate, lw=0.8, color="k")
    for s, w in zip(starts, why):
        ax.axvspan(s / 1e6, (s + clip_us) / 1e6, color=REASON_COLOR[w], alpha=0.2 if w == "ok" else 0.35, lw=0)
    ax.set_xlabel("s"); ax.set_ylabel("k events/s")
    cnt = Counter(why)
    ax.set_title(f"{path.name}: kept {cnt.get('ok', 0)} (green) | dropped: " +
                 ", ".join(f"{k} {v}" for k, v in cnt.items() if k != "ok") +
                 "   [red low, orange smear, purple edge, grey odd, cyan over --best]", fontsize=8)
    for i, s in enumerate(thumbs):
        sel = (t >= s) & (t < s + clip_us)
        img = np.zeros((320, 320, 3))
        np.add.at(img, (ev["y"][sel], ev["x"][sel], np.where(ev["polarity"][sel] == 1, 0, 2)), 1)
        img = np.clip(img / max(1, np.percentile(img[img > 0], 99)), 0, 1)
        img[..., 1] = 0
        tax = fig.add_axes([0.05 + i * 0.117, 0.03, 0.11, 0.42])
        tax.imshow(img, interpolation="nearest"); tax.set_xticks([]); tax.set_yticks([])
        tax.set_title(f"{s / 1e6:.1f}s", fontsize=7)
    fig.savefig(png, dpi=80); plt.close(fig)


def cut(p, cls, label, style, raw_name=None, move=True):
    """Tile one take into clips; returns the number written."""
    prefix = f"gesture_{cls:02d}-{label}-{style}-"
    raw_dir = sess_dir / "raw"
    if raw_name is None:   # river-tools sample: number it after existing raw takes
        n_raw = len(list(raw_dir.glob(prefix + "take*.bin"))) if raw_dir.exists() else 0
        raw_name = f"{prefix}take{n_raw + 1:02d}.bin"
    data = p.read_bytes()
    n = struct.unpack("!I", data[:4])[0] if len(data) >= 4 else -1
    if n != len(data) - 4 or n % 13 or n == 0:
        print(f"skip {p.name}: incomplete (still recording? header {n}, file {len(data) - 4} bytes)")
        return 0
    ev = np.frombuffer(data, EV, offset=4)
    t = ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])
    dur = int(t[-1])
    starts = list(range(trim_us, dur - trim_us - clip_us + 1, clip_us))
    if a.max_clips:
        starts = starts[:a.max_clips]
    if not starts:
        print(f"skip {p.name}: only {dur / 1e6:.1f} s long (need {a.clip + 2 * a.trim:.1f} s)")
        return 0
    bounds = np.searchsorted(t, starts + [starts[-1] + clip_us])
    n_ev, cover, border, hist = window_quality(ev, t, starts)
    why = choose(n_ev, cover, border, hist)
    kept = list(why == "ok")
    med = float(np.median(n_ev))
    existing = [int(m.group(1)) for q in dst_dir.glob(prefix + "*.bin") if (m := re.search(r"-(\d+)\.bin$", q.name))]
    k = k0 = max(existing, default=0)
    drops = Counter(w for w in why if w != "ok")
    print(f"{p.name}: {dur / 1e6:.1f} s, {len(ev)} events, median {med / 1e3:.0f}k/clip "
          f"-> {sum(kept)} clips" + (f", dropped {dict(drops)}" if drops else ""))
    for i, (s, ok) in enumerate(zip(starts, kept)):
        if not ok:
            continue
        lo, hi = bounds[i], np.searchsorted(t, s + clip_us)
        k += 1
        if not a.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            chunk = data[4 + 13 * lo: 4 + 13 * hi]
            (dst_dir / f"{prefix}{k:02d}.bin").write_bytes(struct.pack("!I", len(chunk)) + chunk)
    if k > k0:
        print(f"  -> {dst_dir}/{prefix}{k0 + 1:02d}..{k:02d}.bin")
    if not a.dry_run:
        if move:
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), raw_dir / raw_name)
        if not a.no_preview:
            (sess_dir / "previews").mkdir(exist_ok=True)
            png = sess_dir / "previews" / (Path(raw_name).stem + ".png")
            preview(p, t, starts, kept, ev, png, why)
            print(f"  preview {png}")
    return k - k0


clip_us, trim_us = int(a.clip * 1e6), int(a.trim * 1e6)
total = 0
NAME_RE = r"gesture_(\d+)-(.+)-([a-z]+)-take\d+\.bin$"
if a.recut:
    takes = sorted((sess_dir / "raw").glob("gesture_*.bin"))
    if not takes:
        raise SystemExit(f"no raw takes in {sess_dir / 'raw'}")
    old = list(dst_dir.glob("gesture_*.bin"))
    print(f"recut: removing {len(old)} clips in {dst_dir}" + (" (dry run: not really)" if a.dry_run else ""))
    if not a.dry_run:
        for q in old:
            q.unlink()
    for p in takes:
        m = re.match(NAME_RE, p.name)
        if m:
            total += cut(p, int(m.group(1)), m.group(2), m.group(3), raw_name=p.name, move=False)
elif a.labeled:
    takes = sorted((sess_dir / "labeled").glob("gesture_*.bin"))
    if not takes:
        raise SystemExit(f"no labeled takes in {sess_dir / 'labeled'} (record_session.py writes them)")
    for p in takes:
        m = re.match(NAME_RE, p.name)
        if not m:
            print(f"skip {p.name}: unexpected name")
            continue
        total += cut(p, int(m.group(1)), m.group(2), m.group(3), raw_name=p.name)
else:
    if a.cls is None:
        raise SystemExit("--cls is required (or use --labeled / --summary)")
    takes = sorted((sess_dir / "samples").glob("sample_*.bin"))
    if not takes:
        raise SystemExit(f"no new takes in {sess_dir / 'samples'} - record with:\n"
                         f"  cd {sess_dir} && river-tools capture --host root@192.168.77.1")
    for p in takes:
        total += cut(p, a.cls, a.label or CLASSES[a.cls], a.style)
print(("dry run - would write " if a.dry_run else "wrote ") + f"{total} clips")
if not a.dry_run:
    summary()
