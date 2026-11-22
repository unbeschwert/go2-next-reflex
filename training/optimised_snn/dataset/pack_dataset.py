#!/usr/bin/env python3
"""Append compatible sessions to a copy of the Motion Buddy archive."""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

if __package__:
    from .layout import CLIP_MEMBER, CLIP_NAME, recording_size, session_ids
else:
    from layout import CLIP_MEMBER, CLIP_NAME, recording_size, session_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path,
                    default=Path.home() / "Downloads" / "durance-ai-handgesture-recognition (1).zip")
    ap.add_argument("--dataset", type=Path, default=Path.home() / "go2-hackathon" / "data")
    ap.add_argument("--sessions", required=True, help="comma list, e.g. 20,21")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--as-no-action", action="store_true",
                    help="explicitly label appended clips as Motion Buddy class 05 (no action)")
    args = ap.parse_args()
    try:
        sessions = session_ids(args.sessions)
    except ValueError as exc:
        ap.error(str(exc))
    if args.base.resolve() == args.out.resolve():
        ap.error("--out must be a different file from --base")

    with zipfile.ZipFile(args.base) as archive:
        have, classes = set(), {}
        for name in archive.namelist():
            member = CLIP_MEMBER.fullmatch(name)
            if member is None:
                continue
            clip = CLIP_NAME.fullmatch(member[2])
            if clip is None:
                ap.error(f"invalid recording name in base archive: {name}")
            have.add(member[1])
            cid, label = int(clip[1]), clip[2]
            if classes.setdefault(cid, label) != label:
                ap.error(f"base archive already uses conflicting names for class {cid:02d}")

    clash = sorted(set(sessions) & have)
    if clash:
        ap.error(f"sessions {clash} already exist in the base zip; use new numbers")
    if args.as_no_action and classes.get(5) != "no-action":
        ap.error("--as-no-action requires class 05 to be named no-action in the base archive")

    additions = {}
    for session in sessions:
        files = sorted((args.dataset / session).glob("gesture_*.bin"))
        if not files:
            ap.error(f"no clips found for session {session} in {args.dataset}")
        entries = []
        for index, path in enumerate(files, 1):
            clip = CLIP_NAME.fullmatch(path.name)
            if clip is None or not recording_size(path):
                ap.error(f"invalid recording: {path}")
            cid, label, style = int(clip[1]), clip[2], clip[3]
            if args.as_no_action:
                name = f"gesture_05-no-action-{style}-{index:02d}.bin"
            elif classes.get(cid) != label:
                ap.error(f"{path.name} conflicts with base class {cid:02d} ({classes.get(cid, 'absent')}); "
                         "use --as-no-action only if these clips are intended as negatives")
            else:
                name = path.name
            entries.append((path, f"motion_buddy_dataset/{session}/{name}"))
        additions[session] = entries

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent, prefix=".pack-dataset-") as work:
        packed = Path(work) / "dataset.zip"
        shutil.copyfile(args.base, packed)
        with zipfile.ZipFile(packed, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for session, entries in additions.items():
                for source, name in entries:
                    archive.write(source, name)
                print(f"  session {session}: {len(entries)} clips added")
        packed.replace(args.out)
    print(f"done: {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
