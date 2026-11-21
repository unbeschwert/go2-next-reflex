"""Render a local event-camera pilot and create its CVAT labeling tasks."""

import argparse
from getpass import getpass
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import imageio_ffmpeg
import numpy as np


SIZE = 320
TICKS_PER_FRAME = 50_000
PREVIEW_FPS = 20
EVENT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">u8")])
PALETTE = np.array([(0, 0, 0), (255, 143, 84), (100, 213, 255), (255, 255, 255)], dtype=np.uint8)
CLIPS = (
    "motion_buddy_dataset/01/gesture_03-hello-normal-01.bin",
    "motion_buddy_dataset/01/gesture_04-dance-normal-01.bin",
    "motion_buddy_dataset/03/gesture_05-no-action-normal-01.bin",
)
SEED_PATH = Path(__file__).with_name("demo_keyframes.json")


def decode_recording(raw, name):
    payload = memoryview(raw)[4:]
    if not payload or len(payload) % EVENT.itemsize or int.from_bytes(raw[:4], "big") != len(payload):
        raise ValueError(f"Invalid event payload length in {name}")
    events = np.frombuffer(payload, dtype=EVENT)
    if events["x"].max() >= SIZE or events["y"].max() >= SIZE or events["p"].max() > 1:
        raise ValueError(f"Invalid coordinates or polarity in {name}")
    timestamps = events["t"].astype(np.uint64)
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError(f"Nonmonotonic timestamps in {name}")
    return events, timestamps


def render_clip(events, timestamps, output):
    offsets = timestamps - timestamps[0]
    frame_count = int(offsets[-1]) // TICKS_PER_FRAME + 1
    boundaries = np.searchsorted(offsets, np.arange(frame_count + 1, dtype=np.uint64) * TICKS_PER_FRAME)
    temporary = output.with_suffix(".tmp.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(temporary), (SIZE, SIZE), fps=PREVIEW_FPS, quality=8,
        codec="libx264", macro_block_size=1, ffmpeg_log_level="error",
    )
    try:
        writer.send(None)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            frame_events = events[start:end]
            planes = np.zeros((2, SIZE, SIZE), dtype=np.uint8)
            planes[frame_events["p"], frame_events["y"], frame_events["x"]] = 1
            writer.send(PALETTE[planes[0] + 2 * planes[1]].tobytes())
    finally:
        writer.close()
    temporary.replace(output)
    return frame_count


def prepare(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with ZipFile(archive) as recordings:
        for name in CLIPS:
            member = PurePosixPath(name)
            video = f"{member.parent.name}-{member.stem}.mp4"
            events, timestamps = decode_recording(recordings.read(name), name)
            frame_count = render_clip(events, timestamps, output / video)
            manifest.append({
                "video": video,
                "recording": name,
                "first_timestamp": int(timestamps[0]),
                "ticks_per_frame": TICKS_PER_FRAME,
                "preview_fps": PREVIEW_FPS,
                "frames": frame_count,
            })
            print(f"{video}: {frame_count} frames from {len(events):,} events")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def add_track(task, keyframes):
    from cvat_sdk import models

    label = next(label for label in task.get_labels() if label.name == "palm")
    shapes = [
        models.TrackedShapeRequest(
            type="points", frame=frame, points=[x, y], outside=outside,
            occluded=False, z_order=0, attributes=[],
        )
        for frame, x, y, outside in keyframes
    ]
    track = models.LabeledTrackRequest(
        frame=shapes[0].frame, label_id=label.id, shapes=shapes, attributes=[], source="manual",
    )
    task.set_annotations(models.LabeledDataRequest(tracks=[track], shapes=[], tags=[]))


def upload(output, server, user):
    from cvat_sdk import make_client, models

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    keyframes = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    videos = [output / clip["video"] for clip in manifest]
    for video in videos:
        if not video.is_file():
            raise FileNotFoundError(video)

    password = os.environ.get("CVAT_PASSWORD") or getpass(f"CVAT password for {user}: ")
    with make_client(host=server, credentials=(user, password)) as client:
        existing = {task.name: task for task in client.tasks.list()}
        for video in videos:
            name = f"Go2 palm demo · {video.stem}"
            if name in existing:
                task = existing[name]
                print(f"Existing task: {server.rstrip('/')}/tasks/{task.id}")
                continue
            task = client.tasks.create_from_data(
                spec=models.TaskWriteRequest(name=name, labels=[models.PatchedLabelRequest(name="palm", type="points")]),
                resources=[video],
                data_params={"image_quality": 95},
                status_check_period=1,
            )
            if video.name in keyframes:
                add_track(task, keyframes[video.name])
            print(f"Created task: {server.rstrip('/')}/tasks/{task.id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("prepare", help="Render the three pilot clips from the local ZIP")
    render.add_argument("--archive", type=Path, default=Path("durance-ai-handgesture-recognition.zip"))
    push = commands.add_parser("upload", help="Create video labeling tasks on CVAT")
    push.add_argument("--server", default="http://localhost:8080")
    push.add_argument("--user", required=True)
    for command in (render, push):
        command.add_argument("--output", type=Path, default=Path(".local/labeling"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.archive, args.output)
    else:
        upload(args.output, args.server, args.user)


if __name__ == "__main__":
    main()
