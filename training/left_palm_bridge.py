"""Run the quantized palm model on a recording or a stream of binned event counts."""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch

from event_data import decode_recording
from training.data import bin_events
from training.export_left_palm import load_model


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "left-palm-40-q26.json"


def recording_bins(raw, name, bin_us, grid):
    events, timestamps = decode_recording(raw, name)
    counts = bin_events(events, timestamps, bin_us, grid=grid)
    for index, frame in enumerate(counts):
        yield int(timestamps[0]) + index * bin_us, frame


def stdin_bins(stream, shape, max_count):
    for number, line in enumerate(stream, 1):
        try:
            row = json.loads(line)
            start = row["window_start_us"]
            frame = np.asarray(row["counts"])
            if type(start) is not int or start < 0 or frame.shape != shape or frame.dtype.kind not in "iu":
                raise ValueError("expected a nonnegative timestamp and integer counts in model input order")
            if np.any(frame < 0):
                raise ValueError("event counts must be nonnegative")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Input line {number}: {error}") from error
        yield start, np.minimum(frame, max_count).astype(np.uint8)


def forecasts(frames, model, weights, metadata):
    bin_us = metadata["model_args"]["ticks_per_step"]
    horizon = metadata["horizon_us"]
    history = deque(maxlen=metadata["input_shape"][0])
    expected_start = None
    bins_since_gap = 0
    for start, frame in frames:
        if start != expected_start:
            history.clear()
            bins_since_gap = 0
        expected_start = start + bin_us
        history.append(frame)
        bins_since_gap += 1
        if len(history) != history.maxlen or bins_since_gap % (horizon // bin_us):
            continue
        counts = np.stack(history)
        result = {"target_window_start_us": expected_start, "target_window_end_us": expected_start + horizon,
                  "left_palm_x": None, "left_palm_y": None}
        if counts.any():
            with torch.inference_mode():
                output = model(torch.from_numpy(counts)[None], weights=weights)[0]
                coordinates = output * metadata["coordinate_scale"] + metadata["coordinate_offset"]
            result["left_palm_x"], result["left_palm_y"] = coordinates.tolist()
        yield result


def publish(messages, topic):
    if topic is None:
        for message in messages:
            print(json.dumps(message, allow_nan=False), flush=True)
        return

    import rclpy
    from std_msgs.msg import String

    rclpy.init(args=[])
    node = rclpy.create_node("left_palm_forecast")
    try:
        publisher = node.create_publisher(String, topic, 10)
        for message in messages:
            if not rclpy.ok():
                break
            publisher.publish(String(data=json.dumps(message, allow_nan=False)))
            rclpy.spin_once(node, timeout_sec=0)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--recording", type=Path, help="Replay the first second of a .bin capture")
    source.add_argument("--member", help="Replay this recording inside --archive")
    parser.add_argument("--archive", default=Path("go2_gestures_s20.zip"), type=Path)
    parser.add_argument("--ros-topic", help="Publish forecast JSON to this ROS 2 std_msgs/String topic")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, weights, metadata = load_model(args.model)
    bin_us = metadata["model_args"]["ticks_per_step"]
    grid = metadata["model_args"]["grid"]
    if args.recording:
        frames = recording_bins(args.recording.read_bytes(), str(args.recording), bin_us, grid)
    elif args.member:
        with ZipFile(args.archive) as archive:
            frames = recording_bins(archive.read(args.member), args.member, bin_us, grid)
    else:
        frames = stdin_bins(sys.stdin, tuple(metadata["input_shape"][1:]), metadata["max_event_count"])
    try:
        publish(forecasts(frames, model, weights, metadata), args.ros_topic)
    except (ValueError, ModuleNotFoundError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
