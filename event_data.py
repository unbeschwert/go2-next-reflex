"""Reader for the event records in the local Durance gesture archive."""

import numpy as np


SIZE = 320
EVENT = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">u8")])


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
