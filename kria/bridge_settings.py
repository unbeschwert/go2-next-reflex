"""Gesture-control settings of live_gesture_bridge.py, shared with the dashboard (one source of truth).

The bridge reads SETTINGS_FILE (JSON {key: value}) about once a second. Keys that are missing, unknown
or out of range fall back to the default (and are reported), so a broken file can never make the robot
more trigger-happy than the tuned defaults allow. Runs on the Kria (Python 3.8+, stdlib only).
"""
import json
import math
import os

SETTINGS_FILE = "/root/bridge_settings.json"
STATUS_FILE = "/tmp/bridge_status.json"

LABELS = {
    "own": ["Heart", "Turn left 360", "Turn right 360", "Follow me", "none"],
    "dataset": ["Sit", "Lay down", "Hello", "Dance", "none"],
}

# Tuned defaults (see TEST_RULES / ROBOT_RULES in live_gesture_bridge.py for how they were measured).
# robot = the real Go2 (--require-clear): same rule, the gesture is held a little longer.
_BASE = {
    # follow_hold_s: "follow me" starts the robot walking, so it is held longer (4-gesture model
    # 2d8fac76 replay on unseen people: false Follow on no-action 3 -> 0, Follow still 96.5 %, 0.77 s)
    "own": {"min_rate": [0.50, 0.50, 0.50, 0.50], "margin": 0.30, "stable_s": 0.20, "quiet": 0.35,
            "beat_none": 0.20, "min_frames": 40, "follow_hold_s": 0.50},
    "dataset": {"min_rate": [0.60, 0.50, 0.60, 0.30], "margin": 0.15, "stable_s": 0.30, "quiet": 0.45,
                "beat_none": 0.0, "min_frames": 25},
}
_ROBOT = {"own": {"stable_s": 0.30}, "dataset": {}}
COMMON = {"window_s": 0.5, "cooldown_s": 0.5, "rearm_max_s": 1.5, "turn_rate": 1.5, "paused": False}


def schema(classes="own"):
    names = LABELS.get(classes, LABELS["own"])
    rows = [
        dict(key="window_s", label="Observation window", group="Detection", type="float", min=0.2, max=1.5,
             step=0.05, unit="s", help="How long the bridge watches the network outputs before deciding (a "
             "sliding window). Longer = steadier but reacts later."),
    ]
    for i in range(4):
        rows.append(dict(key=f"min_rate_o{i}", label=f"{names[i]}: trigger level", group="Detection",
                         type="float", min=0.2, max=0.95, step=0.01, unit="",
                         help=f"Share of frames in the window in which the '{names[i]}' output must fire. "
                              "Higher = fewer false triggers, slower and may miss weak gestures."))
    rows += [
        dict(key="margin", label="Lead over the next gesture", group="Detection", type="float", min=0.0,
             max=0.8, step=0.01, unit="", help="The winning gesture must beat the second-best gesture by "
             "this much."),
        dict(key="beat_none", label="Lead over 'none'", group="Detection", type="float", min=0.0, max=0.8,
             step=0.01, unit="", help="The winner must also beat the 'no gesture' output by this much "
             "(0 = ignore the 'none' output)."),
        dict(key="stable_s", label="Hold time", group="Detection", type="float", min=0.0, max=1.5, step=0.05,
             unit="s", help="How long the same gesture must keep winning before the robot moves."),
    ]
    if classes == "own":
        rows.append(dict(key="follow_hold_s", label="Follow me: hold time", group="Detection", type="float",
                         min=0.0, max=2.0, step=0.05, unit="s",
                         help="'Follow me' starts the robot walking, so it must keep winning this long "
                              "(the longer of this and the hold time counts)."))
    rows += [
        dict(key="min_frames", label="Minimum frames per window", group="Detection", type="int", min=10,
             max=150, step=1, unit="frames", help="Frames that never arrived count as silence. Protects "
             "against the network firing by itself on near-empty frames in a quiet scene."),
        dict(key="quiet", label="Re-arm level", group="After a trick", type="float", min=0.05, max=0.8,
             step=0.01, unit="", help="The gesture that just ran is accepted again once its output drops "
             "below this (other gestures are accepted at once)."),
        dict(key="rearm_max_s", label="Maximum re-arm wait", group="After a trick", type="float", min=0.3,
             max=10.0, step=0.1, unit="s", help="...or at the latest this long after the robot became free."),
        dict(key="cooldown_s", label="Cooldown", group="After a trick", type="float", min=0.0, max=5.0,
             step=0.1, unit="s", help="Camera input ignored this long after a trick ends."),
        dict(key="turn_rate", label="Turn speed", group="Robot", type="float", min=0.3, max=2.0, step=0.05,
             unit="rad/s", help="Maximum yaw rate of the 360 degree turns (1.5 = a full circle in ~4.6 s)."),
        dict(key="paused", label="Pause gesture control", group="Safety", type="bool",
             help="The bridge keeps watching but sends no robot commands (manual mode and E-STOP set this)."),
    ]
    return rows


def defaults(classes="own", robot=False):
    base = dict(_BASE.get(classes, _BASE["own"]))
    if robot:
        base.update(_ROBOT.get(classes, {}))
    out = dict(COMMON)
    for i, v in enumerate(base.pop("min_rate")):
        out[f"min_rate_o{i}"] = v
    out.update(base)
    return out


def validate(values, classes="own"):
    """Return (clean, errors): clean has only known keys with values inside their range."""
    rows = {r["key"]: r for r in schema(classes)}
    clean, errors = {}, {}
    for key, value in (values or {}).items():
        r = rows.get(key)
        if r is None:
            errors[key] = "unknown setting"
            continue
        if r["type"] == "bool":
            if not isinstance(value, bool):
                errors[key] = "expected true or false"
                continue
            clean[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            errors[key] = "expected a number"
            continue
        if not r["min"] <= value <= r["max"]:
            errors[key] = "must be between %s and %s" % (r["min"], r["max"])
            continue
        clean[key] = int(round(value)) if r["type"] == "int" else float(value)
    return clean, errors


def load(path=SETTINGS_FILE, classes="own", robot=False, base=None):
    """Defaults (then `base`, e.g. command-line values) overlaid with the valid values of the settings
    file. Returns (values, errors)."""
    values = defaults(classes, robot)
    values.update(base or {})
    if not path or not os.path.exists(path):
        return values, {}
    try:
        with open(path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        return values, {"_file": "unreadable: %s" % e}
    clean, errors = validate({k: v for k, v in raw.items() if not k.startswith("_")}, classes)
    values.update(clean)
    return values, errors


def save(path, values):
    """Atomic write (the bridge may read at any moment)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(values, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
