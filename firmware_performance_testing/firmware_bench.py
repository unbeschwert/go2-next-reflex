#!/usr/bin/env python3
r"""Compare firmware builds of the same network on the Kria: FPGA latency, board power, identical outputs.

Every build is a board setup in /root/setups/NAME (firmware + server config + camera config + weights, the
folder /root/switch_setup.sh loads). --firmware NAME=DIR creates NAME as a copy of --base-setup with the
spleat.bit.bin/.dtbo from the laptop folder DIR, so only the firmware differs between the setups.

Per setup and round: load it (switch_setup.sh stops the camera/bridge first), read the FPGA fabric clock,
temperatures and fan, sample the board's input power (INA260) while idle, then replay the same clips with
fpga_eval.py while sampling power. Rounds alternate the order (A B, B A, A B, ...) to reduce the effect of drift from warm-up. At the end: one table per setup, and whether each build gives exactly the same output spikes as the
first one. After measurements start, the script attempts to restore the requested setup and camera state even on errors.

From the repo root (see README.md in this folder):

    python3 firmware_performance_testing/firmware_bench.py \
        --firmware nb40_better_latency=~/firmware/nb40_better_latency \
        --setups nb40_own4,nb40_better_latency --rounds 3 --out ~/firmware/bench_strategies \
        --zip ~/data/go2_gestures_s20-33.zip --sessions 22,29,30,33 --limit 25 --size 40 \
        --class-names 'Heart,Turn left,Turn right,Follow me,No action'

Needs on the board: /root/switch_setup.sh (copy of switch_setup.sh in this folder), /root/inference.sh,
the --base-setup folder in /root/setups/ and python3. On the laptop: numpy and fpga_eval.py.
"""
import argparse
import base64
import hashlib
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FPGA_EVAL = next((p for p in (HERE / "fpga_eval.py", HERE.parent / "training" / "optimised_snn" / "fpga_eval.py")
                  if p.exists()), HERE / "fpga_eval.py")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
SETUP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
FIRMWARE_FILES = ("spleat.bit.bin", "spleat.dtbo")
NETWORK_FILES = ("config.yaml", "metavision_camera.json", "weights.txt", "model_classes")
BOARD_FILES = ("/usr/lib/firmware/xilinx/spleat/spleat.bit.bin",
               "/usr/lib/firmware/xilinx/spleat/spleat.dtbo",
               "/root/.config/spleat-server/config.yaml", "/root/metavision_camera.json",
               "/root/weights.txt", "/root/model_classes")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--board", default="root@192.168.77.1")
    ap.add_argument("--setups", required=True, help="comma list of /root/setups names; the first is the reference")
    ap.add_argument("--firmware", action="append", default=[], metavar="NAME=DIR",
                    help="create/refresh setup NAME from --base-setup + the firmware in laptop folder DIR")
    ap.add_argument("--base-setup", default="nb40_own4")
    ap.add_argument("--restore-setup", default="nb40_own4")
    ap.add_argument("--restore-mode", default="start", choices=("start", "start --headless", "none"), help="camera state after restoring the setup")
    ap.add_argument("--out", required=True, help="folder for per-run .npz/.log and results.json")
    ap.add_argument("--rounds", type=int, default=3, help="measure every setup this many times, order alternating")
    ap.add_argument("--idle-s", type=float, default=15.0, help="seconds of idle power sampling per measurement")
    ap.add_argument("--dry-run", action="store_true", help="only check files and board, change nothing")
    ap.add_argument("--fpga-eval", default=str(FPGA_EVAL), help="replay tool (default: %(default)s)")
    args, replay = ap.parse_known_args(argv)
    setups = [name.strip() for name in args.setups.split(",")]
    if len(setups) < 2 or len(set(setups)) != len(setups):
        ap.error("--setups needs at least two distinct setup names")
    if args.rounds < 1 or not 0 < args.idle_s < float("inf"):
        ap.error("rounds and idle time must be positive")
    if not args.board or args.board.startswith("-") or any(c.isspace() for c in args.board):
        ap.error("--board must be an SSH host or user@host")
    if any(option.split("=", 1)[0] in {"--host", "--save", "--selftest", "--check-data"}
           for option in replay):
        ap.error("the benchmark controls replay host and output and requires a real recording archive")
    names = [*setups, args.base_setup, args.restore_setup]
    staged = {}
    for spec in args.firmware:
        name, sep, directory = spec.partition("=")
        if not sep or not directory or name not in setups or name in staged:
            ap.error("use --firmware NAME=DIR once for each build to stage in --setups")
        if name in (args.base_setup, args.restore_setup):
            ap.error("use a separate setup name for staged firmware so the base and restore setups remain available")
        staged[name] = Path(directory).expanduser().resolve()
    if any(SETUP_NAME.fullmatch(name) is None for name in names):
        ap.error("setup names must start with a letter or number and contain only letters, numbers, - or _")
    return args, replay, setups, staged


# Board-side power sampler: prints "<seconds> <microwatts>" every 50 ms until killed.
SAMPLER = r"""
import glob, sys, time
for d in glob.glob('/sys/class/hwmon/hwmon*'):
    if open(d + '/name').read().startswith('ina260'):
        break
else:
    sys.exit('no ina260 power sensor')
t0 = time.monotonic()
while True:
    print(f'{time.monotonic() - t0:.3f} {open(d + "/power1_input").read().strip()}', flush=True)
    time.sleep(0.05)
"""
SENSORS = ("for h in /sys/class/hwmon/hwmon*; do n=$(cat $h/name); for f in $h/temp*_input $h/pwm1; do "
           "[ -f $f ] && echo \"$n/$(basename $f) $(cat $f)\"; done; done")


def ssh(cmd, check=True, timeout=120):
    r = subprocess.run(SSH + [args.board, cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"board command failed ({r.returncode}): {cmd}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


class PowerSampler:
    def __init__(self):
        self.samples, self.proc = [], None

    def __enter__(self):
        code = base64.b64encode(SAMPLER.encode()).decode()   # no quoting problems through ssh
        self.proc = subprocess.Popen(SSH + [args.board, f"python3 -u -c 'import base64; exec(base64.b64decode(\"{code}\"))'"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            parts = line.split()
            if len(parts) == 2:
                try:
                    self.samples.append((float(parts[0]), int(parts[1]) / 1e6))
                except ValueError:
                    continue

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.reader.join(timeout=5)
        self.proc.stdout.close()

    def watts(self, skip_s=1.0):
        if not self.samples:
            return None
        start = self.samples[0][0] + skip_s
        w = np.array([p for t, p in self.samples if t >= start] or [p for _, p in self.samples])
        return {"mean": round(float(w.mean()), 4), "median": round(float(np.median(w)), 4),
                "std": round(float(w.std()), 4), "max": round(float(w.max()), 4), "n": int(w.size)}


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def fabric_clock_mhz():
    line = ssh("grep ' pl0_ref ' /sys/kernel/debug/clk/clk_summary", check=False)
    nums = [int(x) for x in line.split() if x.isdigit()]
    return round(nums[3] / 1e6, 3) if len(nums) >= 4 else None


def sensors():
    """{'ams/temp1_input': deg C, ..., 'pwmfan/pwm1': raw}"""
    out = {}
    for line in ssh(SENSORS, check=False).splitlines():
        key, _, value = line.partition(" ")
        if value.strip().lstrip("-").isdigit():
            out[key] = int(value) / 1000 if "temp" in key else int(value)
    return out


def checksums(paths):
    reply = ssh("md5sum " + shlex.join(paths))
    digests = {}
    for line in reply.splitlines():
        digest, _, filename = line.partition(" ")
        if re.fullmatch(r"[0-9a-f]{32}", digest):
            digests[filename.lstrip()] = digest
    if set(digests) != set(paths):
        raise RuntimeError("board did not return checksums for every requested file")
    return digests


def setup_checksums(name):
    paths = [f"/root/setups/{name}/{file}" for file in (*FIRMWARE_FILES, *NETWORK_FILES)]
    ssh(" && ".join(f"test -s {shlex.quote(path)}" for path in paths))
    return {Path(path).name: digest for path, digest in checksums(paths).items()}


def preflight(staged, replay_args):
    for name, directory in staged.items():
        for file in FIRMWARE_FILES:
            path = directory / file
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"{path} is missing or empty")
        print(f"setup {name}: {args.base_setup} + firmware {directory}")

    subprocess.run([sys.executable, args.fpga_eval, "--check-data", *replay_args], check=True, timeout=180)
    ssh("test -x /root/switch_setup.sh && test -x /root/inference.sh")
    fingerprints = {
        name: setup_checksums(name)
        for name in dict.fromkeys([args.restore_setup, *([args.base_setup] if staged else []), *setups])
        if name not in staged
    }
    for name, directory in staged.items():
        fingerprints[name] = fingerprints[args.base_setup] | {file: md5(directory / file) for file in FIRMWARE_FILES}

    reference = fingerprints[setups[0]]
    for name in setups[1:]:
        differences = [file for file in NETWORK_FILES if fingerprints[name][file] != reference[file]]
        if differences:
            raise ValueError(f"setup {name} differs from {setups[0]} in {', '.join(differences)}")
    return fingerprints


def create_setups(staged, expected):
    for name, directory in staged.items():
        stage = ssh("mktemp -d /root/setups/.firmware-bench.XXXXXX")
        if not re.fullmatch(r"/root/setups/\.firmware-bench\.[A-Za-z0-9]+", stage):
            raise RuntimeError("unexpected staging path from the board")
        target = f"/root/setups/{name}"
        backup = f"{stage}.previous"
        try:
            ssh(f"cp -a /root/setups/{args.base_setup}/. {stage}/")
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes",
                            *(str(directory / file) for file in FIRMWARE_FILES),
                            f"{args.board}:{stage}/"], check=True, timeout=120)
            copied = checksums([f"{stage}/{file}" for file in (*FIRMWARE_FILES, *NETWORK_FILES)])
            if any(copied.get(f"{stage}/{file}") != digest for file, digest in expected[name].items()):
                raise RuntimeError(f"staged setup {name} does not match the expected files")
            # Keep the old setup until the complete replacement has been installed.
            ssh(
                f"set -e; if test -e {target}; then mv {target} {backup}; fi; "
                f"if mv {stage} {target}; then rm -rf {backup}; "
                f"else if test -e {backup}; then mv {backup} {target}; fi; exit 1; fi"
            )
        finally:
            ssh(f"rm -rf {stage}", check=False, timeout=20)


def switch_setup(name, expected):
    loaded = ssh(f"/root/switch_setup.sh {name}", timeout=180)
    last = loaded.splitlines()[-1] if loaded else ""
    if "server active" not in last or "app 1" not in last:
        raise RuntimeError(f"{name} did not load cleanly: {last}")
    actual = dict(zip((*FIRMWARE_FILES, *NETWORK_FILES), BOARD_FILES))
    board_digests = checksums(BOARD_FILES)
    changed = [file for file, path in actual.items() if board_digests.get(path) != expected[file]]
    if changed:
        raise RuntimeError(f"setup {name} did not install the expected {', '.join(changed)}")
    return last


def parse_eval(text):
    res = {}
    m = re.search(r"FPGA accuracy ([\d.]+)%", text)
    res["accuracy_pct"] = float(m.group(1)) if m else None
    m = re.search(r"FPGA latency per frame \(ms\): median ([\d.]+)\s+mean ([\d.]+)\s+p90 ([\d.]+)\s+p99 ([\d.]+)\s+max ([\d.]+)", text)
    if m:
        res["latency_ms"] = dict(zip(("median", "mean", "p90", "p99", "max"), map(float, m.groups())))
    rates = re.findall(r"(\d+) frames/s", text)
    res["replay_frames_per_s"] = int(rates[-1]) if rates else None
    return res


def bench(name, rnd, expected):
    print(f"\n=== round {rnd}: {name} ===", flush=True)
    last = switch_setup(name, expected)
    print(" ", last)
    info = {"setup": name, "round": rnd, "switch": last,
            "bitstream_md5": expected["spleat.bit.bin"], "weights_md5": expected["weights.txt"],
            "fabric_clock_mhz": fabric_clock_mhz()}
    time.sleep(3)
    info["sensors_before"] = sensors()
    with PowerSampler() as idle:
        time.sleep(args.idle_s)
    info["power_idle_w"] = idle.watts()
    npz = out_dir / f"{name}_round{rnd}.npz"
    cmd = [sys.executable, args.fpga_eval, "--host", args.board.split("@")[-1], "--save", str(npz)] + eval_args
    with PowerSampler() as busy:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    (out_dir / f"{name}_round{rnd}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f"fpga_eval failed on {name}:\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    info["power_replay_w"] = busy.watts()
    info["sensors_after"] = sensors()
    info["eval"] = parse_eval(r.stdout) | {"npz": str(npz)}
    power = [f"{info[key]['mean']:.3f} W" if info[key] else "unavailable"
             for key in ("power_idle_w", "power_replay_w")]
    print(f"  clock {info['fabric_clock_mhz']} MHz | idle {power[0]} | replay "
          f"{power[1]} | accuracy {info['eval']['accuracy_pct']} % | latency "
          f"{info['eval'].get('latency_ms')} | temps {info['sensors_after']}", flush=True)
    return info


def compare(ref_npz, npz):
    with np.load(ref_npz) as ref, np.load(npz) as run:
        if any(not np.array_equal(ref[key], run[key]) for key in ("labels", "sessions")):
            return "different replay clips"
        a, b = ref["outputs"], run["outputs"]
    if a.shape != b.shape:
        return f"different shapes {a.shape} vs {b.shape}"
    if a.dtype == b.dtype and a.tobytes() == b.tobytes():
        return "bit-exact"
    return f"{np.mean(a == b):.4%} equal by value"


def power_summary(rows, key):
    readings = [row[key]["mean"] for row in rows if row[key] is not None]
    if not readings:
        return None
    return {"mean": round(float(np.mean(readings)), 4),
            "std_between_rounds": round(float(np.std(readings)), 4)}


def format_power(data):
    return f"{data['mean']:.3f}±{data['std_between_rounds']:.3f}" if data else "unavailable"


def summary(results, setup_names):
    ref_npz = results[0]["eval"]["npz"]
    table = {}
    for name in setup_names:
        rows = [r for r in results if r["setup"] == name]
        if not rows:
            continue
        latencies = []
        for row in rows:
            with np.load(row["eval"]["npz"]) as measurement:
                latencies.append(measurement["latency_us"])
        lat = np.concatenate(latencies).astype(float) / 1000.0
        lat = lat[np.isfinite(lat) & (lat > 0)]
        table[name] = {
            "rounds": len(rows),
            "bitstream_md5": rows[0]["bitstream_md5"],
            "fabric_clock_mhz": sorted({r["fabric_clock_mhz"] for r in rows}, key=str),
            "latency_ms": {k: round(float(f(lat)), 4) for k, f in (
                ("median", np.median), ("mean", np.mean), ("p90", lambda x: np.percentile(x, 90)),
                ("p99", lambda x: np.percentile(x, 99)), ("max", np.max), ("min", np.min))} if lat.size else None,
            "power_idle_w": power_summary(rows, "power_idle_w"),
            "power_replay_w": power_summary(rows, "power_replay_w"),
            "accuracy_pct": sorted({r["eval"]["accuracy_pct"] for r in rows}, key=str),
            "outputs_vs_reference": sorted({compare(ref_npz, r["eval"]["npz"]) for r in rows}),
        }
    print(f"\n{'setup':24s} {'MHz':>5s} {'lat med':>8s} {'lat p99':>8s} {'lat max':>8s} {'idle W':>14s} {'replay W':>14s} {'acc %':>6s}  outputs")
    for name, t in table.items():
        lat = t["latency_ms"] or {}
        print(f"{name:24s} {str(t['fabric_clock_mhz'][0]):>5s} {lat.get('median', 0):8.3f} {lat.get('p99', 0):8.3f} "
              f"{lat.get('max', 0):8.3f} {format_power(t['power_idle_w']):>14s} "
              f"{format_power(t['power_replay_w']):>14s} "
              f"{str(t['accuracy_pct'][0]):>6s}  {', '.join(t['outputs_vs_reference'])}")
    return table


def restore(expected):
    print(f"\nrestoring {args.restore_setup} + inference.sh {args.restore_mode} ...", flush=True)
    print(" ", switch_setup(args.restore_setup, expected))
    if args.restore_mode == "none":
        return
    error = None
    for _ in range(2):
        try:
            started = ssh(f"/root/inference.sh {args.restore_mode}", timeout=180)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            error = exc
            continue
        print(" ", started.splitlines()[-1:])
        time.sleep(12)
        status = ssh("/root/inference.sh status", check=False)
        frames = re.findall(r"frames=(\d+) \((\d+)/s camera", status)
        if frames and int(frames[-1][1]) > 0:
            print(f"  camera feed ok: {frames[-1][1]} frames/s")
            return
    raise RuntimeError("the setup was restored but camera feed could not be verified; check /tmp/despot.log") from error


def main(argv=None):
    global args, eval_args, setups, out_dir
    args, eval_args, setups, staged = parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    expected = preflight(staged, eval_args)
    if args.dry_run:
        print("dry run: setup files, network equivalence and replay data passed preflight")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    create_setups(staged, expected)
    results = []
    try:
        for rnd in range(1, args.rounds + 1):
            for name in (setups if rnd % 2 else setups[::-1]):
                results.append(bench(name, rnd, expected[name]))
                (out_dir / "results.json").write_text(json.dumps({"runs": results}, indent=1) + "\n")
    finally:
        failure = sys.exc_info()[1]
        try:
            restore(expected[args.restore_setup])
        except Exception as exc:
            if failure is None:
                raise
            failure.add_note(f"Board restoration also failed: {exc}")

    table = summary(results, setups)
    (out_dir / "results.json").write_text(json.dumps({"summary": table, "runs": results}, indent=1) + "\n")
    print("\nsaved", out_dir / "results.json")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"firmware benchmark failed: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        sys.exit(1)
