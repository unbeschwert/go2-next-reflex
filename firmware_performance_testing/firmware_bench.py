#!/usr/bin/env python3
"""Compare firmware builds of the same network on the Kria: FPGA latency, board power, identical outputs.

Every build is a board setup in /root/setups/NAME (firmware + server config + camera config + weights, the
folder /root/switch_setup.sh loads). --firmware NAME=DIR creates NAME as a copy of --base-setup with the
spleat.bit.bin/.dtbo from the laptop folder DIR, so only the firmware differs between the setups.

Per setup and round: load it (switch_setup.sh stops the camera/bridge first), read the FPGA fabric clock,
temperatures and fan, sample the board's input power (INA260) while idle, then replay the same clips with
fpga_eval.py while sampling power. Rounds alternate the order (A B, B A, A B, ...) so slow drift (warm-up)
cancels. At the end: one table per setup, and whether each build gives exactly the same output spikes as the
first one. The board is always put back to --restore-setup and `inference.sh --restore-mode`, also on errors.

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
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FPGA_EVAL = next((p for p in (HERE / "fpga_eval.py", HERE.parent / "training" / "optimised_snn" / "fpga_eval.py")
                  if p.exists()), HERE / "fpga_eval.py")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                             allow_abbrev=False)
ap.add_argument("--board", default="root@192.168.77.1")
ap.add_argument("--setups", required=True, help="comma list of /root/setups names; the first is the reference")
ap.add_argument("--firmware", action="append", default=[], metavar="NAME=DIR",
                help="create/refresh setup NAME from --base-setup + the firmware in laptop folder DIR")
ap.add_argument("--base-setup", default="nb40_own4")
ap.add_argument("--restore-setup", default="nb40_own4")
ap.add_argument("--restore-mode", default="start", help="inference.sh argument afterwards ('none' = leave stopped)")
ap.add_argument("--out", required=True, help="folder for per-run .npz/.log and results.json")
ap.add_argument("--rounds", type=int, default=3, help="measure every setup this many times, order alternating")
ap.add_argument("--idle-s", type=float, default=15.0, help="seconds of idle power sampling per measurement")
ap.add_argument("--dry-run", action="store_true", help="only check files and board, change nothing")
ap.add_argument("--fpga-eval", default=str(FPGA_EVAL), help="replay tool (default: %(default)s)")
args, eval_args = ap.parse_known_args()   # everything else goes to fpga_eval.py (--zip, --sessions, ...)

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
out_dir = Path(args.out).expanduser().resolve()
setups = args.setups.split(",")

# Board-side power sampler: prints "<seconds> <microwatts>" every 50 ms until killed.
SAMPLER = r"""
import glob, sys, time
for d in glob.glob('/sys/class/hwmon/hwmon*'):
    if open(d + '/name').read().startswith('ina260'):
        break
else:
    sys.exit('no ina260 power sensor')
t0 = time.time()
while True:
    print(f'{time.time() - t0:.3f} {open(d + "/power1_input").read().strip()}', flush=True)
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
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            parts = line.split()
            if len(parts) == 2:
                self.samples.append((time.time(), int(parts[1]) / 1e6))

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=10)

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


def create_setups():
    for spec in args.firmware:
        name, _, folder = spec.partition("=")
        folder = Path(folder).expanduser().resolve()
        for f in ("spleat.bit.bin", "spleat.dtbo"):
            if not (folder / f).exists():
                sys.exit(f"{folder / f} missing")
        local = {f: md5(folder / f) for f in ("spleat.bit.bin", "spleat.dtbo")}
        print(f"setup {name}: {args.base_setup} + firmware {folder} (bit {local['spleat.bit.bin'][:12]})")
        if args.dry_run:
            continue
        ssh(f"test -d /root/setups/{args.base_setup}")
        ssh(f"rm -rf /root/setups/{name}.tmp && cp -a /root/setups/{args.base_setup} /root/setups/{name}.tmp")
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(folder / "spleat.bit.bin"), str(folder / "spleat.dtbo"),
                        f"{args.board}:/root/setups/{name}.tmp/"], check=True, timeout=120)
        remote = ssh(f"cd /root/setups/{name}.tmp && md5sum spleat.bit.bin spleat.dtbo")
        for f, h in local.items():
            if h not in remote:
                sys.exit(f"copy of {f} to the board does not match (local {h})")
        ssh(f"rm -rf /root/setups/{name} && mv /root/setups/{name}.tmp /root/setups/{name}")


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


def bench(name, rnd):
    print(f"\n=== round {rnd}: {name} ===", flush=True)
    loaded = ssh(f"/root/switch_setup.sh {name}", timeout=180)
    last = loaded.splitlines()[-1] if loaded else ""
    print(" ", last)
    if "server active" not in last or "app 1" not in last:
        raise RuntimeError(f"{name} did not load cleanly: {last}")
    info = {"setup": name, "round": rnd, "switch": last,
            "bitstream_md5": ssh(f"md5sum /root/setups/{name}/spleat.bit.bin").split()[0],
            "weights_md5": ssh("md5sum /root/weights.txt").split()[0],
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
    print(f"  clock {info['fabric_clock_mhz']} MHz | idle {info['power_idle_w']['mean']:.3f} W | replay "
          f"{info['power_replay_w']['mean']:.3f} W | accuracy {info['eval']['accuracy_pct']} % | latency "
          f"{info['eval'].get('latency_ms')} | temps {info['sensors_after']}", flush=True)
    return info


def compare(ref_npz, npz):
    a, b = np.load(ref_npz)["outputs"], np.load(npz)["outputs"]
    if a.shape != b.shape:
        return f"different shapes {a.shape} vs {b.shape}"
    return "bit-exact" if np.array_equal(a, b) else f"{np.mean(a == b):.4%} identical"


def summary(results):
    ref_npz = results[0]["eval"]["npz"]
    table = {}
    for name in setups:
        rows = [r for r in results if r["setup"] == name]
        if not rows:
            continue
        lat = np.concatenate([np.load(r["eval"]["npz"])["latency_us"] for r in rows]).astype(float) / 1000.0
        lat = lat[np.isfinite(lat) & (lat > 0)]
        idle = np.array([r["power_idle_w"]["mean"] for r in rows])
        busy = np.array([r["power_replay_w"]["mean"] for r in rows])
        table[name] = {
            "rounds": len(rows),
            "bitstream_md5": rows[0]["bitstream_md5"],
            "fabric_clock_mhz": sorted({r["fabric_clock_mhz"] for r in rows}, key=str),
            "latency_ms": {k: round(float(f(lat)), 4) for k, f in (
                ("median", np.median), ("mean", np.mean), ("p90", lambda x: np.percentile(x, 90)),
                ("p99", lambda x: np.percentile(x, 99)), ("max", np.max), ("min", np.min))} if lat.size else None,
            "power_idle_w": {"mean": round(float(idle.mean()), 4), "std_between_rounds": round(float(idle.std()), 4)},
            "power_replay_w": {"mean": round(float(busy.mean()), 4), "std_between_rounds": round(float(busy.std()), 4)},
            "accuracy_pct": sorted({r["eval"]["accuracy_pct"] for r in rows}, key=str),
            "outputs_vs_reference": sorted({compare(ref_npz, r["eval"]["npz"]) for r in rows}),
        }
    print(f"\n{'setup':24s} {'MHz':>5s} {'lat med':>8s} {'lat p99':>8s} {'lat max':>8s} {'idle W':>14s} {'replay W':>14s} {'acc %':>6s}  outputs")
    for name, t in table.items():
        lat = t["latency_ms"] or {}
        print(f"{name:24s} {str(t['fabric_clock_mhz'][0]):>5s} {lat.get('median', 0):8.3f} {lat.get('p99', 0):8.3f} "
              f"{lat.get('max', 0):8.3f} {t['power_idle_w']['mean']:7.3f}±{t['power_idle_w']['std_between_rounds']:.3f} "
              f"{t['power_replay_w']['mean']:7.3f}±{t['power_replay_w']['std_between_rounds']:.3f} "
              f"{str(t['accuracy_pct'][0]):>6s}  {', '.join(t['outputs_vs_reference'])}")
    return table


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    create_setups()
    have = ssh("ls /root/setups").split()
    created = {spec.partition("=")[0] for spec in args.firmware} if args.dry_run else set()
    missing = [s for s in setups + [args.restore_setup] if s not in have and s not in created]
    if missing:
        sys.exit(f"not on the board: {missing} (have {have})")
    if args.dry_run:
        print("dry run: setups present:", have, "| fpga_eval args:", eval_args)
        return
    results = []
    try:
        for rnd in range(1, args.rounds + 1):
            for name in (setups if rnd % 2 else setups[::-1]):
                results.append(bench(name, rnd))
                (out_dir / "results.json").write_text(json.dumps({"runs": results}, indent=1) + "\n")
    finally:
        print(f"\nrestoring {args.restore_setup} + inference.sh {args.restore_mode} ...", flush=True)
        print(" ", ssh(f"/root/switch_setup.sh {args.restore_setup}", check=False, timeout=180).splitlines()[-1:])
        if args.restore_mode != "none":
            # The camera client sometimes dies at start-up after FPGA reloads ("Error TimeHigh discrepancy" in
            # /tmp/despot.log, seen 2026-11-22): no frames reach the SNN. Check and start once more if so.
            for attempt in (1, 2):
                print(" ", ssh(f"/root/inference.sh {args.restore_mode}", check=False, timeout=180).splitlines()[-1:])
                time.sleep(12)
                status = ssh("/root/inference.sh status", check=False)
                m = re.findall(r"frames=(\d+) \((\d+)/s camera", status)
                if m and int(m[-1][1]) > 0:
                    print(f"  camera feed ok: {m[-1][1]} frames/s")
                    break
                print(f"  camera feed not running ({m[-1] if m else 'no status'}), "
                      + ("starting again" if attempt == 1 else "CHECK THE BOARD: /tmp/despot.log"))
    table = summary(results)
    (out_dir / "results.json").write_text(json.dumps({"summary": table, "runs": results}, indent=1) + "\n")
    print("\nsaved", out_dir / "results.json")


if __name__ == "__main__":
    main()
