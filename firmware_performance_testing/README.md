# Firmware performance testing

This benchmark checks the inference stage of the Go2 gesture reflex. It replays recorded gesture clips on the Kria KV260 and checks whether a firmware build changes the output spikes that the ROS 2 bridge uses to decide how the robot should respond. Keeping the trained network fixed lets us measure the effect of each build on inference latency and board power. See the [gesture pipeline](../doc/gesture-pipeline.md) for the live camera and robot path.

The first comparison covers three riversdk.dev build strategies for the deployed 40×40 gesture network. Use the same network, exported weights, server and camera configuration, and class map for every build you compare.

## Result: nb40, three strategies (2026-11-22)

Same network (`models/optimised_snn/nb40_model.json`, 40×40 input), same form options (KV260, GenX320, weights Q2.6, potentials Q4.12), same weights on the board (`own_4gestures_allpeople_nb40_q8h`, md5 `2d8fac769fbe`). Only the strategy differs.

| | Faster generation | Better latency | Less resources |
|---|---|---|---|
| riversdk.dev job | #28 (on board 1, setup `nb40_own4`) | #40 | #42 ¹ |
| Build time | **8 min 6 s** | 9 min 2 s | 8 min 41 s |
| FPGA fabric clock | 100 MHz | 100 MHz | 100 MHz |
| FPGA latency per frame, median / mean | 0.184 / 0.203 ms | 0.184 / 0.203 ms | 0.184 / 0.203 ms |
| FPGA latency per frame, p99 | 0.545 ms | 0.543 ms | 0.556 ms |
| Frames over 1 ms | 0.18 % | 0.19 % | 0.20 % |
| Board power, idle | **3.415 W** | 3.421 W | 3.419 W |
| Board power, during replay | **3.775 W** | 3.782 W | 3.783 W |
| Outputs vs Faster generation | reference | bit-exact (3/3 runs) | bit-exact (3/3 runs) |

¹ The first try (#41) failed after 30 s at stage "Design hardware" with no log; the identical resubmission built fine.

**Keep Faster generation as the default for nb40.** It built fastest, and the three builds produced the same outputs and nearly the same inference latency.

- **Better latency and Less resources produce the same hardware.** Their bitstreams differ only in the 3-byte build time in the file header. Faster generation is a different place-and-route: 1.36 MB of the 7.8 MB bitstream differ.
- **Same clock, same overlay.** All three `.dtbo` files are byte-identical (fabric clock 100 MHz). The median per-frame latency was 0.184 ms for each build.
- **Same results.** Every output spike of every frame matched in all 9 runs.
- **Power:** The recorded idle and replay means differ by less than 0.01 W across the three builds. Each mean covers three rounds.
- Builds are reproducible: two Faster generation builds of the same model (#28, #32) differ only in the build time stamp.

Rerun the comparison for a larger network before choosing its build strategy.

Test data: 65 clips from sessions 22, 29, 30, 33 (`go2_gestures_s20-33.zip`), 40 empty frames + 100 frames per clip = 9,100 frames per measurement, 3 rounds per build. The accuracy was 100 % for all builds. These weights were trained on all people including these sessions, so that number is only a consistency check, not a generalisation score. Raw data: [results/2026-11-22_nb40_strategies/](results/2026-11-22_nb40_strategies/) (`builds.json` = jobs and checksums, `results.json` = every measurement, `measurements/` = per-frame outputs and latency (`.npz`) and the replay log of every measurement).

## Files

| File | What it does |
|---|---|
| `firmware_bench.py` | The test (laptop side). Creates one board setup per build, measures, compares, restores the board |
| `dtbo_dump.py` | Prints a device-tree overlay (`.dtbo`) without `dtc`, e.g. to compare clocks between builds |
| `switch_setup.sh` | Board script, copy to `/root/switch_setup.sh`. Loads `/root/setups/NAME` (firmware, server config, camera config, weights) |
| `results/` | Results of past runs |

The replay itself is `training/optimised_snn/fpga_eval.py`.

## How the test works

```
laptop: fpga_eval.py ──TCP :3333, one frame at a time──► spleat-server ──► FPGA (build under test)
                     ◄── 5 output spikes + FPGA latency ──
board power sensor (INA260) ── read every 50 ms over ssh ──► laptop
```

1. **Preflight.** The benchmark validates the requested replay clips and checks that every setup uses the same weights, server config, camera config, and class map. `--firmware NAME=DIR` stages `spleat.bit.bin` and `spleat.dtbo` from the laptop with the other files from `--base-setup` (default `nb40_own4`). The files are verified before the candidate setup is installed. Use a separate name for each staged build.
2. **One measurement** (about 38 s):
   - `switch_setup.sh NAME` stops the camera and bridge, copies the files into place, reprograms the FPGA (`xmutil loadapp spleat`) and restarts the server. The test only continues once the SPLEAT app and server are active and the installed files match the expected checksums.
   - Records the checksums of the loaded firmware and weights, the fabric clock (`pl0_ref` in the clock tree), the chip temperatures and the fan.
   - 15 s of idle board power (FPGA loaded, no frames).
   - Replays the clips with `fpga_eval.py` (block sums, as the camera client sends them) while sampling power. The server returns the output spikes and the FPGA latency for every frame.
3. **Rounds alternate the order** (A B C, C B A, A B C), so slow drift such as warm-up affects every build equally.
4. **Comparison:** latency over all frames of all rounds; power as the mean per measurement, averaged over the rounds; outputs compared byte for byte with the first run of the first setup after checking clip labels and session order.
5. **Restore.** Once measurements begin, the benchmark attempts to reload `--restore-setup` and run `inference.sh start`, including after errors. It checks the camera frame rate, retries camera start once, and reports a restoration failure. Use `--restore-mode "start --headless"` for a headless board or `--restore-mode none` to leave camera inference stopped.

## Running it

**1. Build the firmware** with `scripts/firmware_job.py` (needs the riversdk.dev session cookie, see [doc/deploy-model.md](../doc/deploy-model.md)). Same model and formats, one build per option you want to compare:

```bash
for s in better-latency less-resources faster-generation; do
  python3 scripts/firmware_job.py submit --model models/optimised_snn/nb40_model.json \
    --weights Q2.6 --potentials Q4.12 --prophesee-ip genx320 --strategy $s \
    --name nb40-$s --download --out ~/firmware/nb40_$s
done
```

`submit` reuses a finished job with the same model and options. Prophesee IP must be GenX320, otherwise the firmware has no camera interface.

**2. Compare the overlays** (clock rates):

```bash
python3 firmware_performance_testing/dtbo_dump.py ~/firmware/nb40_better-latency/spleat.dtbo | grep -E 'rate|frequency'
diff <(python3 firmware_performance_testing/dtbo_dump.py A/spleat.dtbo --all) \
     <(python3 firmware_performance_testing/dtbo_dump.py B/spleat.dtbo --all)
```

**3. Board prerequisites:** ssh key login to `root@192.168.77.1`, `switch_setup.sh` in `/root/`, a base setup in `/root/setups/` whose weights match the builds' formats, and nobody else using the board while the test runs.

```bash
scp firmware_performance_testing/switch_setup.sh root@192.168.77.1:/root/
```

**4. Dry run, then the test.** Run from the repository root. The dry run validates the dataset and reads checksums over SSH without changing the board. Replay options such as `--zip` and `--size` go to `fpga_eval.py`.

```bash
python3 firmware_performance_testing/firmware_bench.py --dry-run \
  --firmware nb40_better_latency=~/firmware/nb40_better-latency \
  --setups nb40_own4,nb40_better_latency --out ~/firmware/bench \
  --zip ~/data/go2_gestures_s20-33.zip --sessions 22,29,30,33 --limit 25 --size 40

python3 firmware_performance_testing/firmware_bench.py \
  --firmware nb40_better_latency=~/firmware/nb40_better-latency \
  --firmware nb40_less_resources=~/firmware/nb40_less-resources \
  --setups nb40_own4,nb40_better_latency,nb40_less_resources --rounds 3 --out ~/firmware/bench \
  --zip ~/data/go2_gestures_s20-33.zip --sessions 22,29,30,33 --limit 25 --size 40 \
  --class-names 'Heart,Turn left,Turn right,Follow me,No action'
```

The first name in `--setups` is the reference. 3 builds × 3 rounds take about 6.5 minutes. During that time the camera and bridge are off. Output: one `.npz` (per-frame outputs and latency) and one `.log` per measurement, plus `results.json` with a summary table.

## What it does not measure

- **FPGA resource use** (LUTs, BRAM): riversdk.dev provides no utilisation report.
- **The live camera path**: the replay feeds the server directly. The camera IP is the same in every build.
- **The robot's reaction time**: about 0.6 s, almost all of it the bridge's evidence window. The FPGA takes about 0.2 ms per frame.
- **The FPGA's own power**: the INA260 measures the whole board (about 3.4 W idle).
