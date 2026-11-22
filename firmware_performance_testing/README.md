# Firmware performance testing

Compares SPLEAT firmware builds of **the same network** on the Kria KV260. It measures FPGA latency per frame and board power, and checks that every build gives bit-identical outputs. It was written to compare the three **implementation strategies** of the riversdk.dev firmware generator ("Better latency", "Less resources", "Faster generation"). It works for any set of builds of one network, for example different Q formats as long as the weights match them.

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

**Conclusion: keep Faster generation.** For nb40 the strategy changes nothing that matters:

- **Better latency and Less resources produce the same hardware.** Their bitstreams differ only in the 3-byte build time in the file header. Faster generation is a different place-and-route: 1.36 MB of the 7.8 MB bitstream differ.
- **Same clock, same overlay.** All three `.dtbo` files are byte-identical (fabric clock 100 MHz). Without a faster clock, a "latency" layout gains nothing, and the measured latency is identical.
- **Same results.** Every output spike of every frame matched in all 9 runs.
- **Power:** Faster generation draws 5 mW less idle and 7 mW less during the replay. That is about three times the gap between the two identical designs (#40 vs #42: 1–2.5 mW), so it is probably real, but it is 0.2 % of the board's power.
- Builds are reproducible: two Faster generation builds of the same model (#28, #32) differ only in the build time stamp.

The strategy would matter for a larger network that does not fit or does not meet timing with Faster generation.

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

1. **Only the firmware differs.** `--firmware NAME=DIR` creates `/root/setups/NAME` as a copy of `--base-setup` (default `nb40_own4`) with `spleat.bit.bin` and `spleat.dtbo` from the laptop folder `DIR`. Checksums are compared after the copy.
2. **One measurement** (about 38 s):
   - `switch_setup.sh NAME` stops the camera and bridge, copies the files into place, reprograms the FPGA (`xmutil loadapp spleat`) and restarts the server. The test only continues if it reports `server active | app 1`.
   - Records the checksums of the loaded firmware and weights, the fabric clock (`pl0_ref` in the clock tree), the chip temperatures and the fan.
   - 15 s of idle board power (FPGA loaded, no frames).
   - Replays the clips with `fpga_eval.py` (block sums, as the camera client sends them) while sampling power. The server returns the output spikes and the FPGA latency for every frame.
3. **Rounds alternate the order** (A B C, C B A, A B C), so slow drift such as warm-up affects every build equally.
4. **Comparison:** latency over all frames of all rounds; power as the mean per measurement, averaged over the rounds; outputs compared value by value with the first run of the first setup.
5. **Restore:** the board always goes back to `--restore-setup` and `inference.sh start`, also after an error. The camera client sometimes dies at start-up after FPGA reloads (`Error TimeHigh discrepancy` in `/tmp/despot.log`, 0 frames/s), so the test checks the camera frame rate and starts once more if needed.

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

**4. Dry run, then the test** (from the repo root; options the script does not know go to `fpga_eval.py`):

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
