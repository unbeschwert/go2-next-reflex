# Debugging the SNN on the FPGA: why the hardware and the simulation disagreed

A step-by-step record of how we checked that our trained SNN runs correctly on the Kria KV260.
It covers what didn't match, each hypothesis we tested, how we tested it, and what we learned.
The scripts are in [`scripts/fpga_debug/`](../scripts/fpga_debug/), so every experiment can be
re-run on a new deployment.

**Short version:** the FPGA computes almost the same thing as `riverops`' fixed-point simulation.
The big difference was **state**: `spleat-server` **never resets the neurons' membrane potentials**,
not between clips and not when a client disconnects. The only reset is **restarting the service**
(`systemctl restart spleat-server.service`). The model was trained and simulated as if every clip
starts from zero. With a restart before each clip, the hardware starts from zero too: zero input then
gives the same output every time, the first difference with the simulation moves from frame 0 to frames
3–15, spike counts per class agree within a few spikes, and predictions agree 10/10.
What remains after that is **arithmetic**: weights, biases, threshold and leak are bit-identical on both
sides, and the first difference is a neuron sitting exactly at the threshold that crosses one frame later
on the hardware. That's a 1-LSB rounding difference from the hardware's internal order of operations
(Hypothesis F).

---

## 1. Setup

| | |
|---|---|
| Model | run `motion-buddy-40-v1`: 5-layer SNN, 40×40×2 input, 5 classes (see [training/README.md](../training/README.md)) |
| Formats | weights Q2.6 (8 bit), potentials Q4.12 (16 bit), hard reset |
| Firmware | riversdk.dev job #28, built from exactly our `model.json` (same SHA-256) |
| On the board | bitstream + overlay, `weights.txt`, `config.yaml` patched to 40×40, `metavision_camera.json` with downsampling 8; previous setup backed up in `/root/backup-20250529-225932/` |
| Restart | `systemctl stop spleat-server; xmutil unloadapp; xmutil loadapp spleat; systemctl start spleat-server` (restarting the service alone does **not** load a new bitstream) |

**How we talk to the FPGA:** [`scripts/fpga.py`](../scripts/fpga.py) (`FPGAClient`) sends one frame per
request to `spleat-server` on port 3333, as a 4-byte length plus CBOR `{"id", "payload": float32[2×40×40]}`.
It gets back one output per frame: `float32[5×1×1]`, which turned out to be **0/1 spikes** of the 5 output
neurons for that time step. A clip is 100 such frames (10 ms each), and the prediction is the class with
the most spikes.

**What we compare against:** `riverops.nn.prepare_ptq(model, formats)` gives the "fixed-point simulation".
It folds BatchNorm into the convolutions and rounds every intermediate value to the Q formats, i.e. it
approximates the hardware's integer arithmetic in PyTorch.

**How we measure agreement:**
- **identical spikes:** the share of (time step, class) slots where FPGA and simulation output the same 0/1.
  This is strict: a spike that comes one frame late counts as two mismatches.
- **same prediction:** both pick the same class for a clip. This is what matters for the robot.

---

## 2. Symptom

The first comparison: 5 test clips (person 13, never seen in training), with zero frames between clips to "let things settle".

| | |
|---|---|
| Same prediction | 4/5 |
| Identical spikes | ~74 % |
| Pattern | the FPGA fires **fewer** spikes: classes that aren't winning fire 0–2 times on the FPGA and 14–24 times in simulation |

Accuracy was fine (7/12 = 58 % on a larger sample, in line with the 56 % test accuracy), but the
simulation clearly wasn't describing the hardware.

---

## 3. The hypotheses we tested

### Hypothesis A: the biases are read in another format, or ignored

**Idea:** the BN-folded biases are exported in the weight format (Q2.6). If the hardware read them as Q4.12,
they'd be 64× too small. That would remove the background firing of the non-winning classes.

**Test** ([`fpga_reconnect_check.py`](../scripts/fpga_debug/fpga_reconnect_check.py)): compare the FPGA with
three simulations: biases as exported, biases × 0, biases ÷ 64.

| Simulation | Identical spikes | Same prediction |
|---|---|---|
| as exported | 73.1 % | 27/30 |
| biases × 0 | 77.0 % | 11/30 |
| biases ÷ 64 | 77.2 % | 11/30 |

**Verdict: rejected.** Removing the biases matches individual spikes slightly better but gets the
predictions badly wrong. The hardware does use the biases.

### Hypothesis B: the pipeline adds a delay

**Idea:** SPLEAT runs one hardware unit per layer, as a pipeline. The output might lag the input by a few frames.

**Test** ([`fpga_diag.py`](../scripts/fpga_debug/fpga_diag.py)): compare FPGA frame *t + k* with simulation frame *t* for k = 0…7.

| Lag k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Identical spikes | 74.0 % | 73.7 % | 74.0 % | 73.6 % | 73.9 % | 73.3 % | 73.5 % | 72.2 % |

**Verdict: rejected.** No delay improves the match. The output belongs to the frame just sent. The same
run showed that FPGA and simulation already differ at **frame 0** in every clip.

### Hypothesis C: fixed-point vs floating-point arithmetic

**Idea:** the hardware computes only with integers, while PyTorch uses floats. Rounding could change which neurons cross the threshold.

**Test** ([`fixed_point_check.py`](../scripts/fpga_debug/fixed_point_check.py)), on the same 10 clips:

1. **Precision ladder:** how much does going from float to fixed point change the spikes of the same model?
2. **Rounding mode:** `riverops` rounds to nearest. Hardware often rounds down or truncates, so we ran the
   simulation with each, by swapping `riverops.nn.quantization.RoundSTE.forward`.

| Model | vs fixed-point simulation | vs FPGA | Total spikes |
|---|---|---|---|
| QAT model, float maths with fake quantization | 95.5 %, 10/10 | 74.3 %, 9/10 | 1,430 |
| Fixed point, round to nearest | n/a | **74.0 %, 9/10** | 1,433 |
| Fixed point, truncate | 77.5 %, 8/10 | 68.6 %, 7/10 | 1,727 |
| Fixed point, round down | 59.0 %, 4/10 | 55.1 %, 4/10 | 2,185 |
| **FPGA** | 74.0 %, 9/10 | n/a | **845** |

**Verdict: a small effect, not the cause.** Fixed-point arithmetic changes only ~4.5 % of spikes. The
other rounding modes match the FPGA *worse* and push the spike count *up*, while the FPGA fires fewer.
`riverops`' round-to-nearest is the closest of the three.

### Hypothesis D: soft vs hard reset

**Idea:** after a spike, a neuron's potential is either set to 0 (hard) or reduced by the threshold (soft).
A mismatch here would change the timing of spikes.

**Check:** we trained with `reset_mode: HARD`, and the hardware defaults to hard reset. Also, the mismatch
starts at frame 0, before any neuron has spiked, so the reset rule can't be involved yet.

**Verdict: rejected.** Note: `model.json` doesn't record the reset mode, so the hardware always uses its
default whatever we train with.

### Hypothesis E: the potentials are never reset ✔

**Idea:** if the FPGA doesn't start each clip from zero potentials, it differs at frame 0, and differs between
runs of the same clip. We first assumed a new TCP connection resets the potentials.

**Test 1: is the FPGA consistent with itself?** ([`fixed_point_check.py`](../scripts/fpga_debug/fixed_point_check.py), part 3)
The same 4 clips, each sent 3 times on fresh connections:
- identical spikes between runs: **83–86 %**
- same prediction: 100 %

**The hardware doesn't even agree with itself** on the same input.

**Test 2: zeros → disconnect → reconnect → zeros** ([`reset_check.py`](../scripts/fpga_debug/reset_check.py)):

| Run (new connection each time) | Output spikes per class over 100 zero frames |
|---|---|
| 1–5 | `[0 0 0 0 68]`, `[0 0 0 0 70]`, `[0 0 0 0 66]`, `[0 0 0 0 70]`, `[0 0 0 0 64]`: not identical |
| after a real clip | `[9 0 4 0 65]`: the clip's activity **carries over** into the next connection |

**Test 3: what does zero input look like in simulation, from a true zero start?**

| Simulation, zero input | Output spikes per class |
|---|---|
| frames 0–99 (starting from zero potentials) | `[22 21 22 23 71]`: a start-up burst in the first ~25 frames |
| frames 100–199 (warmed up) | `[0 0 0 4 63]` |
| frames 200–299 | `[0 0 0 0 66]` |

The FPGA's "fresh" connection looks exactly like the simulation's **warmed-up** state, `[0 0 0 0 ~66]`, never
like the start-up burst. **Reconnecting does not reset the potentials;** the network just keeps running.

**Test 4: give both sides the same starting state** ([`warm_match.py`](../scripts/fpga_debug/warm_match.py)).
On one connection, send 300 zero frames and then the clip, to both the FPGA and the simulation. The 300 frames
drive both into the same warmed-up state, whatever came before. Compare only the clip frames.

| | Identical spikes | Same prediction | Spike counts |
|---|---|---|---|
| Each side's own start (before) | 74 % | 27/30 | FPGA 845 vs sim 1,433 |
| **Both warmed up (300 zero frames)** | **87.6 %** (82.6–94.2 %) | **10/10** | within a few spikes per class, e.g. `[64 0 2 0 6]` vs `[68 0 0 0 9]` |

This made the state theory very likely, but the comparison still differed from frame 0: the two warm
states aren't exactly the same (class 4 fires in a repeating rhythm even with no input, and the two sides can
be at different points in that rhythm). We needed a real reset.

**Test 5: restart the service before every sample** ([`restart_check.py`](../scripts/fpga_debug/restart_check.py)).
`systemctl restart spleat-server.service` is the only thing that resets the potentials. After each restart,
stream one sample and compare it with the simulation started from zero potentials, which is the state the
model was trained in.

| After a restart | Result |
|---|---|
| 100 zero frames, 3 restarts | **identical every time**: `[20 19 21 22 71]`; simulation `[22 21 22 23 71]`, 94.8 % identical spikes |
| 10 clips, restart before each | first difference at **frames 3–15** (instead of 0); 86.0 % identical spikes (78.8–92.6 %); **same prediction 10/10** |
| Spike counts, e.g. dance clip | FPGA `[85 16 19 16 14]` vs simulation `[85 17 19 17 15]` |

**Verdict: confirmed, this is the main cause.** With a real reset, the hardware is deterministic and follows the
simulation exactly for the first frames. The two then drift apart slowly: a single spike that differs, for example
because a potential rounds differently right at the threshold, changes everything downstream. **That residual drift
is the actual fixed-point effect** (Hypothesis C), and it is small: spike counts per class stay within a few spikes.

### Hypothesis F: the remaining drift is arithmetic (LSB rounding) ✔

There's no faster reset in this server version, so to compare many clips with a clean start we **concatenate**
them. We restart once, stream 20 clips back-to-back (2,000 frames) on one connection, and run the same
concatenated sequence through the simulation from zero. Same starting state, same input history: whatever still
differs must come from the arithmetic.
([`concat_check.py`](../scripts/fpga_debug/concat_check.py) talks to the board and saves the FPGA outputs;
[`concat_offline.py`](../scripts/fpga_debug/concat_offline.py) tests more variants against the saved outputs
without the board.)

**The hardware is deterministic:** two runs, each after its own restart, gave identical outputs for all 2,000 frames.

**Same numbers on both sides:** all 174,880 weights and 301 biases in `weights.txt` are identical to the simulation's,
and every layer has threshold 4096 (1.0) and leak 287 (287/4096 ≈ 0.0701).

**Variants of the neuron update we tried** (identical spikes over 2,000 frames, first differing frame, same prediction out of 20):

| Variant | Identical | First diff | Same pred. |
|---|---|---|---|
| riverops: `v − round(v·0.07) + I`, round to nearest | 84.9 % | 3 | 18 |
| exported leak 287/4096, nearest / half-up | 85.9 % | 3 | 18 |
| leak 287/4096, floor (`v·287 >> 12`) / truncate | 84.5–84.7 % | 3 | 18–20 |
| single rounding of the whole update `v − v·leak + I`, 4 rounding modes, 2 leak values | 83.9–85.9 % | 3 | 18–19 |
| strict `v > threshold` instead of `≥` | 83.6–84.5 % | **0** | 19–20 |
| leak applied after adding the input | 76.7 % | **0** | 15 |
| inputs clamped to the Q4.12 range | no change (largest input is 2.09, limit 8) | 3 | 18 |

**What happens at frame 3:** frames 0–2 match exactly. At frame 3 a single output neuron (class 0) fires in the
simulation, and on the FPGA it fires one frame later, at frame 4. In the simulation, that neuron's potential is
**exactly at the threshold (0 LSB)** at that moment. One LSB (1/4096) less on the hardware, and it waits a frame.
After that, the two runs have different spike histories and drift apart slowly.

**Verdict: confirmed.** The remaining mismatch is attributable to arithmetic: LSB-level rounding differences in the
neuron update. They flip spikes only for neurons that sit exactly at the threshold, which happens for 0.02–0.11 % of
neuron-steps. The order of operations inside the hardware isn't known exactly, so no simple variant reproduces it
bit-exactly. The structural choices are confirmed, though: `≥` threshold, leak before input, and hard reset. The
other choices give the same frame-3 divergence or diverge earlier. The effect on the task is small: the same
prediction for 18–20 of 20 clips, and per-class spike counts within a few spikes.

> **Careful with restarts:** systemd allows at most **5 starts per minute** for this service
> (`StartLimitBurst=5`, `StartLimitIntervalUSec=1min`). Our first restart loop hit that limit and the service stayed
> down (`failed (Result: start-limit-hit)`) until we ran `systemctl reset-failed spleat-server.service` and started
> it again. `restart_check.py` therefore runs `systemctl reset-failed …; systemctl restart …` before every sample.
> Restarting also interrupts anyone else using the server, e.g. the live camera or ROS.

---

## 4. What we learned

1. **An SNN is a stateful system.** Each neuron's membrane potential carries information from one time step to the
   next. "Run the model on a clip" only means something once you know the starting state.
2. **The hardware runs continuously.** `spleat-server` never resets the potentials: not between frames, not between
   clips, and not when a client disconnects. Only restarting the service resets them. The live camera runs the
   same way, with no resets at all.
3. **Training and simulation must use the same starting state as deployment.** We trained every clip from zero
   potentials, which never happens on the board. When the starting state is matched, simulation and hardware agree well.
4. **Change one thing at a time, and measure the noise floor first.** The FPGA's own run-to-run agreement (83–86 %)
   was the key number: no model of the hardware can match it better than the hardware matches itself.
5. **Look at the spike rasters, not just accuracy.** Accuracy looked fine (9/10 same predictions) while the
   spike-level picture showed the systematic difference.
6. **Bias, rounding mode, reset mode and pipeline lag were all reasonable suspects.** Each was ruled out with a
   cheap, targeted experiment instead of guesswork.

---

## 5. Consequences for the project

- **Testing on the FPGA:**
  - **To measure accuracy as trained** (every clip from zero): restart the service before each clip
    (`systemctl reset-failed …; systemctl restart …`) and compare with the simulation from zero. This is the
    `restart_check.py` method. It's slow (a restart per clip), so use it on a sample of clips.
  - **For fast checks without restarts:** warm up on the same connection (300 zero frames), then the clip, and compare
    with a simulation warmed up the same way (`warm_match.py`). This is close but not bit-exact, because the warm states can differ.
  - Reconnecting is never a reset.
- **Training:** train for the way the board actually runs. Give each training clip a random warm-up (zero frames or
  the end of another clip) and score only the clip frames. Also consider target 0 spikes for wrong classes, so the
  model doesn't rely on a bias-driven background rate (see Hypothesis A).
- **Live use / ROS:** `gesture_node` averages the per-frame outputs over a sliding window, which already suits a
  network that never resets.
- **Open questions for the organisers:**
  - In this server version, restarting the service is the only reset, so concatenate samples for comparisons (Hypothesis F).
  - The exact order of operations in the hardware is hard to predict, so 1-LSB differences are expected. A
    bit-exact simulator would need the hardware's exact update order; the task-level effect is small.

---

## 6. Re-running the experiments

Run from the repository root, with the board reachable at `durance-kria.local` and the train dependency group
installed (`uv sync --group train`). Each script loads `runs/motion-buddy-40-v1`; edit `run = Path(...)` for
another run.

| Script | What it does | Takes |
|---|---|---|
| `fpga_check.py` | first comparison: 1 clip per class, zero frames in between (Section 2) | ~1 min |
| `fpga_reconnect_check.py` | bias variants, new connection per clip (Hypothesis A) | ~5 min |
| `fpga_diag.py` | records FPGA outputs, lag test, first differing frame, saves `runs/fpga_debug/fpga_diag.npz` (B) | ~2 min |
| `fixed_point_check.py` | precision ladder, rounding modes, FPGA vs itself (C, E1); needs `fpga_diag.npz` | ~3 min |
| `reset_check.py` | zeros → reconnect → zeros, carry-over after a clip (E2, E3) | ~1 min |
| `warm_match.py` | matched warm start: 300 zero frames, then the clip (E4) | ~2 min |
| `restart_check.py` | **restarts spleat-server before each sample**; zeros and 10 clips vs simulation from zero (E5) | ~3 min |
| `concat_check.py` | **one restart, 20 clips back-to-back**; neuron-update variants, near-threshold margins; saves `runs/fpga_debug/concat_fpga.npz` (F) | ~3 min |
| `concat_offline.py` | more variants (input clamp, leak order, single rounding) against the saved FPGA outputs, no board needed (F) | ~2 min |
| `fpga_accuracy.py --run <run> --budget <s>` | FPGA vs fixed-point simulation accuracy on sampled val/test clips, restart per clip; any run, frames sent as despot sends them (Section 7) | ~4.5 s per clip |
| `twos_complement_check.py --run <run> --clips <n>` | same clips with signed and two's complement `weights.txt`, compares raw outputs; restores the signed file (Section 7) | ~3 min for 20 clips |

```bash
uv run python scripts/fpga_debug/restart_check.py
```

`fpga_check.py`, `fpga_reconnect_check.py` and `reset_check.py` use the approaches we later found to be flawed (zero
frames between clips, or reconnecting). They're kept to show how we got there. For new measurements, use the
`concat_check.py` method (one restart, clips back-to-back, simulated the same way). It's the fastest clean comparison,
since this server version has no faster reset than a service restart.

---

## 7. Checks on the 80×80 avg-pool model (2026-11-22)

Model `motion-buddy-80-v4` ([training-log.md](training-log.md) #34): 80×80 input, avg-pool layers folded into
plain convs (k8 s4, k3, k6 s2, k6 s2, k2), Q1.7 weights with per-layer threshold scaling k = [2, 1, 1, 1, 2],
Q4.12 potentials. Firmware: riversdk.dev job #45 (KV260, GenX320, less resources). Board config: inputs 80×80,
`scaling: 256`, despot downsampling 4. Frames were sent as despot sends them (block sums), with a service restart
before every clip.

**Two's complement weights** (`twos_complement_check.py`, 10 val + 10 test clips). All 48,911 negative 8-bit
weights and biases written as v + 256 (−1 → 255, −110 → 146) instead of signed decimals:

| | signed | two's complement |
|---|---|---|
| raw outputs identical (every frame, every class) | 20/20 | 20/20 |
| prediction agrees with the fixed-point simulation | 19/20 | 19/20 |
| mean output spikes per clip | 148.2 | 148.2 |

spleat-server keeps only the low 8 bits of each weight and bias, so both encodings are the same to the hardware;
our export's signed decimals are fine.

**What the agreement with the simulation confirms** (19/20 clips, same prediction): the board runs the model the way
our fixed-point simulation does, including the parts that were only assumed so far:

- **per-layer thresholds** from `weights.txt`: thresholds differ by up to 2× between layers here. If SPLEAT
  ignored them, those layers would spike at a very different rate and the predictions would not agree;
- **avg-pool folding**: the firmware sees ordinary convs, and they compute what the pooled network computes;
- **input scaling** `4096 / downsampling²` with despot-style block sums.

20 clips are too few to compare accuracy (12/20 correct for both FPGA and simulation; the expected 72 % is inside
the uncertainty). For an accuracy estimate, run `fpga_accuracy.py --run runs/motion-buddy-80-v4` with a budget of
~30 min (all 1,042 val/test clips at ~4.5 s each would take ~80 min).

**Deploying a model temporarily.** We backed up the board's firmware, `weights.txt`, both configs and the command
lines of the running pipeline (`/root/backup-before-<model>-<date>/`, with `SHA256SUMS`), installed ours, tested,
then restored the backup and checked every file against the checksums. Two pitfalls: `pgrep -f`/`pkill -f` over SSH
also match the remote shell running the command (it contains the pattern) and kill your own session; and the live
gesture bridge sends commands to the Go2, so don't restart it unattended.
