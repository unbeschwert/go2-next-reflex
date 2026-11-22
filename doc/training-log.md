# Training log

What we have trained, with which settings, and what came out. Add a row (and a section if there's something to learn from it) for every run worth keeping, including the failures.

## How to read the numbers

- **Splits are by person:** train 01–10, val 11–12, test 13–14. Test persons are recorded differently from everyone else: person 13 has several people in the frame, and person 14 is small at the left edge with a large moving distractor on the right. Test accuracy is mostly about this shift, and val and test can disagree.
- **Accuracy** = one prediction per 1 s clip: sum the output spikes over the 100 steps, take the argmax (ties go to the lowest index, which is *sit*). Val is 487 clips and test is 555.
- **Fixed point** = the 8-bit model as the FPGA runs it. That's the number that counts; float is the upper bound.
- **With 2 test people, ±2–3 points is noise.**
- All runs are on the GPU server under `runs/<run>/`. The configs are in `training/configs/`, and each run also keeps the config it used in `runs/<run>/config.yaml`.

## Networks

| Name | Layers (out, kernel, stride, padding) | Sizes | Neuron | Parameters |
|---|---|---|---|---|
| **v1 net** | (16,5,2,1) (16,3,1,1) (8,6,2,2) (256,9,1,0) (5,1,1,0) | 40→19→19→9→1→1 | PLIF, fixed τ = 14.3 | 175,191 |
| **v3 net** | (16,5,2,1) (16,3,1,1) (16,6,2,2) (128,6,2,0) (5,2,1,0) | 40→19→19→9→2→1 | PLIF, learnable τ from 2 | 88,799 |

Both: 2 polarity channels at 40×40 (downsampling 8), 100 steps of 10 ms, "mean" encoding, hard reset, threshold 1, MSE loss to 80 % / 20 % of T spikes. The v3 net has a different layer shape, so it needs its own firmware build.

## All runs

Val / test accuracy in %. Unless noted: batch 32, constant lr 3e-3, weight decay 1e-4, ±2 px shift.

| # | Run | Date | Net | Training | Augmentation | 8-bit export | Float | Fixed point |
|---|---|---|---|---|---|---|---|---|
| 1 | `motion-buddy-40-v1` | 11-21 15:22 | v1 | 30 epochs | none | Q2.6 + 10 QAT | 73.3 / 57.1 | 75.2 / 56.0 |
| 2 | `motion-buddy-40-e300` | 11-21 17:52 | v1 | 300 epochs | none | Q2.6 + 10 QAT | 77.6 / 62.9 | 78.0 / 60.0 |
| 3 | `motion-buddy-40-e300-ls` | 11-21 | v1 | = #2 | none | first per-layer try (layer 0 only, k 4), Q2.6 + QAT | 77.6 / 62.9 | 75.2 / 55.3 |
| 4 | `abl-spatial` | 11-21 19:29 | v1 | 50 epochs | ±4 px, zoom + rotation | Q2.6 + 10 QAT | 78.6 / 65.4 | 72.5 / 60.7 |
| 5 | `abl-temporal` | 11-21 19:29 | v1 | 50 epochs | speed, time shift | Q3.5 + 10 QAT | 72.9 / 53.9 | 74.9 / 57.3 |
| 6 | `abl-camera` | 11-21 19:29 | v1 | 50 epochs | dropout, rate, noise, hot pixels | Q3.5 + 10 QAT | 72.5 / 57.3 | 70.6 / 59.3 |
| 7 | `abl-warm-start` | 11-21 19:29 | v1 | 50 epochs | warm start | Q2.6 + 10 QAT | 78.0 / 54.2 | 79.1 / 57.7 |
| 8 | `abl-extras` | 11-21 19:29 | v1 | 50 epochs | full set + flip, cutout, polarity | Q3.5 + 10 QAT | 74.9 / 59.6 | 74.9 / 62.9 |
| 9 | `motion-buddy-40-aug` | 11-21 19:25 | v1 | 300 epochs | full set (below) | Q3.5 + 10 QAT | 84.0 / 70.3 | 82.1 / 66.3 |
| 10 | `motion-buddy-40-aug-q26` | 11-21 | v1 | = #9 | = #9 | Q2.6 forced + 10 QAT | 84.0 / 70.3 | 80.1 / 62.9 |
| 11 | `motion-buddy-40-aug-lsA` | 11-21 | v1 | = #9 | = #9 | per-layer scaling, Q1.7 + 10 QAT | 84.0 / 70.3 | 82.8 / 66.3 |
| 12 | `motion-buddy-40-aug-lsB` | 11-21 | v1 | = #9 | = #9 | per-layer, sized on weights only (biases clip) + 10 QAT | 84.0 / 70.3 | 55.6 / 36.8 |
| 13 | `motion-buddy-40-aug-lsA0` | 11-21 | v1 | = #9 | = #9 | per-layer scaling, Q1.7, **no QAT** | 84.0 / 70.3 | 83.2 / 67.9 |
| 14 | **`motion-buddy-40-v3`** | 11-21 22:38 | **v3** | 40 epochs, batch 16, cosine lr, average of last 10, dropout 0.2 | v3 set (below) | per-layer scaling, Q1.7, no QAT | 71.5 / 76.8 | **71.9 / 75.3** |
| 15 | `motion-buddy-40-v3-qat` | 11-21 23:07 | v3 | = #14 | = #14 | = #14 + 5 QAT epochs (lr 1e-4) | 71.5 / 76.8 | 76.6 / 73.0 |
| 16 | `motion-buddy-40-v3-affine` | 11-21 23:07 | v3 | = #14 | v3 set + zoom 0.8–1.2 / rotation ±10° (p 0.5) | = #14 | 79.1 / 72.4 | 79.3 / 72.4 |

Rows 3, 10–13 and 15 re-quantize an existing float model with `remote/qat.sh`, so their float numbers are those of the run they start from.

**Best 8-bit model so far: #14 `motion-buddy-40-v3`, 75.3 % test.** Best on val: #13 `aug-lsA0`, 83.2 %.

### Augmentation sets

| | Full set (#9) | v3 set (#14) |
|---|---|---|
| Integer shift | ±4 px | ±4 px |
| Event dropout | p 0.5, up to 30 % | p 0.5, up to 40 % |
| Speed | p 0.5, ×0.8–1.25 | off |
| Time shift | p 0.5, ±20 frames | p 0.5, ±20 frames |
| Zoom + rotation | p 0.5, 0.8–1.2, ±10° | off (#16 turns it on) |
| Horizontal flip | off | p 0.5 |
| Rate | p 0.5, ×0.7–1.3 | p 0.5, ×0.7–1.4 |
| Polarity swap | off | p 0.3 |
| Noise, hot pixels | on | off |
| Warm start | on | p 0.5, 10–50 frames |
| Mixup | – | p 0.5, α 0.4 |

What each augmentation does: [data-augmentation.md](data-augmentation.md).

## Per-person results (fixed point)

| Run | P11 (val) | P12 (val) | P13 (test) | P14 (test) | Weakest classes on test |
|---|---|---|---|---|---|
| #14 `motion-buddy-40-v3` | 71.0 | 72.8 | 84.4 | 66.3 | hello (P13 51 %, P14 23 %), sit P14 60 % |
| #15 `motion-buddy-40-v3-qat` | 75.3 | 78.0 | 83.7 | 62.4 | hello (P13 43 %, P14 23 %), sit P14 42 %, lay-down P14 66 % |
| #16 `motion-buddy-40-v3-affine` | 77.3 | 81.5 | 80.1 | 64.9 | hello (P13 63 %, P14 33 %), dance (P13 53 %, P14 56 %), sit P14 58 % |

`scripts/eval_per_person.py runs/<run>` prints this plus the per-class recall and confusion matrix. For #9 in float: P13 78 %, P14 62 %, dance recall 93 %, sit weak for P14.

## What we learned, by run

**#1 → #2: longer training helps but doesn't fix test.** 300 epochs gave +4 val and +4 test in float, but train accuracy went to ~99 %: the network overfits the 10 clean, centred training people. QAT on the 300-epoch model went NaN (fixed with `zero_tiny_gammas`, see [data-augmentation.md](data-augmentation.md#related-fix-nan-in-qat-after-long-training)).

**#4–#8: ablations (50 epochs).** Spatial augmentation helped test the most (+2.5 float over #2 at a sixth of the epochs). Temporal and camera augmentation showed no test gain: the test people differ in framing and clutter, not in timing or sensor. The 50-epoch runs underestimate augmentation, which needs more epochs.

**#9: augmentation is the big win.** +7 test points in float over #2 (70.3 vs 62.9), dance recall 32–44 % → 93 %.

**#10–#13: the 8-bit format matters more than QAT.** Layer 0's weights are ~8× larger than the rest, so one shared format either clips layer 0 (Q2.6: 50 % of its weights clipped) or wastes precision everywhere else (Q3.5). Per-layer scaling divides each layer's weights, bias and threshold by k = 2ⁿ, which gives the same spikes and lets the shared format be Q1.7. Biases must fit too: sizing on the weights alone (#12) clipped 9 biases and broke the model. Without QAT (#13) was the best: QAT validation swung between 43 % and 81 % from epoch to epoch.

**#14: the v3 recipe generalizes much better.** A teammate's v3 notebook: fast learnable leak (τ from 2), flip, polarity, mixup, dropout 0.2 in the middle layers, cosine lr, average of the last 10 epochs, and a network half the size. Test went up 7.4 points over #13 (75.3 vs 67.9) while val went down 11 points (71.9 vs 83.2): on val, dance is often predicted as lay-down (46 of 102 clips). 8-bit costs only 1.5 test points. Weight averaging scored lower on val than the best epoch (71.5 vs 75.8), but the average is used to avoid picking an epoch by val noise.

**#15: QAT trades test for val again.** 5 QAT epochs on #14 raised val 4.7 points (76.6 vs 71.9; dance on val 37 → 58 %) but lowered test 2.3 points (73.0 vs 75.3), all of it on person 14 (66.3 → 62.4; sit 60 → 42 %). QAT picks its best epoch by val, so it tunes towards the val people. Same outcome as #11 vs #13: for the test people, per-layer scaling without QAT stays the better export.

**#16: zoom/rotation on the v3 net: better val, slightly lower test.** Val +7.4 (79.3 vs 71.9), test −2.9 (72.4 vs 75.3), which is close to the noise level. It moves errors between classes: test hello 36 → 47 % (P14 23 → 33 %), test dance 71 → 54 % (now often predicted as sit). No 8-bit loss (float and fixed point both 72.4 % test). Train accuracy stayed at 65–71 % and val was still rising in the last epochs, so with zoom on, 40 epochs is probably too short. Next: the same config with more epochs.

Assumptions in #14 (the notebook isn't in the repo): dropout after layers 1–3, the augmentation probabilities, mixup α 0.4, cosine over the first 30 epochs.

## Open questions and next runs

- v3 + zoom/rotation with more epochs (#16 was still improving at epoch 40).
- One seed per config so far: a second seed of #14 and #16 would show how much of the 3-point test difference is noise.
- The 88 % test figure we were aiming for: probably the notebook's split (train 01–03, val 04, test 05). Not comparable to our split until we run that split.
- Training on persons 01–12 (no separate val) for the final model.
- The FPGA never resets potentials between clips: check accuracy on a sliding window without reset.
- Not yet on the board: v3 needs a firmware build (Q1.7, new layer sizes), and `scripts/fpga_debug/threshold_fpga_check.py` must confirm SPLEAT uses the per-layer thresholds.
