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
| 17 | `e10-40-v3` | 11-22 00:04 | v3 | **10 epochs**, cosine over 7, average of last 3 | v3 set | = #14 | 72.3 / 55.1 | 69.0 / 55.9 |
| 18 | `e10-80-v3-s4` | 11-22 00:05 | v3 at **80×80**, first layer k7 s4 (80→19) | = #17 | v3 set, ±8 px | = #14 | 71.0 / 62.0 | 70.6 / 58.2 |
| 19 | `e10-80-v3-deep` | 11-22 00:07 | v3 at **80×80** + 16-ch k4 s2 layer in front (80→40) | = #17 | v3 set, ±8 px | = #14 | 67.6 / 59.8 | 67.4 / 59.8 |
| 20 | `motion-buddy-80-v3-s4` | 11-22 00:39 | v3 at **80×80**, first layer k7 s4 | = #14 (40 epochs) | v3 set, ±8 px | = #14 | 78.6 / 74.4 | 78.4 / 72.6 |
| 21 | `motion-buddy-80-v3-s4-adamw` | 11-22 00:42 | = #20 | = #20 with **AdamW**, weight decay 1e-2 on conv weights only | = #20 | = #14 | 80.1 / 64.7 | 78.6 / 67.7 |
| 22 | `motion-buddy-80-v3-s4-ce` | 11-22 00:48 | = #20 | = #20 with **loss ce+mse** (τ 10, MSE weight 1) | = #20 | = #14 | 77.6 / 69.5 | 76.8 / 69.4 |
| 23 | `optim-adam` | 11-22 04:26 | 80×80 s4 | **20 epochs**, cosine over 15, average of last 5, decay on conv weights only, Adam 1e-4 | v3 set, ±8 px | = #14 | 71.0 / 66.7 | 71.9 / 66.8 |
| 24 | `optim-adam-seed43` | 11-22 04:26 | = #23 | = #23, **seed 43** | = #23 | = #14 | 77.6 / 59.6 | 76.6 / 59.5 |
| 25 | `optim-adamw-1e-2` | 11-22 04:26 | = #23 | = #23 with AdamW 1e-2 | = #23 | = #14 | 73.1 / 67.4 | 73.9 / 65.8 |
| 26 | `optim-adamw-1e-1` | 11-22 04:26 | = #23 | = #23 with AdamW 1e-1 | = #23 | = #14 | 72.9 / 67.2 | 72.9 / 69.0 |
| 27 | `arch-head` | 11-22 07:18 | 80×80 s4 + **spatial head** (5×2×2 output, k1) | = #23 (BatchNorm recomputed) | = #23 | = #14 | 75.8 / 64.9 | 75.4 / 64.3 |
| 28 | `arch-head-seed43` | 11-22 07:18 | = #27, seed 43 | = #27 | = #23 | = #14 | 75.4 / 62.9 | 76.4 / 59.5 |
| 29 | `arch-pool` | 11-22 07:18 | 80×80 s4 with **avg-pool downsampling** (k5+pool4, k5+pool2 ×2) | = #27 | = #23 | = #14 | 69.6 / 71.5 | 68.8 / 68.6 |
| 30 | `arch-pool-s43` | 11-22 08:49 | = #29, seed 43 | = #27 | = #23 | = #14 | 76.4 / 66.5 | 76.6 / 69.2 |
| 31 | `arch-head-pool` | 11-22 07:19 | both | = #27 | = #23 | = #14 | 64.9 / 64.1 | 66.1 / 62.7 |
| 32 | `arch-head-pool-s43` | 11-22 08:49 | both, seed 43 | = #27 | = #23 | = #14 | 72.5 / 70.8 | 71.5 / 67.4 |
| 33 | `motion-buddy-40-v3-qat10` | 11-22 | = #14 | = #14 | = #14 | = #14 + **10 QAT epochs**, test logged per epoch | 71.5 / 76.8 | 77.4 / 71.0 |
| 34 | **`motion-buddy-80-v4`** (default recipe) | 11-22 10:28 | = #29 (80×80, avg-pool) | = #29 | #29's set + **speed, zoom/rotation, noise, hot pixels** | = #14 (Q1.7, k [2, 1, 1, 1, 2]) | 73.9 / 72.6 | 74.9 / 72.4 |

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
| #17 `e10-40-v3` | 70.2 | 67.7 | 68.1 | 43.7 | sit P14 19 %, no-action P14 25 % |
| #18 `e10-80-v3-s4` | 65.9 | 75.9 | 65.9 | 50.5 | dance (P13 42 %, P14 31 %), sit P14 35 %, hello P14 26 % |
| #19 `e10-80-v3-deep` | 68.6 | 65.9 | 65.9 | 53.8 | sit collapsed (P13 29 %, P14 0 %, P12 6 %) |
| #20 `motion-buddy-80-v3-s4` | 77.6 | 79.3 | 83.3 | 62.0 | sit P14 35 % (28 test sit clips → dance), hello P14 39 % |
| #21 `motion-buddy-80-v3-s4-adamw` | 76.5 | 81.0 | 78.3 | 57.3 | no-action (P13 57 %, P14 55 %; 50 test clips → sit), sit P14 31 % |
| #22 `motion-buddy-80-v3-s4-ce` | 75.7 | 78.0 | 81.2 | 57.7 | sit predicted too often (99 of 555 test clips wrongly → sit), hello P14 30 % |
| #23 `optim-adam` | 67.1 | 77.2 | 76.8 | 57.0 | sit P14 31 %, hello P14 26 %, dance 41 % |
| #24 `optim-adam-seed43` | 74.5 | 78.9 | 70.3 | 48.7 | sit P14 0 %, hello P14 18 % |
| #25 `optim-adamw-1e-2` | 69.4 | 78.9 | 74.3 | 57.3 | sit P14 37 %, hello 33–41 % |
| #26 `optim-adamw-1e-1` | 71.8 | 74.1 | 76.1 | 62.0 | sit P14 12 % |
| #27 `arch-head` | 72.9 | 78.0 | 72.5 | 56.3 | sit P14 8 % |
| #28 `arch-head-seed43` | 69.4 | 84.1 | 68.8 | 50.2 | sit P14 15 %, dance 39–41 % |
| #29 `arch-pool` | 67.8 | 69.8 | 77.9 | 59.5 | sit P14 17 %, hello P14 26 % |
| #30 `arch-pool-s43` | 71.0 | 82.8 | 74.6 | 63.8 | sit P14 29 %, dance P13 47 % |
| #31 `arch-head-pool` | 67.1 | 65.1 | 70.7 | 54.8 | sit P14 19 % |
| #32 `arch-head-pool-s43` | 68.2 | 75.0 | 75.4 | 59.5 | sit P14 4 % |
| #33 `motion-buddy-40-v3-qat10` | 77.6 | 77.2 | 81.5 | 60.6 | hello P14 16 %, sit P14 40 % |
| #34 `motion-buddy-80-v4` | 70.6 | 79.7 | 82.6 | 62.4 | hello P14 25 % (32 test hello clips → sit), dance P14 50 % |

`scripts/eval_per_person.py runs/<run>` prints this plus the per-class recall and confusion matrix. For #9 in float: P13 78 %, P14 62 %, dance recall 93 %, sit weak for P14.

## What we learned, by run

**#1 → #2: longer training helps but doesn't fix test.** 300 epochs gave +4 val and +4 test in float, but train accuracy went to ~99 %: the network overfits the 10 clean, centred training people. QAT on the 300-epoch model went NaN (fixed with `zero_tiny_gammas`, see [data-augmentation.md](data-augmentation.md#related-fix-nan-in-qat-after-long-training)).

**#4–#8: ablations (50 epochs).** Spatial augmentation helped test the most (+2.5 float over #2 at a sixth of the epochs). Temporal and camera augmentation showed no test gain: the test people differ in framing and clutter, not in timing or sensor. The 50-epoch runs underestimate augmentation, which needs more epochs.

**#9: augmentation is the big win.** +7 test points in float over #2 (70.3 vs 62.9), dance recall 32–44 % → 93 %.

**#10–#13: the 8-bit format matters more than QAT.** Layer 0's weights are ~8× larger than the rest, so one shared format either clips layer 0 (Q2.6: 50 % of its weights clipped) or wastes precision everywhere else (Q3.5). Per-layer scaling divides each layer's weights, bias and threshold by k = 2ⁿ, which gives the same spikes and lets the shared format be Q1.7. Biases must fit too: sizing on the weights alone (#12) clipped 9 biases and broke the model. Without QAT (#13) was the best: QAT validation swung between 43 % and 81 % from epoch to epoch.

**#14: the v3 recipe generalizes much better.** A teammate's v3 notebook: fast learnable leak (τ from 2), flip, polarity, mixup, dropout 0.2 in the middle layers, cosine lr, average of the last 10 epochs, and a network half the size. Test went up 7.4 points over #13 (75.3 vs 67.9) while val went down 11 points (71.9 vs 83.2): on val, dance is often predicted as lay-down (46 of 102 clips). 8-bit costs only 1.5 test points. Weight averaging scored lower on val than the best epoch (71.5 vs 75.8), but the average is used to avoid picking an epoch by val noise.

**#15: QAT trades test for val again.** 5 QAT epochs on #14 raised val 4.7 points (76.6 vs 71.9; dance on val 37 → 58 %) but lowered test 2.3 points (73.0 vs 75.3), all of it on person 14 (66.3 → 62.4; sit 60 → 42 %). QAT picks its best epoch by val, so it tunes towards the val people. Same outcome as #11 vs #13: for the test people, per-layer scaling without QAT stays the better export.

**#16: zoom/rotation on the v3 net: better val, slightly lower test.** Val +7.4 (79.3 vs 71.9), test −2.9 (72.4 vs 75.3), which is close to the noise level. It moves errors between classes: test hello 36 → 47 % (P14 23 → 33 %), test dance 71 → 54 % (now often predicted as sit). No 8-bit loss (float and fixed point both 72.4 % test). Train accuracy stayed at 65–71 % and val was still rising in the last epochs, so with zoom on, 40 epochs is probably too short. Next: the same config with more epochs.

**#17–#19: 80×80 input, 10-epoch comparison.** Same recipe, only the input size and the first layers differ, and all three are undertrained, so compare them only with each other. Test (fixed point) went 55.9 % at 40×40 → 58.2 % (k7 s4) and 59.8 % (deep). Person 14, the small off-centre subject, gained the most: 43.7 → 50.5 → 53.8 %. Val stayed about the same (69–71 %). The gain is consistent across both 80×80 variants but within 2–4 points of noise on test, so it needs the full 40-epoch runs. The deep variant had not learnt *sit* yet at epoch 10 (0 % for P14); it starts slower. k7 s4 lost 3.8 test points in 8-bit (62.0 float → 58.2 fixed), more than the others. With three runs sharing the GPU, an epoch took ~155 s (40×40 and k7 s4) and ~185 s (deep).

**#20: 80×80 at 40 epochs did not confirm the 10-epoch gain.** Compared with #14 (same recipe at 40×40), val went up 6.5 points (78.4 vs 71.9) and test went down 2.7 (72.6 vs 75.3). Person 14 went down too (62.0 vs 66.3), the opposite of the 10-epoch hint (#18). Hello on test improved (36 → 50 %), sit for person 14 got worse (60 → 35 %, mostly predicted as dance). The 8-bit loss was 1.8 points (74.4 float). With one seed per config and 2 test people, 40×40 and 80×80 are roughly equal; 80×80 costs ~4× the input events on the FPGA for no clear gain.

**#21: AdamW (wd 1e-2, conv weights only) is worse on test.** Same val as Adam (#20: 78.6 vs 78.4) but test 67.7 vs 72.6, float test 64.7 vs 74.4. The new error is no-action predicted as sit (50 of 120 test clips). It trained without NaN problems. Measured afterwards (gradients on 40 augmented training batches, at the trained weights): Adam's L2 decay of 1e-4 is divided by the small per-weight gradient, so on the conv weights it acts about as strongly as AdamW 1e-2 in layers 0–1, but 3–4× stronger in layers 2 and 4 and ~15× stronger in layer 3. So AdamW 1e-2 was the *weaker* regularizer, and layer 3's conv weights ended twice as large (median |w| 0.082 vs 0.037). On the leaks and BatchNorm the L2 term is under 1 % of the gradient, so excluding them from the decay changed little; the different τ values (layer 3: 4.1 vs 2.8) are a different training path, not a missing pull towards τ = 2. A fair comparison needs an AdamW decay that matches Adam's on layer 3 (~1e-1).

**#22: cross-entropy + MSE is worse on test than MSE alone.** Test 69.4 vs 72.6 (#20), val 76.8 vs 78.4. It learnt faster in the first 10 epochs (76.8 % val at epoch 9 vs 66.1 %) but ended lower, and it predicts *sit* too often: 99 test clips of other classes go to sit (dance 36, hello 34, no-action 29). 8-bit cost nothing (69.5 float). Not tried: a higher temperature or a smaller CE weight.

**Summary of #20–#22:** on the 80×80 s4 network, the notebook's Adam + MSE is the best of the three. None of these beat the 40×40 v3 (#14) on test.

Assumptions in #14 (the notebook isn't in the repo): dropout after layers 1–3, the augmentation probabilities, mixup α 0.4, cosine over the first 30 epochs.

**#23–#26: optimizer comparison, and the first measurement of run-to-run noise.** Two identical Adam runs that differ only in the random seed (#23, #24) end **4.7 points apart on val and 7.3 points apart on test** (66.8 vs 59.5 %); seed 43 never learnt *sit* for person 14 (0 %). Both AdamW runs (65.8 and 69.0 % test) fall inside that spread, so at 20 epochs there is **no measurable optimizer effect**. The bigger lesson: on 2 test people a single run is uncertain by several points, so the earlier single-seed comparisons that differ by 2–4 points (#14 vs #15, #16, #20, #22) are not conclusive either. Comparisons need 2–3 seeds per setting, or the mean over seeds.

**Weight averaging and BatchNorm (2026-11-22).** Every run with `average_last` up to here (#14–#22, and the four `optim-*` runs, which started before the change) averaged BatchNorm's stored running statistics along with the weights. That is only an approximation: each epoch's statistics belong to that epoch's weights. From the next runs on, the statistics are recomputed for the averaged weights with one pass over the training data (`train.recompute_bn`, as in SWA), and the log shows val accuracy with both.

Measured on #23–#26 with `scripts/recompute_bn.py` (same averaged weights, only the statistics recomputed, then quantized again; runs `optim-*-bn`). Fixed point val / test in %:

| Run | Averaged statistics | Recomputed | Test change |
|---|---|---|---|
| #23 Adam, seed 42 | 71.9 / 66.8 | 72.9 / 67.7 | +0.9 |
| #24 Adam, seed 43 | 76.6 / 59.5 | 74.7 / 60.5 | +1.0 |
| #25 AdamW 1e-2 | 73.9 / 65.8 | 75.4 / 65.9 | +0.1 |
| #26 AdamW 1e-1 | 72.9 / 69.0 | 72.5 / 70.1 | +1.1 |

Test improves in all four (+0.1 to +1.1), val moves ±2 points either way. Because the weights are identical, this is a paired comparison without the seed noise: a small but consistent gain, as expected with a learning rate at its floor over the averaged epochs.

Applied to the deployment candidate #14 (40×40 v3, average of 10 epochs; copy `motion-buddy-40-v3-bn`): fixed point 71.9 / 75.3 → **69.2 / 75.1**, float 71.5 / 76.8 → 71.7 / 75.1. No gain here (test −0.2, val −2.7), so across five runs the effect is about ±1 point on test: the correct method, but not a measurable accuracy change. #14 keeps its original export.

**#27–#32: spatial head and avg-pool downsampling** ([architecture-options.md](architecture-options.md)), two seeds each, against the same recipe without them (#23/#24 with recomputed BatchNorm, `optim-adam*-bn`). Fixed point, test in %:

| Setting | seed 42 | seed 43 | mean | person 14 (mean) | val (mean) |
|---|---|---|---|---|---|
| baseline (#23/#24) | 67.7 | 60.5 | 64.1 | 52.7 | 73.8 |
| spatial head (#27/#28) | 64.3 | 59.5 | 61.9 | 53.3 | 75.9 |
| **avg-pool (#29/#30)** | 68.6 | 69.2 | **68.9** | **61.7** | 72.7 |
| both (#31/#32) | 62.7 | 67.4 | 65.1 | 57.2 | 68.8 |

- **Avg-pool downsampling helps:** both seeds beat both baseline seeds on test (+4.8 points on the mean, +9 on person 14), and the two seeds are 0.6 points apart instead of 7.2. It's the only change so far whose gain is larger than the seed spread. Val is about the same. It costs nothing on the FPGA (same layer count, the first conv is k8 instead of k7: +480 weights) but needs ~19 GB GPU memory per 80×80 run instead of ~11 GB (the first conv runs at stride 1 on 80×80), so only 3–4 runs fit on the A100 at once.
- **The spatial head doesn't:** val goes up 2 points, test down 2. Summing over the 2×2 positions also sums the distractor's activity into the scores, which may cancel the benefit of position-independent weights. Combined with pooling it is worse than pooling alone.
- *Sit* for person 14 stays the weak spot in all of them (4–29 %).

**#33: 10 QAT epochs on #14, with test logged after every epoch.** Validation chose epoch 5, and the fixed-point result is **77.4 / 71.0** (#14 without QAT: 71.9 / 75.3). Per epoch (QAT model, val / test in %):

| QAT epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 72.9 | 71.0 | 76.4 | 73.5 | **77.4** | 73.9 | 76.0 | 74.7 | 73.3 | 76.2 |
| test | 76.2 | 76.6 | 71.7 | 72.3 | 72.3 | 72.1 | 72.6 | 74.1 | 75.3 | 71.7 |

The first two QAT epochs lift test above #14 (76.2, 76.6), but from epoch 3 on val and test move in opposite directions: QAT fine-tunes towards the validation people, and selecting the epoch on val makes that worse (the chosen epoch is 4 points below #14 on test, mostly person 14: 66.3 → 60.6). This is why QAT kept losing on test (#11, #15, #33). If QAT is used, it should be short and not selected on val (e.g. a fixed 1–2 epochs, or the average of the QAT epochs); picking the epoch on test is not an option, as test would then no longer be an unbiased estimate.

**#34: the default recipe `motion_buddy_80_v4` (80×80, avg-pool, 20 epochs) with speed, zoom/rotation, noise and hot pixels added.** Fixed point 74.9 / **72.4** %, against 68.8 / 68.6 % for the same seed without the four augmentations (#29): +3.8 test, +6.1 val, person 14 59.5 → 62.4 %. One seed, so the gain is suggestive, not proven. The first 80×80 model with an automatic, matching firmware: job **#45** (built from the config's model.json before training ended, Q1.7 / Q4.12, KV260, GenX320, less resources; the run's exported model.json is byte-identical). Board: despot downsampling 4, inputs 80×80, `scaling: 256`, 5×1×1 outputs. **Tested on the board** (20 clips, restart per clip): FPGA and fixed-point simulation agree on 19/20 predictions; signed and two's complement `weights.txt` give bit-identical outputs ([fpga-debugging.md](fpga-debugging.md#7-checks-on-the-8080-avg-pool-model-2026-11-22)).

## Open questions and next runs

- 80×80 at 40 epochs: `motion_buddy_80_v3_s4.yaml` and `motion_buddy_80_v3_deep.yaml` (#18/#19 looked promising for person 14 after 10 epochs). Before deploying, check that SPLEAT keeps up at 100 Hz with ~4× more active input pixels.
- v3 + zoom/rotation with more epochs (#16 was still improving at epoch 40).
- One seed per config so far: a second seed of #14 and #16 would show how much of the 3-point test difference is noise.
- The 88 % test figure we were aiming for: probably the notebook's split (train 01–03, val 04, test 05). Not comparable to our split until we run that split.
- Training on persons 01–12 (no separate val) for the final model.
- The FPGA never resets potentials between clips: check accuracy on a sliding window without reset.
- On the board so far only #34 (`motion-buddy-80-v4`, firmware #45), and only on 20 clips: FPGA = simulation on 19/20, which also confirms per-layer thresholds, avg-pool folding and the input scaling ([fpga-debugging.md](fpga-debugging.md#7-checks-on-the-8080-avg-pool-model-2026-11-22)). Still missing: a full FPGA accuracy run (`fpga_accuracy.py`, ~30 min), and firmware for #14 (40×40 v3) if it should be compared on the board.
