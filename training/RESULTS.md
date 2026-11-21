# Gesture recognition results

These are local PyTorch experiments on the Motion Buddy archive. Recording groups `01–10` were used for training (2,691 clips), `11–12` for validation (487), and `13–14` for testing (555). The SNN runs below use seed 17 and select a checkpoint by validation macro-F1. The test groups were reused across experiments, so confirm any improvement on newly held-out recordings before deployment. [Reproduction commands](README.md)

## Five-class comparison

`Hello correct` counts correct greetings out of 106. `Background false commands` counts `no-action` clips assigned a gesture out of 120. A clip prediction is one model decision. A live command rate requires continuous replay with the actual command policy.

| Model or input change | Validation macro-F1 | Test accuracy | Test macro-F1 | Hello correct | Background false commands |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact QAT baseline | 0.472 | 41.6% | 0.363 | 64 | 22 |
| QAT with clip centering | 0.406 | 31.2% | 0.237 | 49 | 24 |
| QAT with speed and density augmentation | 0.458 | 36.2% | 0.314 | 32 | 26 |
| QAT with mild crop and zoom | 0.459 | 30.3% | 0.273 | 48 | 76 |
| QAT with crop, rotation, and shear | 0.462 | 38.9% | 0.312 | 36 | 19 |
| Compact FP32, 100 ms bins | 0.487 | 39.8% | 0.372 | 61 | 35 |
| Wide FP32, 100 ms bins | 0.467 | 41.8% | 0.358 | 51 | 19 |
| Compact FP32, ternary input | 0.295 | 32.1% | 0.277 | 52 | 39 |
| Compact FP32, 50 ms bins | 0.464 | 37.7% | 0.315 | 53 | 31 |

The 50 ms run doubles temporal resolution while keeping the spatial grid, clip duration, and parameter count fixed. The ternary experiment computes `sign(ON - OFF)` per bin. At 100 ms, equal counts canceled in about 13% of active grid cells. The two-channel input retains those events.

## Compact QAT confusion matrix

Rows are actual labels. Columns are predicted labels.

| Actual | Sit | Lay down | Hello | Dance | No action |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sit | **13** | 49 | 7 | 1 | 33 |
| Lay down | 19 | **55** | 17 | 2 | 20 |
| Hello | 0 | 32 | **64** | 1 | 9 |
| Dance | 4 | 93 | 7 | **1** | 8 |
| No action | 2 | 16 | 4 | 0 | **98** |

A global score margin chosen on validation reduced background false commands from 22 to 6 on test, while correct greetings fell from 64 to 33. A margin applied only to `lay-down` reduced background false commands to 16, kept 64 correct greetings, and reduced correct `lay-down` predictions from 55 to 48.

## Removing dance

The accuracy column uses the *same 442 non-dance clips* for each model. The 113 dance clips are a separate rejection check.

| Approach | Accuracy on non-dance clips | `No action` on dance clips |
| --- | ---: | ---: |
| Original five-class checkpoint | 52.0% | 8/113 |
| Original checkpoint with dance output masked | 52.3% | 9/113 |
| Retrain without dance clips | 43.7% | 3/113 |
| Retrain with dance as `no-action` | 49.8% | 40/113 |

## Priorities for the next dataset

Collect paired `hello` and `lay-down` gestures from the same people at comparable distances and viewpoints. Include ordinary movement that resembles those cues and record continuous event streams with gesture start and end times. Keep new recording groups locked for the final comparison. Measure per-command recall and precision at a matched background false-command rate.

A diagnostic RBF SVM using pooled polarity counts and motion centroids scored 0.566 validation macro-F1 and 0.247 test macro-F1. This is another signal to check generalization across people and recording conditions before drawing conclusions from validation alone.

The portable INT8 export of the QAT baseline reproduced its PyTorch logits on all 555 test clips. RiverOps export and FPGA replay remain pending. The four-command demo has a separate result of 80/80 on generated inputs, which verifies the command pipeline on synthetic events.
