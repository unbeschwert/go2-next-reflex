# Five-class gesture SNN

The classifier predicts `sit`, `lay-down`, `hello`, `dance`, or `no-action` from the local Durance event archive. It uses three Conv–LIF layers and a linear readout over the early and late spike activity. The default PyTorch run fake-quantizes weights, inputs, and membrane updates from the first training batch. The FP32 option keeps the spiking neurons and uses full-precision arithmetic to study recognition before quantization. RiverOps export and RiverFPGA inference require validation with the Durance SDK.

See [recognition results](RESULTS.md) for the held-out comparisons, the original confusion matrix, and the command rejection tests.

## Reproduce the run

Use Python with NumPy and [PyTorch](https://pytorch.org/get-started/locally/) installed. The CPU run uses Python 3.13, NumPy 2.3.5, and PyTorch 2.12. Run these commands from the repository root with the dataset ZIP in that directory.

```sh
python -m training.data
python -u -m training.train --epochs 16 --threads 6 --batch-size 32
```

The first command builds a local cache from the archive. It counts each event polarity separately on a `32 × 32` grid for each of ten 100,000-tick bins, relative to the first event of a clip. The cache and training outputs are stored under the ignored `.local/training/` directory. The trainer accepts `--device cuda` on a compatible PyTorch installation.

| Split | Recording groups | Clips |
| --- | --- | ---: |
| Train | `01–10` | 2,691 |
| Validation | `11–12` | 487 |
| Test | `13–14` | 555 |

Training augments spatial position and flips clips horizontally. It saves the checkpoint with the highest validation macro-F1 as `gesture-snn.pt` and evaluates it on the held-out groups. `metrics.json` contains epoch history, test metrics, per-class recall, and a confusion matrix with true classes in rows and predicted classes in columns. The grouping uses archive folder IDs. Their relationship to individual participants is unverified.

To try the optional spatial normalization, supply `--center-events --output .local/training/centered` to the training command. This re-centers and scales the main event activity in each clip using the same transform for all ten bins.

To test variation in gesture speed and event density, add `--event-augmentation --output .local/training/event-augmentation`. This changes training batches only. Validation, test, and inference use the original event tensors, so the new run is comparable to the baseline on the same recording groups.

On the 555 held-out recordings, the baseline reached 41.6% accuracy and 0.363 macro-F1. The speed and event-density augmentation trial reached 36.2% and 0.314. The baseline also scored higher on validation macro-F1, 0.472 compared with 0.458. These are local PyTorch results.

### Spatial augmentation

The `crop` option adds a small crop, zoom, or pad. The `affine` option also adds up to eight degrees of rotation and a small horizontal shear. Each sample uses one transform across all time bins and both event polarities. These options apply only during training. The four-command model keeps horizontal flips disabled because they would reverse `turn_left` and `turn_right`.

```sh
python -m training.train --epochs 16 --threads 6 --batch-size 32 --spatial-augmentation crop --output .local/training/spatial-crop
python -m training.train --epochs 16 --threads 6 --batch-size 32 --spatial-augmentation affine --output .local/training/spatial-affine
```

With seed 17 and the same 555 test recordings, crop and zoom reached 30.3% accuracy and 0.273 macro-F1. Adding rotation and shear reached 38.9% and 0.312. The baseline scored higher on validation macro-F1, so spatial augmentation remains opt-in.

### Four-class comparison without `dance`

Both runs use `sit`, `lay-down`, `hello`, `no-action` in that order. `drop` omits dance clips from training. `background` keeps the training dance clips and labels them `no-action`. Both select checkpoints on the same 385 non-dance validation clips and report scores on the same 442 non-dance test clips. The 113 held-out dance clips can be replayed separately to check whether the model selects another gesture in response.

```sh
python -m training.train --epochs 16 --threads 6 --batch-size 32 --dance-policy drop --output .local/training/dance-drop
python -m training.train --epochs 16 --threads 6 --batch-size 32 --dance-policy background --output .local/training/dance-background
```

The original checkpoint reached 52.0% accuracy on the same 442 clips. Masking the `dance` score reached 52.3%, dropping the recordings reached 43.7%, and training them as `no-action` reached 49.8%. The background run returned `no-action` on 40 of 113 held-out dance recordings, compared with 3 for the drop run. These are local replay results across the original recording split.

Run one prediction directly from the ZIP. Pass `--checkpoint` when using a different run directory.

```sh
python -m training.predict motion_buddy_dataset/13/gesture_04-dance-normal-01.bin
```

## Compare model widths before quantization

Both runs keep the same three Conv–LIF layers, event cache, recording-group split, and readout. The compact model uses `8 → 16 → 24` channels (8,645 trainable parameters). The wide model uses `16 → 32 → 48` channels (26,501 parameters). Both train without fake quantization for this comparison.

```sh
python -m training.train --epochs 16 --threads 6 --batch-size 32 --precision fp32 --model-width compact --output .local/training/fp32-compact
python -m training.train --epochs 16 --threads 6 --batch-size 32 --precision fp32 --model-width wide --output .local/training/fp32-wide
```

In this matched run, the compact model reached 0.487 validation macro-F1 and 0.372 test macro-F1. The wide model reached 0.467 and 0.358. On the 555 test clips, widening raised accuracy from 39.8% to 41.8% and reduced false commands on `no-action` clips from 35 to 19. It also reduced correct `hello` predictions from 61 to 51 out of 106.

## Compare ternary event inputs

The baseline keeps the OFF and ON counts in two channels per time bin. The ternary run feeds one channel with `sign(ON - OFF)`, so each bin and cell contains `-1`, `0`, or `+1`. Zero includes empty cells and cells where equal counts cancel. The experiment uses the same time bins, raw event cache, recording groups, and FP32 training schedule as the compact comparison above.

```sh
python -m training.train --epochs 16 --threads 6 --batch-size 32 --precision fp32 --model-width compact --event-encoding ternary --output .local/training/fp32-ternary
```

## Compare temporal resolution

The 50 ms experiment keeps the two polarity count channels, spatial resolution, one-second clip, and compact FP32 model. It doubles the number of bins from ten to twenty. The membrane decay per step scales with bin duration so that it retains the same decay over a fixed amount of time.

```sh
python -m training.data --bin-us 50000 --output .local/training/gestures-50ms.npz
python -m training.train --epochs 16 --threads 6 --batch-size 32 --precision fp32 --model-width compact --cache .local/training/gestures-50ms.npz --output .local/training/fp32-50ms
```

On the same 555 test clips, 50 ms bins reached 37.7% accuracy and 0.315 macro-F1. The 100 ms FP32 baseline reached 39.8% and 0.372. Correct `hello` detections fell from 61 to 53 out of 106, and false commands on background fell from 35 to 31 out of 120. The checkpoints were selected on validation macro-F1.

## Export INT8 weights

The portable NPZ contains signed INT8 convolution and classifier weights, per-output-channel scales, FP32 biases, and the model's event and neuron settings. PyTorch replay dequantizes the weights when loading the NPZ and uses the same quantized neuron simulation. RiverOps board export is a separate SDK step.

```sh
python -m training.export
python -m training.predict motion_buddy_dataset/13/gesture_04-dance-normal-01.bin --weights .local/training/run/gesture-snn-int8.npz
```

## Movement command pilot

The movement SNN uses this fixed output order. Label directions as they appear in the unmirrored camera frames and check the mapping to robot left and right if the camera orientation changes.

| Output | Command | Record this cue |
| ---: | --- | --- |
| 0 | `turn_left` | One deliberate sweep toward image left |
| 1 | `turn_right` | One deliberate sweep toward image right |
| 2 | `stop` | A distinct outward palm sweep followed by a short hold |
| 3 | `no_action` | Ordinary arm movement without a command |

The replay outputs one command ID per window. A tied score, or a lead no greater than `--minimum-margin`, returns `no_action`. Keep the output order the same when passing FPGA spike counts to the decoder.

```sh
python -m training.command_replay decode 3 8 1 0
```

### Run the local smoke test

The synthetic event patterns exercise training, the portable INT8 export, and decoding. The local replay decoded all 80 generated test clips as expected. Real-camera command accuracy still needs labeled recordings.

```sh
python -m training.command_data --synthetic --output .local/training/commands-synthetic.npz
python -m training.train --cache .local/training/commands-synthetic.npz --output .local/training/commands-demo --epochs 12 --threads 6 --batch-size 24
python -m training.export --checkpoint .local/training/commands-demo/gesture-snn.pt --output .local/training/commands-demo/commands-int8.npz
python -m training.command_replay replay --cache .local/training/commands-synthetic.npz --weights .local/training/commands-demo/commands-int8.npz
```

### Train and export for RiverSDK

Collect labeled event-camera clips in the same binary recording format as the Durance archive. Group all clips from a person or recording session in the same split. Include all four cues in each split and put the files in the ZIP under `movement_commands/<split>/<group>/<command>/<file>.bin`. Use `train`, `validation`, and `test` for split names. Directional samples use positional jitter with horizontal flipping disabled.

Run the following with the Durance RiverOps SDK installed, using the SDK's supported Python and PyTorch versions. This path configures Q1.7 weight and Q4.12 neuron-potential quantization before the first optimizer step. Actual SDK export and board replay are pending verification.

```sh
python -m training.command_data --archive movement_commands.zip --output .local/training/commands.npz
python -m training.train --backend riverops --cache .local/training/commands.npz --output .local/training/commands-riverops
python -m training.riverops_commands --checkpoint .local/training/commands-riverops/gesture-snn.pt --cache .local/training/commands.npz --output .local/training/commands-riverops
```

Upload `movement_model.json` to RiverSDK to build the firmware. The board loads the matching `weights.txt`. Reproduce the training input when replaying events on the FPGA: two polarity channels on a `32 × 32` grid, ten 100,000-tick bins, and counts capped at 63 and divided by the source pixel area per cell. From the live camera this requires binning compatible with that grid and timing. Verify the four output counts on held-out recordings against RiverOps before using them as motion requests. [Durance model export guide](https://riversdk.dev/docs/hackathon/train_handgesture_recognition.html#Export-the-model-architecture)
