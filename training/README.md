# Five-class gesture SNN

The classifier predicts `sit`, `lay-down`, `hello`, `dance`, or `no-action` from the local Durance event archive. It uses three Conv–LIF layers and a linear readout over the early and late spike activity. PyTorch fake-quantizes weights, inputs, and membrane updates from the first training batch. This is a PyTorch research checkpoint. RiverOps export and RiverFPGA inference require validation with the Durance SDK.

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

Run one prediction directly from the ZIP. Pass `--checkpoint` when using a different run directory.

```sh
python -m training.predict motion_buddy_dataset/13/gesture_04-dance-normal-01.bin
```

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
