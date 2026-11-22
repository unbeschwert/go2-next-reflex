# REACT-style gesture recognition

This PyTorch baseline reads Motion Buddy recordings directly. It maps the camera coordinates and polarity to an embedding, updates two residual C-SiLIF blocks using the elapsed time between selected events, and classifies the running mean. The C-SiLIF state persists after a spike. The prototype uses a 40×40 coordinate grid and 32 channels by default. It uses real-valued residual features and no inter-block pooling. The recurrence follows the [REACT preprint](https://arxiv.org/html/2609.19204v1#S3.SS2).

Each person belongs to a single split. The default config trains on people 01–10, picks a checkpoint by macro F1 on 11–12, and reports test scores on 13–14. This is the same split used by the 40×40 spiking CNN.

## Run it

Use a Python environment with PyTorch, NumPy, and PyYAML. The project training environment provides these dependencies.

```bash
python -m training.react --run-dir runs/react-motion-buddy --device cuda --workers 4
```

The run saves its settings, the best checkpoint, accuracy, macro F1, and confusion matrices. Use a new run directory for each experiment. The default cap of 2,048 events per clip samples across the entire clip, retaining the order and physical timestamp gaps. It is suitable for an offline comparison. Live capture needs a causal rate limit or event filter that is also used for training.

For a short local check, use `--device cpu --epochs 1 --limit-per-split 20 --max-events 256`. To check equivalence of the batched and streaming paths, run `python -m unittest training.react.test_model`.

## Compare spatial augmentation

`--spatial-augment` translates training clips by up to four grid cells on each axis and scales them by 0.8–1.2 about the sensor center. It discards events that move outside the image and preserves event times for the remaining events. The cap on sampled events applies before the transform.

```bash
python -m training.react --run-dir runs/react-spatial-09-10 --device cuda --workers 2 \
    --val-subjects 09 10 --spatial-augment --skip-test --check-position-shift
```

Use the same options without `--spatial-augment` for the control. The validation override uses people from the configured training and validation pool. The resulting metrics include per-person accuracy, macro F1, and class recall. The position check evaluates the selected checkpoint on validation clips moved four grid cells left and right.

## What the Kria port needs

1. **A trained baseline and continuous replay.** Compare held-out subjects with the CNN. Replay longer recordings with background motion and gesture transitions to measure false triggers per minute and gesture-onset-to-decision latency. The prototype's running mean starts at the first event. Continuous operation needs a readout that can forget older gestures.
2. **A raw-event feed and a throughput target.** Route timestamped `x, y, polarity` events from the GenX320. Measure peak event rates over short windows and choose a causal sampling or filtering policy when the feed exceeds the accelerator's capacity.
3. **A fixed-point reference.** Quantize the embedding, linear weights, complex state and transition coefficients. Fold the embedding normalization into the lookup table. Compare fixed-point state, spikes and decisions against the PyTorch `step()` path on recording replays.
4. **Custom FPGA logic.** Implement the event queue, embedding lookup, two-component state updates and readout using Vivado or Vitis HLS. Synthesize alongside the camera interface and verify resource use, timing, sustained throughput and event loss on the KV260. [AMD custom hardware workflow](https://edf.docs.amd.com/en/latest/hwdev/custom-hardware-development.html)
