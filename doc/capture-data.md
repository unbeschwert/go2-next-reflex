# Recording event data with `river-tools`

Condensed from the RIVER SDK cheatsheet "Capture data", plus the source of `river_tools-0.1.0`.

## Install

You need Python ≥ 3.10, `pipx` and SSH.

```bash
pipx install ./data/river_tools-0.1.0-py3-none-any.whl
```

## Record

1. Connect the camera to the Kria.
2. Connect the Kria to your laptop with Ethernet and power it on.
3. Wait for it to boot.

Then, from the folder where you want recordings saved:

```bash
river-tools capture --host root@durance-kria.local
```

- This SSHes into the Kria, starts `despot_debug_camera` there (streaming to your laptop), and opens a live view.
- **Start recording** / **Stop recording** make one sample; a red circle shows while recording.
- Samples are saved as `samples/sample_YYYYMMDD_HHMMSS_ffffff.bin`.
- Closing the window saves any recording in progress and stops the camera on the Kria.

## Play back

```bash
river-tools play samples/<file>.bin      # Space = pause/resume, loops until closed
```

You can also use our viewer, which shows the clip as a grid of time slices:

```bash
uv run scripts/view_events.py samples/<file>.bin
```

## Deploy files to the Kria

```bash
river-tools deploy profile.yaml root@durance-kria.local     # lists the files and asks before copying
```

The package includes a profile template, `river_tools/deployment/deployment_template.yaml`.

## File format

The files are **the same format as the Motion Buddy dataset** (see [motion-buddy-dataset.md](motion-buddy-dataset.md)):

```
4 bytes   big-endian uint32: number of bytes that follow
then 13 bytes per event, big-endian:  x uint16 | y uint16 | polarity uint8 | timestamp int64 (µs)
```

## Turning recordings into training data

Your own recordings differ from the dataset clips in two ways:

- **Length:** they run from Start to Stop, while dataset clips are exactly 1 s. The reference training code only uses the first second, so cut each recording into 1 s pieces around the gesture.
- **Name:** they carry no class. The training code reads the class from the name, `gesture_<NN>-<name>-<style>-<take>.bin` (e.g. `gesture_01-sit-normal-03.bin`), so rename them to match.

Recording tips:
- Keep one folder per person, so you can test on people the model hasn't seen.
- Vary speed and size the way the dataset does (`normal` / `fast` / `large`).
- Record a "no action" class of people moving without doing a gesture.

Our cued recorder does the cutting and naming for you: it shows which gesture to do, labels each take
from the cue and cuts it into 1 s clips with quality filters. See
[`training/optimised_snn/dataset/README.md`](../training/optimised_snn/dataset/README.md).
