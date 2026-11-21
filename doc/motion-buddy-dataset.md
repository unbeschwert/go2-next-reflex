# Motion Buddy dataset

The supplied hand-gesture dataset: `data/durance-ai-handgesture-recognition.zip` (4.09 GB), extracted to `data/motion_buddy_dataset/`. Each file is a **1-second event-camera recording of one person doing one hand gesture**, recorded with the GenX320.

## Layout

```
motion_buddy_dataset/<person>/gesture_<class>-<name>-<style>-<take>.bin
e.g.  motion_buddy_dataset/01/gesture_01-sit-normal-01.bin
```

- **Person:** folders `01`–`14`, 181–297 clips each, **3,733 clips** in total.
- **Style:** `normal` (1,181 clips), `fast` (1,290) or `large` (1,262), meaning normal speed, faster, or bigger movements.
- **Take:** `01`–`20`.
- Skip the macOS leftovers (`.DS_Store`).

| Class | Name | Clips | Go2 action it's meant for (`api_id`) |
|---|---|---|---|
| 01 | sit | 680 | Sit (1009) |
| 02 | lay-down | 762 | Lie down (1005) |
| 03 | hello | 733 | Hello, waves a paw (1016) |
| 04 | dance | 771 | Dance (1022) |
| 05 | no-action | 787 | Nothing (background / other movement) |

The training notebook's label is `class − 1`, so 0 = Sit … 4 = No action.

## File format

The same as `river-tools` recordings (`river_tools.capture.events`):

```
[4 bytes]  big-endian uint32 = number of bytes that follow = 13 × number of events
then one 13-byte record per event, big-endian:
  x          uint16   0–319
  y          uint16   0–319       (320×320 sensor)
  polarity   uint8    0 = got darker (OFF), 1 = got brighter (ON)
  timestamp  int64    microseconds
```

```python
import numpy as np
EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
events = np.fromfile(path, dtype=EVENT_DTYPE, offset=4)
t = events["timestamp"] - events["timestamp"][0]      # µs since the clip started
```

## What we checked across all 3,733 files

- The length header always matches the file size.
- x and y never go above 319, and polarity is only 0 or 1.
- Timestamps never go backwards.
- Every clip covers **1.0 s**.
- Timestamps don't start at zero (5.7 s to 1,699 s), because the clips were cut from longer recordings. Subtract the first timestamp before using them.
- Events per clip: minimum 1,390, **median ~167k**, maximum 2.05 M. Fast and large gestures produce far more. Look at the near-empty clips before training.

## Looking at clips

```bash
uv sync                                        # once
uv run scripts/view_events.py data/motion_buddy_dataset/01/gesture_01-sit-normal-01.bin
uv run scripts/view_events.py <file.bin> --slices 16 --save out.png
```

The viewer cuts the clip into equal time slices and shows each one as a 320×320 picture: red = ON events, blue = OFF events.

### The array behind the pictures

```
frames.shape = (slices, polarity, y, x) = (12, 2, 320, 320), int32 event counts
np.add.at(frames, (slice_index, polarity, y, x), 1)
```

For `01/gesture_01-sit-normal-01.bin` (109,664 events):

- Only **3.1%** of cells are non-zero.
- Counts in non-zero cells: median 1, 99th percentile 5, maximum 126.
- There's no threshold on events. Brightness is scaled by the 99th percentile of non-zero counts, so 5 or more events in a slice shows at full colour. This only affects the display.
- **There's a hot pixel at (x=128, y=231).** It fires 120 ON + 50 OFF events in one ~83 ms slice while its neighbours fire 0–1, so it's sensor noise, not motion. For training, consider masking hot pixels and filtering out isolated events. The camera's own `noise_filtering_factor` (see [deploy-model.md](deploy-model.md)) does something similar on the Kria.

## The reference split

The training notebook splits **by person**:
- training: folders 01–03
- validation: 04
- test: 05

Folders 06–14 aren't used, so there's room to train on more data. Always keep test people out of training.
