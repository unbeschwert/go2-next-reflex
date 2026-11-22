# Recording our own gesture dataset

These scripts turn live recordings from the board's event camera into training clips in the
Motion Buddy layout, so `train_snn_v8.py --zip` reads them like the Durance AI team's dataset.

```
record_session.py  ──►  cut_takes.py --labeled  ──►  export_sessions.py   (share zip, person-safe split)
 cued recording          1 s clips + quality          or pack_dataset.py   (added to a copy of the
 labels from the cue     filters + previews                                 Motion Buddy zip)
```

| file | what it does |
|---|---|
| `gestures.json` | the gestures: id, on-screen instruction, Go2 action, styles, groups (`all`, `follow`) |
| `record_session.py` | cued session: shows the gesture to do with a countdown, records the board camera, labels each take from the cue |
| `cut_takes.py` | cuts takes into 1 s clips, drops bad windows, writes previews, keeps the raw takes for re-cutting |
| `export_sessions.py` | builds the shareable zip (`labels.csv`, `gestures.json`, README) and checks the split by person |
| `pack_dataset.py` | appends our sessions to a copy of the Motion Buddy zip (mixed training) |
| `label_recordings.py` | files plain `river-tools capture` recordings (one per gesture) without cutting |

## Gestures

| id | gesture | output | Go2 action |
|---|---|---|---|
| 1 | heart (arms make a heart over the head) | o0 | finger heart (1036) |
| 2 | turn left 360 (left arm out to the side and back) | o1 | closed-loop turn left 360° |
| 3 | turn right 360 (right arm out to the side and back) | o2 | closed-loop turn right 360° |
| 4 | follow me (walk on the spot) | o3 | LiDAR person following |
| 5 | no action (natural movement, none of the gestures) | o4 | nothing |

Output = id − 1, so the board keeps 5 outputs. Turn left and turn right are mirror images
(`"mirror"` in `gestures.json`): the flip augmentation must swap their labels (`--mirror 1:2`).

## Record

Needs `river-tools` (its Python has the camera stream code), `numpy`, `matplotlib`, and the board
connected over Ethernet. Only one program can open the camera, so stop the board's inference first.

```bash
ssh root@192.168.77.1 '/root/inference.sh stop'
RT=~/.local/share/uv/tools/river-tools/bin/python

# one new person: heart, turn left, turn right, no action; 1 min each (20 s normal / fast / large)
$RT record_session.py --session 20

# follow me for a person who already recorded (session 20): new session, same person
$RT record_session.py --session 31 --classes follow --no-action 0 --person 20 --minutes 3

ssh root@192.168.77.1 '/root/inference.sh start --headless'
```

Keys: space = pause/resume, r = redo the previous take, n = skip, q = quit. Keep the laptop
screen out of the camera's view; the person must be fully in view (feet too for follow me).

Recording tips that mattered for us:
- **Repeat one gesture without stopping** for the whole take: clips are 1 s slices of an ongoing
  gesture, like the Motion Buddy clips. A held, still arm is invisible to an event camera.
- **One session number per person** (20, 21, ...). Extra sessions of the same person get
  `--person <their first session>`, saved in `person.txt`, so they stay in the same split.
- The camera must not move during a session.

## Cut into clips

```bash
python3 cut_takes.py --session 20 --labeled      # cue-labelled takes -> 1 s clips
python3 cut_takes.py --session 20 --summary      # clips per gesture and style
python3 cut_takes.py --session 20 --recut --min-events 1500   # cut again with other thresholds
```

Each take loses its first and last second, the rest is tiled into 1 s windows, and a window is
dropped when it has too few events (< 2,000: pause or person out of view), when motion smears
over most of the frame (camera moved), when activity sits at the image border, or when it is an
outlier against the rest of the take. `--best N` keeps the N most typical clips per take.

Folders:

```
~/river-captures/s20/labeled/   takes from record_session.py
~/river-captures/s20/raw/       takes after cutting (for --recut)
~/river-captures/s20/previews/  one PNG per take: event rate, kept windows, thumbnails
~/river-captures/s20/person.txt who recorded the session
~/go2-hackathon/data/20/gesture_<id>-<name>-<style>-<k>.bin   the clips
```

Clip format = Motion Buddy format (see [`doc/capture-data.md`](../../../doc/capture-data.md)):
a 4-byte big-endian length, then 13 bytes per event (x, y, polarity, timestamp in µs).

## Export

```bash
python3 export_sessions.py --sessions 20,21,22,23,24,25,26,27,28,29,30,31,32,33 \
  --val 28 --test 22,29,30,33          # -> ~/go2-hackathon/share/go2_gestures_s20-33.zip
python3 export_sessions.py --sessions 20 --raw   # also zip the raw takes
```

The split is by person, so test scores show how a model does on people it has never seen.
The exporter refuses a split that puts one person's sessions in different parts, and its README
prints the matching `train_snn_v8.py` command. `--time-split 70,15,15` splits every take in time
instead, for when there are fewer than three people.

To train on our sessions together with the Durance AI team's data:

```bash
python3 pack_dataset.py --sessions 20,21 --out motion_buddy_plus.zip
```

## Our dataset

Sessions 20-30 are 11 people with heart / turn left / turn right / no action (2,442 clips).
Sessions 31-33 are follow me by the people of sessions 20-22 (613 clips). Split used:
validation 28, test 22, 29, 30, 33 (`go2_gestures_s20-33.zip`, shared separately).
