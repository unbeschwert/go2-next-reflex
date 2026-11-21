# Deploying a new SNN on the Kria

Condensed from the RIVER SDK cheatsheet "Deploy an SNN". Copy the files with `scp`, or with `river-tools deploy` (see [capture-data.md](capture-data.md)).

## What has to change

| File | Where on the Kria | What it is |
|---|---|---|
| `spleat.bit.bin` | `/usr/lib/firmware/xilinx/spleat/spleat.bit.bin` | FPGA bitstream |
| `spleat.dtbo` | `/usr/lib/firmware/xilinx/spleat/spleat.dtbo` | Linux device-tree overlay |
| `config.yaml` | `/root/.config/spleat-server/config.yaml` | SPLEAT server config: input/output shape, weights file |
| `metavision_camera.json` | `/root/metavision_camera.json` | Camera and despot settings: frame rate, downsampling, noise filter |
| `weights.txt` | `/root/weights.txt` | SNN parameters as integers |
| ROS node | see [go2-kria-overview.md](go2-kria-overview.md) | Needs updating if the number or meaning of outputs changes |

## 1. Bitstream and device-tree overlay

Generate them on https://riversdk.dev:

1. Export your model's JSON (`export_for_river`, see [training-snn.md](training-snn.md)).
2. Log in to the RIVER SDK.
3. Click **Open firmware jobs**, then **New firmware**.
4. Fill in the form, picking board **KV260** and Prophesee IP **GenX320**.
5. Submit and wait for the job.
6. Copy the two files to `/usr/lib/firmware/xilinx/spleat/`.

### From the command line: `scripts/firmware_job.py`

Steps 2–5 can be scripted: submit, follow the job, and download when it's ready.

**The firmware depends only on `model.json` (layer shapes) and the form options, not on the weights.** Retraining the same network with the same formats doesn't need new firmware; only `weights.txt` changes. So `submit` first looks for a finished job with an identical `model.json` and options and reuses it. Building one takes 6–8 minutes otherwise.

```bash
# once: your browser's riversdk.dev session cookie (DevTools → Storage → Cookies → __Host-riversdk_session)
mkdir -p ~/.config/riversdk && (umask 077; echo '<cookie value>' > ~/.config/riversdk/session)
#   (or export RIVERSDK_SESSION=...). Never commit it; it is a login.

uv run scripts/firmware_job.py list                                  # your team's jobs
uv run scripts/firmware_job.py submit --run runs/<run> --dry-run     # show the form values, reuse check
uv run scripts/firmware_job.py submit --run runs/<run> --download    # submit (or reuse), poll, download, unpack
uv run scripts/firmware_job.py status 32                             # one job's status, formats, SHA256
uv run scripts/firmware_job.py wait 32                               # poll every 30 s until it's done or failed
uv run scripts/firmware_job.py download 32 --out runs/<run>/firmware-job32
```

- With `--run`, the form is filled from the training run. `model.json` comes from `<run>/deploy/`. The weight and potential formats come from `export_summary.json`, so they always match `weights.txt`. The strategy is the run's `fpga.strategy`. Prophesee IP defaults to **GenX320**, needed for the camera. Any field can be overridden: `--weights Q2.6 --potentials Q4.12 --strategy ... --prophesee-ip none --name ... --description ...`; or use `--model file.json` without a run.
- `--force` submits even when an identical job exists.
- `download` checks the file against the SHA256 shown on the job page and unpacks `spleat.bit.bin` and `spleat.dtbo`. It also saves the job's `model.json` and a `job.json` with the options.
- How it works: it loads `/jobs/new` for the form's one-time `_csrf` and `_submission_key` tokens, then posts the same multipart form the browser sends. The job ID comes from the redirect to `/jobs/<id>`, and the status is read from the job page ("Waiting for a runner", …, "✓ Firmware generated"). If the site changes its HTML, the script stops with an error rather than guessing.

## 2. Server config: `/root/.config/spleat-server/config.yaml`

```yaml
application:
  host: "0.0.0.0"          # keep
  port: 3333               # keep, since the ROS gesture_node connects here
  weights_file: /root/weights.txt
  max_concurrency: 40      # keep
inputs:
  channels: 2              # ON/OFF polarity
  height: 80               # must match the frames despot sends (320 / downsampling)
  width: 80
  scaling: 4096.0          # = 2^frac_bits of the potential format (Q4.12 → 4096)
outputs:
  channels: 5              # number of classes / output neurons
  height: 1
  width: 1
```

## 3. Camera config: `/root/metavision_camera.json`

```json
{
  "frequency": 100.0,
  "downsampling": 4,
  "noise_filtering_factor": 0.5,
  "biases": {
    "bias_diff": 51, "bias_diff_off": 40, "bias_diff_on": 40,
    "bias_fo": 29, "bias_hpf": 0, "bias_refr": 82
  }
}
```

- **`frequency`:** how many frames per second despot builds from the events. 100 Hz means 10 ms frames, the same as the training notebook.
- **`downsampling`:** one of 1, 2, 4, 8, 16, 32, 64, applied to both sides of the image. 1 gives 320×320, 2 gives 160×160, **4 gives 80×80**, 8 gives 40×40. It must match `inputs.height/width` in `config.yaml` **and** the size you trained at.
- **`noise_filtering_factor`:**
  - very high (e.g. 100) filters almost nothing
  - very low (e.g. 0.0001) filters almost everything, useful events included
  - 0.1–1.0 is sensible
  - delete the line to switch filtering off
- **`biases`:** settings for the Prophesee sensor itself. See docs.prophesee.ai.

## 4. Weights: `/root/weights.txt`

One integer per line. **No empty lines, not even at the end.** The order is:

```
[weights layer 1] … [weights layer N]
[biases layer 1]  … [biases layer N]
threshold layer 1
leak layer 1
…
threshold layer N
leak layer N
```

This is exactly what the notebook writes to `motion_buddy_parameters.txt`. Rename it, or point `weights_file` at it.

## 5. Restart and check

```bash
systemctl restart spleat-server.service      # or start by hand to see logs, see kria-setup.md
journalctl -u spleat-server.service -f
```

The server then sends one result per frame over TCP (port 3333): a 4-byte length followed by CBOR `{"id", "payload", "latency", "error"}`. `payload` holds `outputs.channels` × float32. `gesture_node` in `go2-kria` reads these.
