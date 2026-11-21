# Palm tracking with CVAT Community

This pilot turns three recordings from the Durance gesture archive into CVAT video tasks. The `hello` task starts with a hand-labeled palm point track. The `dance` and `no-action` tasks are available for annotation. Use a separate, manually reviewed set for evaluation.

## Start the local CVAT server

Use Docker with Compose 2.24.4 or later, Git, and Python 3.12 or 3.13. Run the commands from the repository root. The Docker configuration uses the official CVAT Community v2.83.0 stack and publishes its web UI on `127.0.0.1:8080`.

```sh
git clone --depth 1 --branch v2.83.0 https://github.com/cvat-ai/cvat.git .local/cvat
docker compose -p go2-reflex-cvat --env-file labeling/cvat.env -f .local/cvat/docker-compose.yml -f labeling/compose.local.yaml up -d
```

Wait for [CVAT](http://localhost:8080) to load, then create a local account. If the demo is already running, use the existing account that owns the demo tasks.

```sh
docker compose -p go2-reflex-cvat --env-file labeling/cvat.env -f .local/cvat/docker-compose.yml -f labeling/compose.local.yaml exec cvat_server python manage.py createsuperuser
```

## Prepare and upload the clips

Place `durance-ai-handgesture-recognition.zip` at the repository root or pass its path with `prepare --archive`. Use an isolated Python environment.

```sh
python -m venv .local/labeling-venv
```

On PowerShell, run `.\.local\labeling-venv\Scripts\Activate.ps1`. On Linux or macOS, run `source .local/labeling-venv/bin/activate`. With the environment active, install the tools and load the pilot.

```sh
python -m pip install -r labeling/requirements.txt
python -m labeling.demo prepare
python -m labeling.demo upload --user YOUR_CVAT_USERNAME
```

The upload prompts for the local CVAT password or reads `CVAT_PASSWORD` if it is set. It prints the three task URLs and skips tasks that already exist under the demo names. Pass `--output` to both commands for a different output directory. The upstream checkout, venv, videos, and timing manifest are ignored under `.local/`.

## Review the pilot

| Clip | Archive group | Preview frames | Events |
| --- | --- | ---: | ---: |
| `hello` | `01` | 20 | 62,751 |
| `dance` | `01` | 20 | 80,388 |
| `no-action` | `03` | 20 | 124,886 |

Open the `hello` task and its job. The seed contains nine visible palm keyframes on frames 7–19. To label the other clips, use a `palm` point track and add keyframes when the palm changes direction or becomes visible again. Mark the track outside when the hand leaves view. A `no-action` gesture clip can still show a hand.

Verified on Docker Desktop with CVAT 2.83.0. All three videos were decoded and imported as 20 frames at `320 × 320`. The server returned the nine stored keyframes for `hello`, and a second upload reused all three tasks.

Each preview frame covers 50,000 source timestamp ticks and is displayed at 20 fps. Use `.local/labeling/manifest.json` to map a frame index `i` back to the original event interval starting at `first_timestamp + i * ticks_per_frame`. The previews retain the original `320 × 320` coordinates. Orange and blue distinguish the two polarities, with white where both occur in the same pixel and time bin. Save labels using CVAT's video annotation export to preserve the tracks.

For a YOLO-assisted round, use a hand detector on the rendered videos and review its proposals in CVAT before converting box centers to palm labels. The pilot uses manual starter points and requires no model weights. Keep the archive and previews on authorized storage.

This command stops the demo and retains its tasks and annotations.

```sh
docker compose -p go2-reflex-cvat --env-file labeling/cvat.env -f .local/cvat/docker-compose.yml -f labeling/compose.local.yaml stop
```
