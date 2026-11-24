# Training on a Runpod GPU pod

When the team GPU server isn't available, a Runpod pod is the training server: `remote/runpod.sh` creates
it and writes `remote/runpod.env`, and the usual `remote/*.sh` scripts use it through
`REMOTE_ENV=remote/runpod.env`. Adapted from the tabfm-bench Runpod setup (same API calls and defaults).

Defaults: **Secure Cloud, EU datacenters, one GPU with ≥ 40 GB, the most VRAM first, up to $2.50/h, the
account's global volume at `/workspace-global`, CUDA ≥ 13.0** (the locked torch 2.14 wheels use CUDA 13.0).
Code, the riverops wheel and the data go up over rsync, so the pod needs no Runpod secrets; the wheel is
proprietary and is never written to the global volume.

## Once per computer

- A Runpod API key (console → *Settings* → *API keys*, permission *All*): `export RUNPOD_API_KEY=…` or
  `apikey = "…"` in `~/.runpod/config.toml`.
- Your SSH public key in console → *Settings* → *SSH public keys* (the pod gets it via `startSsh`).
- `jq`, `rsync`, `uv` locally.

## A run

```bash
remote/runpod.sh list                          # matching GPUs and prices, creates nothing
remote/runpod.sh create                        # asks, creates, waits for SSH, writes remote/runpod.env
export REMOTE_ENV=remote/runpod.env
remote/setup.sh                                # rsync, tmux, uv, the training env, GPU check (~3 GB download)
remote/push_data.sh --config training/configs/motion_buddy_80_v4.yaml   # only the clips it uses (~3 GB)
remote/queue.sh lit training/configs/lit/*.yaml                         # runs/lit/<config>/ on the pod
remote/status.sh lit/sew_v4 -f                 # follow one run
remote/fetch.sh lit                            # → runs/lit/ locally
remote/runpod.sh terminate                     # stop billing
```

From a git worktree (no `data/` or `.venv` of its own), point the scripts at the main checkout:
`DATA_DIR=<main>/data PYTHON=<main>/.venv/bin/python`.

`queue.sh` runs `PARALLEL` configs at a time (default: GPU memory / 20 GB; an 80×80 run needs ~19 GB).
When the queue ends on a pod it copies `runs/<name>` to `/workspace-global/go2-next-reflex/runs/<name>`,
waits `STOP_AFTER_MIN` (60) minutes so you can fetch, then **stops the pod** (`WHEN_DONE=stop`, the default
on a pod; `WHEN_DONE=none` keeps it running). A stopped pod costs nothing for the GPU, but its container
disk is wiped: what wasn't fetched is only on the global volume. Terminate it when you're done.

## Storage

| Where | Holds | Lifetime |
|---|---|---|
| Container disk, `/root/go2-next-reflex` | code, `.venv`, `data/`, `cache/`, `runs/` | wiped when the pod stops |
| Global volume, `/workspace-global/go2-next-reflex/` | `data/` (copy, restored by `push_data.sh` on the next pod), `runs/<queue>` (copy at the end of a queue) | permanent |

The global volume is object storage: no modification times, atomic rename or locking, so the scripts
write to it with `rsync -r --inplace` (no `-t`) and the training never reads or writes it directly.

Pods reuse IP:port pairs with new host keys, so their keys go to `~/.ssh/known_hosts_runpod`
(`StrictHostKeyChecking=accept-new`), not to `~/.ssh/known_hosts`.
