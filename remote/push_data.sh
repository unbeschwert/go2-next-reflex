#!/usr/bin/env bash
# Upload the dataset (and optionally more recording folders) to the server's data/ folder.
# Resumable: re-run it after an interruption.
#
#   remote/push_data.sh [extra_folder ...]      the whole data/motion_buddy_dataset (+ extra folders)
#   remote/push_data.sh --config <cfg.yaml>     only the clips that config's train/val/test splits use
#                                               (5 classes of motion_buddy: 3,733 of 6,175 files, 9.7 of 11 GB)
#
# On a Runpod pod with the global volume mounted, the upload is also copied to
# /workspace-global/go2-next-reflex/data, and later pods restore it from there instead of uploading.
source "$(dirname "$0")/_common.sh"
GLOBAL_DATA=/workspace-global/go2-next-reflex/data
remote "mkdir -p $REMOTE_DIR/data"

# A copy left on the global volume by an earlier pod: restore it first (rsync then only sends what's missing).
remote "if [ -d $GLOBAL_DATA ]; then echo '→ restoring data from the global volume'; rsync -rt --ignore-existing $GLOBAL_DATA/ $REMOTE_DIR/data/; fi"

if [[ ${1:-} == --config ]]; then
    config=${2:?usage: remote/push_data.sh --config training/configs/<name>.yaml}
    list="$(mktemp)"
    trap 'rm -f "$list"' EXIT
    # Paths relative to DATA_DIR, from the same discovery the training uses (data.root under DATA_DIR).
    (cd "$ROOT" && "${PYTHON_CMD[@]}" - "$config" "$DATA_DIR" >"$list" <<'PY'
import sys
from pathlib import Path
from training.snn.config import load_config
from training.snn.data import discover

cfg = load_config(sys.argv[1])
data = Path(sys.argv[2]).resolve()
root = Path(cfg.data.root)
cfg.data.root = str(data / root.relative_to("data")) if not root.is_absolute() else str(root)
for clips in discover(cfg).values():
    for clip in clips:
        print(Path(clip.path).resolve().relative_to(data))
PY
    )
    echo "→ uploading $(wc -l <"$list") clips used by $config"
    rsync -az --partial --info=progress2 --files-from="$list" -e "ssh ${SSH_OPTS[*]}" \
        "$DATA_DIR/" "$REMOTE_HOST:$REMOTE_DIR/data/"
else
    folders=("$DATA_DIR/motion_buddy_dataset" "$@")
    for folder in "${folders[@]}"; do
        echo "→ uploading $folder"
        rsync -az --partial --info=progress2 --exclude '.DS_Store' -e "ssh ${SSH_OPTS[*]}" \
            "${folder%/}" "$REMOTE_HOST:$REMOTE_DIR/data/"
    done
fi

# The global volume can't set times or rename atomically: -r without -t, --inplace (see remote/RUNPOD.md).
remote "if [ -d /workspace-global ]; then echo '→ saving a copy to the global volume'; mkdir -p $GLOBAL_DATA
    rsync -r --inplace --ignore-existing --exclude 'riverops-*.whl' $REMOTE_DIR/data/ $GLOBAL_DATA/; fi"
