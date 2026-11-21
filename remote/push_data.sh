#!/usr/bin/env bash
# Upload the dataset (and optionally more recording folders) to the server's data/ folder.
# Resumable: re-run it after an interruption. Usage: remote/push_data.sh [extra_folder ...]
source "$(dirname "$0")/_common.sh"
folders=("$ROOT/data/motion_buddy_dataset" "$@")
remote "mkdir -p $REMOTE_DIR/data"
for folder in "${folders[@]}"; do
    echo "→ uploading $folder"
    rsync -az --partial --info=progress2 --exclude '.DS_Store' -e "ssh ${SSH_OPTS[*]}" \
        "${folder%/}" "$REMOTE_HOST:$REMOTE_DIR/data/"
done
