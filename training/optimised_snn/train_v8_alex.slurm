#!/bin/bash -l
# Improved Motion Buddy SNN training (train_snn_v8.py) on one A100 (NHR@FAU Alex).
#   cd $WORK/motion_buddy
#   sbatch --job-name=snn-best80 train_v7_alex.slurm --arch best80
#   sbatch --job-name=snn-nb32 --time=04:00:00 train_v7_alex.slurm --arch notebook32
#SBATCH --job-name=snn-v2
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python/3.12-base
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
export OMP_NUM_THREADS=4

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u train_snn_v8.py --zip data/motion_buddy_full.zip --out "runs/$SLURM_JOB_NAME-$SLURM_JOB_ID" \
  --workers "${SLURM_CPUS_ON_NODE:-16}" "$@"
