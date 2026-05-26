#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J sanity_nedo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/train_sanity_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/train_sanity_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Container: $CONTAINER"

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  bash -lc '
    python /workspace/nedo_slm/scripts/20_train_qwen_style.py \
      --train-bin /workspace/nedo_slm/processed/nedo_65k_20b_train.bin \
      --val-bin /workspace/nedo_slm/processed/nedo_65k_20b_val.bin \
      --out-dir /workspace/nedo_slm/models/sanity_test \
      --profile sanity \
      --batch-size 8 \
      --grad-accum 4 \
      --max-steps 100 \
      --lr 5e-4 \
      --warmup-steps 20 \
      --eval-interval 50 \
      --eval-iters 10 \
      --save-interval 100
  '
