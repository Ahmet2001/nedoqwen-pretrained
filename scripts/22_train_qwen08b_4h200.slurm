#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J nedoqwen08b
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/train_qwen08b_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/train_qwen08b_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Container: $CONTAINER"

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  bash -lc '
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1
    export OMP_NUM_THREADS=16
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONUNBUFFERED=1

    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node=4 \
      --master_port=29501 \
      /workspace/nedo_slm/scripts/20_train_qwen_style.py \
      --train-bin /workspace/nedo_slm/processed/nedo_65k_20b_train.bin \
      --val-bin /workspace/nedo_slm/processed/nedo_65k_20b_val.bin \
      --out-dir /workspace/nedo_slm/models/nedoqwen08b_20b \
      --profile qwen08b \
      --batch-size 1 \
      --grad-accum 64 \
      --max-steps 5000 \
      --lr 3e-4 \
      --warmup-steps 500 \
      --eval-interval 500 \
      --eval-iters 20 \
      --save-interval 1000
  '
