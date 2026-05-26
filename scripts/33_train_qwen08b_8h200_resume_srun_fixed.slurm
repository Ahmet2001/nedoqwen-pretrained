#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J nedoqwen8h
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/train_qwen08b_8h_fixed_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/train_qwen08b_8h_fixed_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

RDZV_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
RDZV_PORT=$((20000 + SLURM_JOB_ID % 40000))

RESUME_CKPT=$(ls -t models/nedoqwen08b_20b/final.pt models/nedoqwen08b_20b/ckpt_step_*.pt 2>/dev/null | head -n 1)

if [ -z "$RESUME_CKPT" ]; then
  echo "ERROR: No resume checkpoint found in models/nedoqwen08b_20b"
  exit 1
fi

echo "Job: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "RDZV_HOST: $RDZV_HOST"
echo "RDZV_PORT: $RDZV_PORT"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "RESUME_CKPT: $RESUME_CKPT"
echo "Container: $CONTAINER"
echo "Date: $(date)"

export RDZV_HOST
export RDZV_PORT
export RESUME_CKPT

srun --kill-on-bad-exit=1 --gpu-bind=single:1 \
  apptainer exec --nv \
    --env RDZV_HOST="$RDZV_HOST" \
    --env RDZV_PORT="$RDZV_PORT" \
    --env RESUME_CKPT="$RESUME_CKPT" \
    --bind /arf/scratch/egitimg16:/workspace \
    "$CONTAINER" \
    bash -lc '
      export MASTER_ADDR=$RDZV_HOST
      export MASTER_PORT=$RDZV_PORT

      export RANK=$SLURM_PROCID
      export WORLD_SIZE=$SLURM_NTASKS

      # --gpu-bind=single:1 nedeniyle her task tek GPU görüyor.
      # Bu yüzden local rank her process için 0 olmalı.
      export LOCAL_RANK=0

      export NCCL_DEBUG=INFO
      export NCCL_IB_DISABLE=1
      export OMP_NUM_THREADS=16
      export CUDA_DEVICE_MAX_CONNECTIONS=1
      export PYTHONUNBUFFERED=1

      echo "===== WORKER START ====="
      echo "host=$(hostname)"
      echo "RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE"
      echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
      echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

      python /workspace/nedo_slm/scripts/20_train_qwen_style_resume.py \
        --train-bin /workspace/nedo_slm/processed/nedo_65k_20b_train.bin \
        --val-bin /workspace/nedo_slm/processed/nedo_65k_20b_val.bin \
        --out-dir /workspace/nedo_slm/models/nedoqwen08b_20b_8h200_resume_fixed \
        --profile qwen08b \
        --batch-size 1 \
        --grad-accum 64 \
        --max-steps 24000 \
        --lr 3e-4 \
        --warmup-steps 500 \
        --eval-interval 500 \
        --eval-iters 20 \
        --save-interval 1000 \
        --resume /workspace/nedo_slm/$RESUME_CKPT
    '
