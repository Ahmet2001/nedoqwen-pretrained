#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J nedoqwen8h
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/train_qwen08b_8h_torchrun_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/train_qwen08b_8h_torchrun_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((25000 + SLURM_JOB_ID % 30000))
NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=4

RESUME_CKPT=$(ls -t models/nedoqwen08b_20b/final.pt models/nedoqwen08b_20b/ckpt_step_*.pt 2>/dev/null | head -n 1)

if [ -z "$RESUME_CKPT" ]; then
  echo "ERROR: No resume checkpoint found in models/nedoqwen08b_20b"
  exit 1
fi

echo "Job: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "RESUME_CKPT: $RESUME_CKPT"
echo "Container: $CONTAINER"
echo "Date: $(date)"

export MASTER_ADDR
export MASTER_PORT
export NNODES
export GPUS_PER_NODE
export RESUME_CKPT

srun --kill-on-bad-exit=1 --ntasks-per-node=1 bash -lc '
  NODE_RANK=$SLURM_NODEID

  echo "===== NODE START ====="
  echo "host=$(hostname)"
  echo "NODE_RANK=$NODE_RANK"
  echo "MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT'"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  apptainer exec --nv \
    --env MASTER_ADDR="'$MASTER_ADDR'" \
    --env MASTER_PORT="'$MASTER_PORT'" \
    --env NNODES="'$NNODES'" \
    --env GPUS_PER_NODE="'$GPUS_PER_NODE'" \
    --env RESUME_CKPT="'$RESUME_CKPT'" \
    --env NODE_RANK="$NODE_RANK" \
    --bind /arf/scratch/egitimg16:/workspace \
    "'$CONTAINER'" \
    bash -lc "
      export NCCL_DEBUG=INFO
      export NCCL_IB_DISABLE=1
      export NCCL_P2P_DISABLE=0
      export OMP_NUM_THREADS=16
      export CUDA_DEVICE_MAX_CONNECTIONS=1
      export PYTHONUNBUFFERED=1

      echo INSIDE_HOST=\$(hostname)
      echo INSIDE_NODE_RANK=\$NODE_RANK
      echo INSIDE_MASTER_ADDR=\$MASTER_ADDR
      echo INSIDE_MASTER_PORT=\$MASTER_PORT
      echo INSIDE_CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES

      torchrun \
        --nnodes=\$NNODES \
        --nproc_per_node=\$GPUS_PER_NODE \
        --node_rank=\$NODE_RANK \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
        --rdzv_conf timeout=900 \
        /workspace/nedo_slm/scripts/20_train_qwen_style_resume.py \
        --train-bin /workspace/nedo_slm/processed/nedo_65k_20b_train.bin \
        --val-bin /workspace/nedo_slm/processed/nedo_65k_20b_val.bin \
        --out-dir /workspace/nedo_slm/models/nedoqwen08b_20b_8h200_resume_torchrun \
        --profile qwen08b \
        --batch-size 1 \
        --grad-accum 64 \
        --max-steps 24000 \
        --lr 3e-4 \
        --warmup-steps 500 \
        --eval-interval 500 \
        --eval-iters 20 \
        --save-interval 1000 \
        --resume /workspace/nedo_slm/\$RESUME_CKPT
    "
'
