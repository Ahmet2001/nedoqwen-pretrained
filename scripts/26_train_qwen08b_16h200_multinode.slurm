#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J nedoqwen16h
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/train_qwen08b_16h_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/train_qwen08b_16h_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501
NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=4

echo "Job: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "Container: $CONTAINER"
echo "Date: $(date)"

srun --ntasks-per-node=1 bash -lc "
  echo '===== NODE START ====='
  echo Host: \$(hostname)
  echo SLURM_NODEID: \$SLURM_NODEID
  echo CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

  apptainer exec --nv \
    --bind /arf/scratch/egitimg16:/workspace \
    $CONTAINER \
    bash -lc '
      export NCCL_DEBUG=INFO
      export NCCL_IB_DISABLE=1
      export OMP_NUM_THREADS=16
      export CUDA_DEVICE_MAX_CONNECTIONS=1
      export PYTHONUNBUFFERED=1

      torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=\$SLURM_NODEID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        /workspace/nedo_slm/scripts/20_train_qwen_style.py \
        --train-bin /workspace/nedo_slm/processed/nedo_65k_20b_train.bin \
        --val-bin /workspace/nedo_slm/processed/nedo_65k_20b_val.bin \
        --out-dir /workspace/nedo_slm/models/nedoqwen08b_20b_16h200 \
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
"
