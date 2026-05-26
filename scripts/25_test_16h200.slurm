#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J test16h200
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/test16h200_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/test16h200_%j.err

cd /arf/scratch/egitimg16/nedo_slm

echo "Job: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Num nodes: $SLURM_JOB_NUM_NODES"

srun bash -lc '
echo "===== $(hostname) ====="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader
'
