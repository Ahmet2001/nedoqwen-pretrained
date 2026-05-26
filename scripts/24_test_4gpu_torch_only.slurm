#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J test4torch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/test4torch_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/test4torch_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Container=$CONTAINER"

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  python /workspace/nedo_slm/scripts/test_cuda_count.py
