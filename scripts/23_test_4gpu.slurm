#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J test4gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/test4gpu_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/test4gpu_%j.err

cd /arf/scratch/egitimg16/nedo_slm
CONTAINER=$(cat containers/selected_container.txt)

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  bash -lc '
    echo "INSIDE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi
    python - <<PY
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch cuda count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
  '
