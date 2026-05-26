#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J cancel4gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/cancel4gpu_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/cancel4gpu_%j.err

echo "Cancel job started at $(date)"
echo "Cancelling old 4GPU training job 1271270 if still running..."
scancel 1271270 2>/dev/null || true
echo "Done at $(date)"
