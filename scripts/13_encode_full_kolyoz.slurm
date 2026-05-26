#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J encode_full_nedo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/encode_full_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/encode_full_%j.err

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
    export PYTHONPATH=/workspace/tokenizer/NedoTurkishTokenizer
    python /workspace/nedo_slm/scripts/12_encode_full_parallel.py \
      --corpus /workspace/data/finewebjsonl/finewebtr_combined.jsonl \
      --vocab /workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl \
      --out-prefix /workspace/nedo_slm/processed/nedo_65k_full \
      --workers 16 \
      --val-every 100 \
      --flush-tokens 2000000
  '
