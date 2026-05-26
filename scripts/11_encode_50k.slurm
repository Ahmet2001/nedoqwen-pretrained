#!/bin/bash
#SBATCH -p palamut-cuda
#SBATCH -A egitimg16
#SBATCH -J encode50k
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/encode50k_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/encode50k_%j.err

cd /arf/scratch/egitimg16/nedo_slm
CONTAINER=$(cat containers/selected_container.txt)

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  bash -lc 'PYTHONPATH=/workspace/tokenizer/NedoTurkishTokenizer python /workspace/nedo_slm/scripts/10_encode_dataset.py \
    --corpus /workspace/data/finewebjsonl/finewebtr_combined.jsonl \
    --vocab /workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl \
    --out-prefix /workspace/nedo_slm/processed/nedo_65k_50kdocs \
    --max-docs 50000'
