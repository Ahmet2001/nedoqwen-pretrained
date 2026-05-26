#!/bin/bash
#SBATCH -p palamut-cuda
#SBATCH -A egitimg16
#SBATCH -J nedo_vocab_smoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/vocab_smoke_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/vocab_smoke_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

echo "Container: $CONTAINER"
echo "Running on: $(hostname)"
echo "Date: $(date)"

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  python /workspace/tokenizer/build_nedo_morph_vocab.py \
    --corpus-path /workspace/nedo_slm/processed/smoke_data/finewebtr_5k.jsonl \
    --tokenizer-repo /workspace/tokenizer/NedoTurkishTokenizer \
    --out-dir /workspace/nedo_slm/processed/nedo_vocab_smoke \
    --text-key text \
    --vocab-sizes 8000,16000 \
    --token-mode typed_surface \
    --skip-space \
    --workers 16 \
    --chunks-per-worker 2 \
    --flush-every-docs 2000 \
    --max-docs 5000
