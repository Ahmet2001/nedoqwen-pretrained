#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J samplefinal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/sample_final_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/sample_final_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  python /workspace/nedo_slm/scripts/30_sample_qwen_style.py \
    --ckpt /workspace/nedo_slm/models/nedoqwen08b_20b/final.pt \
    --vocab /workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl \
    --prompt "Yapay zeka kısaca, bilgisayar sistemlerinin" \
    --max-new-tokens 160 \
    --temperature 0.55 \
    --top-p 0.80
