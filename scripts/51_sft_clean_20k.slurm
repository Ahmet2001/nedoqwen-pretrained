#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J sftsmoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/sft_smoke_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/sft_smoke_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  python /workspace/nedo_slm/scripts/42_sft_instruction_smoke.py \
    --ckpt /workspace/nedo_slm/models/nedoqwen08b_20b/final.pt \
    --vocab /workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl \
    --data /workspace/nedo_slm/sft_data/tr_sft_clean_20k.jsonl \
    --out-dir /workspace/nedo_slm/models/nedoqwen08b_sft_clean_20k \
    --block-size 1024 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-steps 5000 \
    --lr 5e-6 \
    --save-interval 1000
