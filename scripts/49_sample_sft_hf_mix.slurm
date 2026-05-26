#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A egitimg16
#SBATCH -J samplehf
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/sample_sft_hf_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/sample_sft_hf_%j.err

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  python /workspace/nedo_slm/scripts/30_sample_qwen_style.py \
    --ckpt /workspace/nedo_slm/models/nedoqwen08b_sft_hf_mix/sft_final.pt \
    --vocab /workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl \
    --prompt-file /workspace/nedo_slm/prompts/sft_hf_unique.txt \
    --max-new-tokens 160 \
    --temperature 0.35 \
    --top-p 0.75
