#!/bin/bash
#SBATCH -J repair600
#SBATCH -A egitimg16
#SBATCH -p kolyoz-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:20:00
#SBATCH --exclude=kolyoz19,kolyoz61
#SBATCH -o ${WORKSPACE}/repair600_%j.out
#SBATCH -e ${WORKSPACE}/repair600_%j.err
set -euo pipefail
SIF=${SYSTEM_CONTAINERS}/truba-ai/pytorch-25.10-py3.sif
cd ${PROJECT_ROOT}
apptainer exec --nv --bind ${PROJECT_SCRATCH}:/workspace "$SIF" bash -lc '
set -euo pipefail
cd ${CONTAINER_PROJECT}
export PYTHONPATH=${CONTAINER_PROJECT}:${CONTAINER_TOKENIZER}:${PYTHONPATH:-}
python - <<"PY"
import torch, sys
print("gpu_check cuda=", torch.cuda.is_available(), "count=", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available(): sys.exit(88)
PY
python scripts/42_sft_instruction_smoke.py \
  --ckpt ${CONTAINER_PROJECT}/models/nedoqwen08b_9000_sft_clean_20k/sft_final.pt \
  --data ${CONTAINER_PROJECT}/sft_data/tr_repair_mix_v2_plus_clean.jsonl \
  --out-dir ${CONTAINER_PROJECT}/models/nedoqwen08b_9000_sft_repair_v2_600 \
  --vocab /workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl \
  --max-steps 600 \
  --save-interval 200 \
  --batch-size 2 \
  --grad-accum 8 \
  --block-size 1024 \
  --lr 1e-5
'
