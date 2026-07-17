#!/bin/bash
#SBATCH -J repaireval
#SBATCH -A egitimg16
#SBATCH -p kolyoz-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:20:00
#SBATCH --exclude=kolyoz19,kolyoz61
#SBATCH -o ${WORKSPACE}/repaireval_%j.out
#SBATCH -e ${WORKSPACE}/repaireval_%j.err
set -euo pipefail
WORK=${WORKSPACE}
SIF=${PROJECT_ROOT}/containers/hf_transformers_pytorch_gpu_latest.sif
CKPT=${CONTAINER_PROJECT}/models/nedoqwen08b_9000_sft_repair_v2_600/sft_final.pt
cd "$WORK"
echo "START $(date)"
apptainer exec --nv \
  --bind ${PROJECT_SCRATCH}:/workspace \
  --bind ${USER_HOME}:/home_u5 \
  "$SIF" bash -lc '
set -euo pipefail
export HF_HOME=/home_u5/.cache/huggingface
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=${CONTAINER_PROJECT}:${CONTAINER_TOKENIZER}:${PYTHONPATH:-}
cd /home_u5/chatgpt_workspaces/nedo_eval_harness_v2
python3 - <<"PY"
import torch, pathlib, sys
ck=pathlib.Path("${CONTAINER_PROJECT}/models/nedoqwen08b_9000_sft_repair_v2_600/sft_final.pt")
print("gpu_check cuda=", torch.cuda.is_available(), "count=", torch.cuda.device_count(), flush=True)
print("ckpt_exists", ck.exists(), ck, flush=True)
if not torch.cuda.is_available(): sys.exit(88)
if not ck.exists(): sys.exit(77)
PY
python3 run_turkishmmlu_sub_logprob.py \
  --data benchmark_data/turkishmmlu/turkishmmlu_sub.json \
  --out-dir repaired_v2_600_turkishmmlu_sub_${SLURM_JOB_ID} \
  --models nedo \
  --nedo-ckpt ${CONTAINER_PROJECT}/models/nedoqwen08b_9000_sft_repair_v2_600/sft_final.pt
python3 run_mcq_logprob_generic.py \
  --data benchmark_data/tumlu_mini/tumlu_mini_turkish_test.json \
  --benchmark tumlu_mini_turkish \
  --out-dir repaired_v2_600_tumlu_mini_${SLURM_JOB_ID} \
  --models nedo \
  --nedo-ckpt ${CONTAINER_PROJECT}/models/nedoqwen08b_9000_sft_repair_v2_600/sft_final.pt
'
echo "END $(date)"
