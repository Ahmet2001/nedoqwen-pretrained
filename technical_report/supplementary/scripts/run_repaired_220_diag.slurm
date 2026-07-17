#!/bin/bash
#SBATCH -J repair220
#SBATCH -A egitimg16
#SBATCH -p kolyoz-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=kolyoz19
#SBATCH -o ${WORKSPACE}/repair220_%j.out
#SBATCH -e ${WORKSPACE}/repair220_%j.err

set -euo pipefail
W=${WORKSPACE}
python3 $W/make_repair_220_eval_script.py

cd ${PROJECT_ROOT}
CONTAINER=$(cat containers/selected_container.txt)
mkdir -p eval/nedo_v2_repair_220
cp $W/prompts_tr_expanded_220.jsonl eval/nedo_v2_repair_220/prompts_tr_expanded_220.jsonl
cp $W/run_generation_eval_v2_repair_only.py eval/nedo_v2_repair_220/run_generation_eval_v2_repair_only.py

apptainer exec \
  --nv \
  --bind ${PROJECT_SCRATCH}:/workspace \
  "$CONTAINER" \
  bash -lc "cd ${CONTAINER_PROJECT} && export PYTHONPATH=${CONTAINER_PROJECT}:${CONTAINER_TOKENIZER}:\$PYTHONPATH && python eval/nedo_v2_repair_220/run_generation_eval_v2_repair_only.py --prompts ${CONTAINER_PROJECT}/eval/nedo_v2_repair_220/prompts_tr_expanded_220.jsonl --vocab /workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl --out-dir ${CONTAINER_PROJECT}/eval/nedo_v2_repair_220/results_repair_220 --max-new-tokens 48 --temperature 0.0"

rm -rf $W/results_repair_220_${SLURM_JOB_ID}
cp -r ${PROJECT_ROOT}/eval/nedo_v2_repair_220/results_repair_220 $W/results_repair_220_${SLURM_JOB_ID}
