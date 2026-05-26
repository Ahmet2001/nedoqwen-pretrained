#!/bin/bash
#SBATCH -J hf_upload_seq
#SBATCH -A egitimg16
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o logs/hf_upload_seq_%j.out
#SBATCH -e logs/hf_upload_seq_%j.err

set -euo pipefail

cd /arf/scratch/egitimg16/nedo_slm
CONTAINER=$(cat containers/selected_container.txt)

apptainer exec \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  bash -lc '
    set -euo pipefail
    export PATH="$HOME/.local/bin:$PATH"
    cd /workspace/nedo_slm

    REPO="Ethosoft/nedo-turkish-65k-tokenized-60b"

    echo "START $(date)"
    hf auth whoami

    echo "Uploading README and manifest..."
    hf upload "$REPO" hf_upload_nedo_65k_tokens/README.md README.md --repo-type dataset
    hf upload "$REPO" hf_upload_nedo_65k_tokens/metadata/manifest.json metadata/manifest.json --repo-type dataset

    echo "Uploading validation shards..."
    for f in hf_upload_nedo_65k_tokens/val/*.bin; do
      base=$(basename "$f")
      echo "UPLOAD val/$base $(date)"
      hf upload "$REPO" "$f" "val/$base" --repo-type dataset
      echo "DONE val/$base $(date)"
    done

    echo "Uploading train shards..."
    for f in hf_upload_nedo_65k_tokens/train/*.bin; do
      base=$(basename "$f")
      echo "UPLOAD train/$base $(date)"
      hf upload "$REPO" "$f" "train/$base" --repo-type dataset
      echo "DONE train/$base $(date)"
    done

    echo "ALL DONE $(date)"
  '
