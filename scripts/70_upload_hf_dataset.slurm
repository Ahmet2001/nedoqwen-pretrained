#!/bin/bash
#SBATCH -J hf_upload_nedo60b
#SBATCH -A egitimg16
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/hf_upload_nedo60b_%j.out
#SBATCH -e logs/hf_upload_nedo60b_%j.err

set -euo pipefail

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "PWD: $(pwd)"
echo "CONTAINER: $CONTAINER"

apptainer exec \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  bash -lc '
    export PATH="$HOME/.local/bin:$PATH"
    cd /workspace/nedo_slm

    echo "Inside container:"
    pwd
    hf auth whoami

    du -sh hf_upload_nedo_65k_tokens
    find hf_upload_nedo_65k_tokens -type f | wc -l

    export HF_XET_HIGH_PERFORMANCE=1

    hf upload-large-folder Ethosoft/nedo-turkish-65k-tokenized-60b \
      --repo-type dataset \
      hf_upload_nedo_65k_tokens \
      --num-workers=2 \
      --no-bars
  '

echo "END: $(date)"
EOFcd /arf/scratch/egitimg16/nedo_slm

cat > scripts/70_upload_hf_dataset.slurm <<'EOF'
#!/bin/bash
#SBATCH -J hf_upload_nedo60b
#SBATCH -A egitimg16
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/hf_upload_nedo60b_%j.out
#SBATCH -e logs/hf_upload_nedo60b_%j.err

set -euo pipefail

cd /arf/scratch/egitimg16/nedo_slm

CONTAINER=$(cat containers/selected_container.txt)

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "PWD: $(pwd)"
echo "CONTAINER: $CONTAINER"

apptainer exec \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  bash -lc '
    export PATH="$HOME/.local/bin:$PATH"
    cd /workspace/nedo_slm

    echo "Inside container:"
    pwd
    hf auth whoami

    du -sh hf_upload_nedo_65k_tokens
    find hf_upload_nedo_65k_tokens -type f | wc -l

    export HF_XET_HIGH_PERFORMANCE=1

    hf upload-large-folder Ethosoft/nedo-turkish-65k-tokenized-60b \
      --repo-type dataset \
      hf_upload_nedo_65k_tokens \
      --num-workers=2 \
      --no-bars
  '

echo "END: $(date)"
