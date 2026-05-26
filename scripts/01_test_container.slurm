#!/bin/bash
#SBATCH -p palamut-cuda
#SBATCH -A egitimg16
#SBATCH -J test_nedo_env
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/arf/scratch/egitimg16/nedo_slm/logs/test_container_%j.out
#SBATCH --error=/arf/scratch/egitimg16/nedo_slm/logs/test_container_%j.err

cd /arf/scratch/egitimg16/nedo_slm

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

CONTAINER=$(cat containers/selected_container.txt)

echo "Container: $CONTAINER"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "User: $USER"

apptainer exec --nv \
  --bind /arf/scratch/egitimg16:/workspace \
  "$CONTAINER" \
  python - <<'PY'
import sys
import os
import pkgutil

print("Python:", sys.version)

try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("TORCH_IMPORT_ERROR:", repr(e))

# Local tokenizer repo path
sys.path.insert(0, "/workspace/tokenizer")
sys.path.insert(0, "/workspace/tokenizer/src")

print("Tokenizer path exists:", os.path.exists("/workspace/tokenizer"))
print("Tokenizer src exists:", os.path.exists("/workspace/tokenizer/src"))

print("Possible tokenizer-related modules:")
for p in ["/workspace/tokenizer", "/workspace/tokenizer/src"]:
    if os.path.exists(p):
        mods = [
            m.name for m in pkgutil.iter_modules([p])
            if "nedo" in m.name.lower() or "token" in m.name.lower()
        ]
        print(p, mods[:50])

# Try common import path
try:
    from nedo_turkish_tokenizer import NedoTurkishTokenizer
    tok = NedoTurkishTokenizer()
    print("Nedo import OK: nedo_turkish_tokenizer")
    print(tok.tokenize("İstanbul'da toplantıya katılamadım")[:10])
except Exception as e:
    print("NEDO_IMPORT_ERROR_1:", repr(e))

# Try alternative common patterns, just in case repo structure differs
try:
    from tokenizer import NedoTurkishTokenizer
    tok = NedoTurkishTokenizer()
    print("Nedo import OK: tokenizer")
    print(tok.tokenize("Türkiye'de yapay zeka çalışmaları hızlandı.")[:10])
except Exception as e:
    print("NEDO_IMPORT_ERROR_2:", repr(e))
PY
