# NEDOQwen / NEDO Turkish SLM

This repository contains code and reproducibility scripts for the NEDO Turkish SLM project.

The large datasets and model checkpoints are released on Hugging Face. This GitHub repository is intended for code, scripts, and documentation only.

## Contents

- `scripts/`: training, encoding, SFT, sampling, inspection, and upload scripts
- `nedo_turkish_tokenizer/`: tokenizer Python package used by the sampling scripts
- `docs/`: usage and artifact documentation
- `requirements.txt`: minimal Python dependencies

## Released artifacts

- Pretraining dataset: `Ethosoft/nedo-turkish-65k-tokenized-60b`
- SFT dataset: `Ethosoft/nedo-turkish-sft-mixtures`
- Base model: `Ethosoft/nedoqwen_0.8b_base_pretrained`
- SFT model: `Ethosoft/nedoqwen_0.8b_pretrained_sft`

## Important note

Large artifacts are intentionally not stored in this GitHub repository.

Excluded files include:

- model checkpoints
- binary token shards
- JSONL datasets
- Hugging Face upload folders
- logs
- cache files
- container images

Use the Hugging Face repositories above for released datasets and model checkpoints.

## Status

This is a research code release. The checkpoints are custom PyTorch checkpoints and are not yet Hugging Face Transformers-native.
