# NEDOQwen / NEDO Turkish SLM

This repository contains code and reproducibility scripts for the NEDO Turkish SLM project.

The large datasets and model checkpoints are released on Hugging Face. This GitHub repository is intended for code, scripts, and documentation only.

## Contents

- `scripts/`: training, encoding, SFT, sampling, inspection, and upload scripts
- `nedo_turkish_tokenizer/`: tokenizer Python package used by the sampling scripts
- `docs/`: usage and artifact documentation
- `technical_report/`: canonical report, reproducible figures, sanitized
  predictions, evaluation scripts, checkpoint hashes, and validation tools
- `requirements.txt`: minimal Python dependencies

## Technical report and reproducibility package

The canonical report is available as
[`technical_report/NEDOQwen_Technical_Report.pdf`](technical_report/NEDOQwen_Technical_Report.pdf).
Its supplementary package deliberately excludes benchmark question/choice text
and model binaries. Exact experimental checkpoints and the tokenizer vocabulary
are identified by SHA-256 in
[`technical_report/SURVIVAL_MANIFEST.md`](technical_report/SURVIVAL_MANIFEST.md)
and [`technical_report/supplementary/manifest.json`](technical_report/supplementary/manifest.json).

To validate the supplementary package locally:

```bash
python3 technical_report/supplementary/validate_supplementary.py
```

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

The headline diagnosis-and-repair results in the technical report use a later
step-9000 experimental lineage. They must not be attributed to the currently
released base-5000 or SFT-5000 checkpoints. The exact step-9000 checkpoint
repositories will be linked here after their Hub release and restore check.

## Status

This is a research code release. The checkpoints are custom PyTorch checkpoints and are not yet Hugging Face Transformers-native.
