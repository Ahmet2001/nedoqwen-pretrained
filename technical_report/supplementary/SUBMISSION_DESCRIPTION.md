# Supplementary Material Description

This package provides the reproducibility materials for the NEDOQwen technical
report. It contains exact SHA-256 identifiers for the three evaluated
checkpoints, aggregate experiment summaries, six sanitized 900-row prediction
files, tokenizer statistics, label-distribution and qualitative diagnostics,
completed job records, training/evaluation scripts, and an independently
rerun exact-overlap analysis for the full and answer-label-only repair sets.

Benchmark questions and answer choices are deliberately excluded. Prediction
files contain only row identifiers, subjects, gold/predicted labels,
correctness, margins, and token/log-probability diagnostics. Private cluster
paths and user identifiers have been replaced with portable placeholders.

The checkpoint binaries are not embedded because each is approximately 3.3
GB. `manifest.json` records the exact checkpoint size, modification time, and
SHA-256 checksum required to identify the evaluated weights.
