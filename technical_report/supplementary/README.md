# NEDOQwen Supplementary Materials

This package contains versioned checkpoint checksums, aggregate experiment summaries,
sanitized prediction rows, evaluation and training scripts, job records, tokenizer
statistics, qualitative diagnostics, and exact-overlap controls.

Benchmark question and answer-choice text is intentionally excluded. Checkpoint binaries
are not embedded because each is approximately 3.3 GB; the manifest identifies each exact
checkpoint by SHA-256. Private cluster paths and user identifiers are replaced by portable
placeholders.

Use manifest.json as the entry point. Prediction files contain only model identifiers,
row indices, subjects, labels, correctness, margins, and log probabilities.
