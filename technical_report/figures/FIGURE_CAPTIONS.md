# Figure Captions

## Figure 1 — Benchmark accuracy

NEDOQwen diagnosis-and-repair results under the common zero-shot answer-label
log-probability diagnostic protocol. Full repair improves TurkishMMLU-sub from
18.11% to 21.00% and TUMLU-mini Turkish from 21.33% to 32.22%. The model remains
below Qwen2.5-1.5B-Instruct, and the TurkishMMLU-sub gain is small. Dashed lines
show the chance reference for each benchmark.

## Figure 2 — Prediction-label distributions

Prediction-label distributions before repair, after answer-label-only repair,
and after full repair. Repair reduces extreme C-only collapse but leaves strong
C/D concentration. The shift demonstrates that increased accuracy does not
eliminate answer-label bias. TUMLU-mini Turkish has four answer choices; E is
therefore shown only in the shared legend.

## Figure 3 — Tokenizer coverage

Tokenizer compactness and unknown-token rate on TurkishMMLU-sub question and
choice text. The Turkish-aware NEDO 65K tokenizer is less compact than Qwen2.5
and Trendyol-LLM on this slice and is the only compared tokenizer with a
nonzero UNK rate. These measurements are coverage diagnostics and do not
establish a causal relationship with model accuracy.
