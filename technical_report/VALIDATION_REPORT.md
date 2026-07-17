# Canonical Technical Report Validation

**Validation date:** 2026-07-16
**PDF:** `NEDOQwen_Technical_Report.pdf`
**SHA-256:** `a222857ffd768d1c2a75b0abe1432ec72b3ba5401b0353a543d28110a91ab7f1`

## Build

- Compiler: Tectonic 0.16.9.
- Build status: successful.
- Pages: 10.
- Page size: US Letter, 612 x 792 pt.
- Undefined citations/references: none.
- Overfull/underfull boxes: none.
- PDF annotation-boundary warnings: none.
- Fonts: all reported fonts embedded.

## Content consistency

- Architecture matches the local canonical registry and inspected public model
  configs: 824,256,000 parameters; 24 layers; 65,536 vocabulary; 1,536 hidden
  size; 16 attention heads; 8 KV heads; 4,096 MLP; 1,024 context.
- Main benchmark values match the consolidated experiment registry:
  - TurkishMMLU-sub: 18.11% pre, 19.56% label-only, 21.00% full repair.
  - TUMLU-mini Turkish: 21.33% pre, 27.00% label-only, 32.22% full repair.
  - Qwen2.5 references: 19.33%/19.89% for 0.5B and 28.67%/42.00% for 1.5B.
- Internal diagnostic, tokenizer, repair-data, label-distribution, uncertainty,
  compute-cost, and subject-level values match the canonical local reports.
- Claims are bounded as diagnostic, protocol-specific, and non-deployment.

## Artifact consistency

- Public base revision inspected:
  `10f136642008dc50d4b07a37a0092ce447a2b53c`.
- Public SFT revision inspected:
  `038f096e8d7af59e867c7f1319f0f71cbc5b8406`.
- Tokenized corpus revision inspected:
  `4cb68d9d3f29cb9e31019c79803f98fd8fffe961`.
- SFT mixture revision inspected:
  `05e29ca6b7f40791026542322b94fc50a6b1437c`.
- Exact TRUBA checkpoint identities were verified:
  - step-9000 clean-SFT: `c51e805b0bac0864a78dc7f27917ba808943ea6bc561cb6f089d63553d22495e`;
  - answer-label-only repair-600: `906dee6423eedea930a8b51731b8db85bf98bd88b742e60424f5f3741f6a1eb1`;
  - full repair-v2-600: `931620bc9ad510805c7c73378269e17a74d34bf0ba003f6b4fe33d65d0f55fa2`.
- The exact evaluated 65K vocabulary was verified: 4,876,444 bytes, SHA-256
  `1d5cfe6dad3b6628964c92745935001307b6089b8f9b155567ed3ed1aae8b025`.
- Six sanitized prediction files contain 900 rows each and exactly reproduce
  all six NEDO benchmark accuracies in the manuscript.
- The overlap scan was rerun against both 900-row benchmarks and both repair
  datasets; all exact-field and >=20-character substring hit counts are zero.
- The supplementary validation script passes its accuracy, privacy, overlap,
  and checkpoint-manifest checks.
- `NEDOQwen_Supplementary_Materials.zip` contains 25 files total; its manifest
  inventories 24 payload files while excluding the inventory itself. The ZIP
  passes archive integrity testing and has SHA-256
  `9947ccc847efba7f015f867af86a320f281f22719bfbbfb1e05395a841f68e69`.
- The report explicitly distinguishes verified-but-unpublished experimental
  checkpoints from the two existing public checkpoints.

## Privacy and visual checks

- Extracted PDF text was scanned for local home paths, TRUBA paths/usernames,
  credentials, `.env`, and API-key patterns: no match.
- PDF metadata contains the confirmed five-person author list in the confirmed
  order.
- Pages 1, 4, 6, 8, and 10 were visually inspected at raster resolution.
- Title, abstract, tables, references, appendix, and artifact links are legible.
- Tables do not clip or cross page boundaries.
- Three data-driven figures were visually inspected both as standalone 300-DPI
  images and inside the compiled manuscript. Captions, labels, and values are
  legible, and no figure crosses a page boundary.

## Expected publication blockers

The five-person author list and order are confirmed. The PDF is not the final
public-upload version until:

1. affiliations, corresponding author, emails, and ORCIDs are confirmed;
2. the verified evaluated checkpoints and prepared artifact package are made public;
3. upstream data-license review is completed;
4. permanent artifact and preprint links replace pending statements.
