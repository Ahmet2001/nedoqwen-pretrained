# NEDOQwen Artifact Alignment

**Audit date:** 2026-07-16
**Canonical local evidence:** `../reports/CONSOLIDATED_EXPERIMENT_REGISTRY.md`
**Hub inspection method:** public `hf` CLI/API metadata and repository cards.

## 1. Checkpoint lineage

| Line | Checkpoint / stage | Evidence | Public artifact | Alignment |
|---|---|---|---|---|
| Public | base pretraining, step 5000 | Hub `metadata/model_info.json` | `Ethosoft/nedoqwen_0.8b_base_pretrained` | Exact public artifact |
| Public | clean-20K SFT from base-5000 | Hub SFT metadata | `Ethosoft/nedoqwen_0.8b_pretrained_sft` | Exact public artifact |
| Experimental | base step 9000 | local experiment registry and diagnostics | no public artifact | Parent lineage recorded; parent binary not separately hashed in this audit |
| Experimental | clean-20K SFT from step-9000 | TRUBA checkpoint verified | no public artifact | SHA-256 `c51e805b...2495e` |
| Experimental | answer-label-only repair-600 | TRUBA checkpoint verified | no public artifact | SHA-256 `906dee64...1eb1` |
| Experimental | full repair-v2-600 | TRUBA checkpoint verified | no public artifact | SHA-256 `931620bc...5fa2` |

### Non-negotiable wording

The benchmark tables in the technical report belong to the **evaluated
experimental line**, not to the two currently public checkpoints. Until the
exact evaluated checkpoints are released, the report must say so explicitly.

## 2. Architecture mapping

| Property | Local registry | Public base | Public SFT | Status |
|---|---:|---:|---:|---|
| Parameters | 824,256,000 | 824,256,000 | 824,256,000 | Match |
| Vocabulary | 65,536 | 65,536 | 65,536 | Match |
| Layers | 24 | 24 | 24 | Match |
| Hidden size | 1,536 | 1,536 | 1,536 | Match |
| Attention / KV heads | 16 / 8 | 16 / 8 | 16 / 8 | Match |
| MLP dimension | 4,096 | 4,096 | 4,096 | Match |
| Context length | 1,024 | 1,024 | 1,024 | Match |
| Position / norm / activation | RoPE / RMSNorm / SwiGLU | same | same | Match |
| Framework | custom PyTorch | custom PyTorch | custom PyTorch | Match |
| Transformers-native | no | no | no | Match |

## 3. Data mapping

| Data component | Public artifact | What is verified | Remaining gap |
|---|---|---|---|
| Tokenized pretraining corpus | `Ethosoft/nedo-turkish-65k-tokenized-60b` | 60,953,033,328 uint16 tokens; 32 shards; ODC-BY card | Exact upstream source snapshot and document-level provenance are unavailable |
| Base-5000 training exposure | base model metadata | 1,310,720,000 tokens seen | Must not be described as training on all 60.95B released tokens |
| Clean SFT data | `Ethosoft/nedo-turkish-sft-mixtures` | clean 20K file and upstream-source list are public | Byte-level identity with every local training copy not checked |
| Full repair mixture | none | 2,005 rows; construction script; SHA-256 `22b6a279...0306`; zero-overlap scan rerun | Licensing and public upload pending |
| Answer-label-only mixture | none | 250 rows; construction script; SHA-256 `e9be450d...cb35`; zero-overlap scan rerun | Licensing and public upload pending |

## 4. Result mapping

| Result family | Aggregate evidence | Row-level evidence in this workspace | Publicly reproducible now? |
|---|---|---|---|
| Architecture and base training metadata | local + Hub metadata | not applicable | Yes for metadata |
| Tokenizer scan / comparison | consolidated registry + TRUBA summaries | aggregate summaries in `supplementary/manifest.json` | Package ready; public upload pending |
| 48- and 220-prompt diagnostics | consolidated registry + TRUBA summaries | aggregate and qualitative records in manifest | Package ready; public upload pending |
| TurkishMMLU-sub results | consolidated registry + six TRUBA output sets | three sanitized 900-row files recovered | Exact checkpoints still need public upload |
| TUMLU-mini Turkish results | consolidated registry + six TRUBA output sets | three sanitized 900-row files recovered | Exact checkpoints still need public upload |
| Answer-label-only ablation | registry + exact checkpoint/results | checkpoint SHA and 1,800 sanitized rows recovered | Public upload pending |
| Prediction-label distributions | registry + row outputs | reproducible from sanitized rows | Public upload pending |
| Bootstrap / sign-flip estimates | historical audit + deterministic rerun | script and result JSON included | Public upload pending |
| Exact-overlap scan | rerun on TRUBA, 2026-07-16 | all question/choice equality and >=20-character substring counts are zero | Script method and counts packaged; public upload pending |
| Qualitative examples | TRUBA ablation/qualitative summary | complete 16-example summary in manifest | Public upload pending |

## 5. Current Hub inventory

### Models

- `Ethosoft/nedoqwen_0.8b_base_pretrained`
  Revision inspected: `10f136642008dc50d4b07a37a0092ce447a2b53c`
- `Ethosoft/nedoqwen_0.8b_pretrained_sft`
  Revision inspected: `038f096e8d7af59e867c7f1319f0f71cbc5b8406`

### Datasets

- `Ethosoft/nedo-turkish-65k-tokenized-60b`
  Revision inspected: `4cb68d9d3f29cb9e31019c79803f98fd8fffe961`
- `Ethosoft/nedo-turkish-sft-mixtures`
  Revision inspected: `05e29ca6b7f40791026542322b94fc50a6b1437c`

### Tokenizer

- Tokenizer code and the 65K vocabulary are bundled in both model repositories.
- The exact evaluated `vocab_65536.jsonl` file was verified on TRUBA: 4,876,444
  bytes, SHA-256
  `1d5cfe6dad3b6628964c92745935001307b6089b8f9b155567ed3ed1aae8b025`.
- Byte identity with the separate tokenizer repository's current revision
  should be checked before claiming that repository is the immutable evaluated
  tokenizer release.

## 6. Required release artifacts

Publish these before claiming full public reproducibility:

1. `Ethosoft/nedoqwen_0.8b_step9000_sft_clean20k` or an equivalently clear
   versioned repository for the exact pre-repair evaluated checkpoint.
2. `Ethosoft/nedoqwen_0.8b_repair_v2_600` for the exact full-repair checkpoint.
3. Upload the prepared `NEDOQwen_Supplementary_Materials.zip` to a paper or
   artifact repository; it already contains configs, scripts, aggregate and
   sanitized row-level outputs, overlap results, and checksums.
4. A release manifest connecting every paper table row to a repository,
   revision SHA, file path, and checksum.
