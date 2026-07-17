# NEDOQwen Survival Manifest

This manifest identifies the minimum private binary lineage that must remain
recoverable before the original TRUBA project tree can be removed. Large files
are intentionally not committed to GitHub.

## Exact checkpoints

| Role | Size (bytes) | SHA-256 |
|---|---:|---|
| Base step-9000 parent | 1,648,597,040 | `ac35cc6c3b887634bcf0437ec9f4a0136dd6d65771b3f4e9f5181d906a2f214c` |
| Step-9000 clean-SFT final | 3,297,107,915 | `c51e805b0bac0864a78dc7f27917ba808943ea6bc561cb6f089d63553d22495e` |
| Answer-label-only repair-600 final | 3,297,107,915 | `906dee6423eedea930a8b51731b8db85bf98bd88b742e60424f5f3741f6a1eb1` |
| Full repair-v2-600 final | 3,297,107,915 | `931620bc9ad510805c7c73378269e17a74d34bf0ba003f6b4fe33d65d0f55fa2` |

## Exact tokenizer vocabulary

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `vocab_65536.jsonl` | 4,876,444 | `1d5cfe6dad3b6628964c92745935001307b6089b8f9b155567ed3ed1aae8b025` |

## Repair data

The private survival archive also retains the exact SFT and repair JSONL files,
including:

- `tr_repair_mix_v2.jsonl`: `22b6a2794db8b73b668d188ebe8a057e69408af4b22b9dd656de9fac399d0306`
- `tr_repair_answer_label_only_v1.jsonl`: `e9be450d92b6f29199fe54fd4c43a7b9b706055253360250b124726dfb47cb35`

## Verified private archive

As of 2026-07-17, these binaries, all SFT/repair JSONLs, the base run config,
and a source/evaluation/log snapshot were copied to the TRUBA survival archive.
Every archived file passed `sha256sum -c SHA256SUMS`.

The source TRUBA project must not be deleted until:

1. the private large-file archive is uploaded to durable remote storage;
2. every remote file matches this manifest and the private `SHA256SUMS` file;
3. a clean restore test succeeds; and
4. the exact deletion manifest receives explicit approval.
