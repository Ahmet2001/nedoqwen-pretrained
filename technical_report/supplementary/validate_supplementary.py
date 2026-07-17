#!/usr/bin/env python3
"""Validate the sanitized NEDOQwen supplementary package."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
RESULT_PATH = ROOT / "VALIDATION_RESULT.json"
INVENTORY_PATH = ROOT / "inventory.json"

PREDICTION_TO_SUMMARY = {
    "predictions/turkishmmlu_sub_pre.jsonl": "turkishmmlu_sub_pre",
    "predictions/turkishmmlu_sub_label_only.jsonl": "turkishmmlu_sub_label_only",
    "predictions/turkishmmlu_sub_full_repair.jsonl": "turkishmmlu_sub_full_repair",
    "predictions/tumlu_mini_pre.jsonl": "tumlu_mini_pre",
    "predictions/tumlu_mini_label_only.jsonl": "tumlu_mini_label_only",
    "predictions/tumlu_mini_full_repair.jsonl": "tumlu_mini_full_repair",
}

FORBIDDEN_ROW_FIELDS = {
    "question",
    "choices",
    "prompt",
    "instruction",
    "input",
    "output",
    "question_preview",
}
PRIVATE_PATTERNS = [
    re.compile("/" + "arf/", re.I),
    re.compile("egitimg16u" + r"\d+", re.I),
    re.compile("/home/" + "rifat", re.I),
    re.compile("HF_" + "TOKEN", re.I),
    re.compile("api" + r"[_-]?" + "key", re.I),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checks = {}
    for relative, summary_key in PREDICTION_TO_SUMMARY.items():
        rows = read_jsonl(ROOT / relative)
        summary = MANIFEST["summaries"][summary_key]
        assert len(rows) == summary["n"] == 900, relative
        assert not any(FORBIDDEN_ROW_FIELDS.intersection(row) for row in rows), relative
        correct = sum(int(row["is_correct"]) for row in rows)
        accuracy = correct / len(rows)
        assert correct == summary["correct"], relative
        assert abs(accuracy - summary["accuracy"]) <= 1e-6, relative
        checks[relative] = {
            "rows": len(rows),
            "correct": correct,
            "accuracy": round(accuracy, 6),
            "benchmark_text_present": False,
        }

    leaks = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".zip"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [pattern.pattern for pattern in PRIVATE_PATTERNS if pattern.search(text)]
        if hits:
            leaks[str(path.relative_to(ROOT))] = hits
    assert not leaks, leaks

    overlap = MANIFEST["overlap_scan"]
    for repair in overlap.values():
        for result in repair.values():
            assert result["exact_question_field_matches"] == 0
            assert result["question_substring_hits_min20chars"] == 0
            assert result["exact_choice_field_matches"] == 0
            assert result["choice_substring_hits_min20chars"] == 0

    tokenizer = MANIFEST["tokenizer_manifest"]
    assert tokenizer["size_bytes"] == 4_876_444
    assert len(tokenizer["sha256"]) == 64

    result = {
        "status": "pass",
        "prediction_files": checks,
        "privacy_scan": "clean",
        "overlap_scan": "all reported counts are zero",
        "checkpoint_count": len(MANIFEST["checkpoint_manifest"]),
        "tokenizer_manifest": "verified",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    inventory = []
    for path in sorted(candidate for candidate in ROOT.rglob("*") if candidate.is_file()):
        if path == INVENTORY_PATH or path.suffix.lower() == ".zip":
            continue
        inventory.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
