#!/usr/bin/env python3
"""Recompute paired bootstrap intervals and exact sign-test p-values."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"
OUTPUT = ROOT / "paired_uncertainty_results.json"
SEED = 42
BOOTSTRAP_SAMPLES = 100_000
CHUNK_SIZE = 2_000

COMPARISONS = {
    "turkishmmlu_full_minus_pre": (
        "turkishmmlu_sub_full_repair.jsonl",
        "turkishmmlu_sub_pre.jsonl",
        [0.222, 5.556],
    ),
    "tumlu_full_minus_pre": (
        "tumlu_mini_full_repair.jsonl",
        "tumlu_mini_pre.jsonl",
        [7.778, 14.000],
    ),
    "turkishmmlu_label_only_minus_pre": (
        "turkishmmlu_sub_label_only.jsonl",
        "turkishmmlu_sub_pre.jsonl",
        [-0.333, 3.333],
    ),
    "tumlu_label_only_minus_pre": (
        "tumlu_mini_label_only.jsonl",
        "tumlu_mini_pre.jsonl",
        [3.444, 8.111],
    ),
    "turkishmmlu_full_minus_label_only": (
        "turkishmmlu_sub_full_repair.jsonl",
        "turkishmmlu_sub_label_only.jsonl",
        [-0.444, 3.333],
    ),
    "tumlu_full_minus_label_only": (
        "tumlu_mini_full_repair.jsonl",
        "tumlu_mini_label_only.jsonl",
        [2.889, 7.556],
    ),
}


def read_rows(filename: str) -> dict[int, dict]:
    rows = {}
    for line in (PRED / filename).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["idx"])] = row
    return rows


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    smaller = min(positive, negative)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_bootstrap(differences: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    values = []
    remaining = BOOTSTRAP_SAMPLES
    while remaining:
        size = min(CHUNK_SIZE, remaining)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        values.append(differences[indices].mean(axis=1) * 100)
        remaining -= size
    estimates = np.concatenate(values)
    return [round(float(x), 3) for x in np.percentile(estimates, [2.5, 97.5])]


def main() -> None:
    results = {
        "method": {
            "bootstrap": "paired percentile bootstrap over row-level correctness differences",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "seed": SEED,
            "p_value": "exact two-sided sign test over discordant pairs",
        },
        "comparisons": {},
        "note": (
            "The historical report used an earlier approximate resampling run. "
            "Small one-grid-step CI differences are expected; the fixed-seed rerun "
            "is included for deterministic reproduction."
        ),
    }
    for name, (left_file, right_file, historical_ci) in COMPARISONS.items():
        left = read_rows(left_file)
        right = read_rows(right_file)
        assert left.keys() == right.keys()
        differences = np.array(
            [int(left[idx]["is_correct"]) - int(right[idx]["is_correct"]) for idx in sorted(left)],
            dtype=np.int8,
        )
        positive = int(np.sum(differences == 1))
        negative = int(np.sum(differences == -1))
        results["comparisons"][name] = {
            "n": len(differences),
            "delta_percentage_points": round(float(differences.mean() * 100), 3),
            "positive_discordant_rows": positive,
            "negative_discordant_rows": negative,
            "ties": int(np.sum(differences == 0)),
            "fixed_seed_bootstrap_95_ci_pp": paired_bootstrap(differences),
            "historical_reported_95_ci_pp": historical_ci,
            "exact_two_sided_sign_p": exact_two_sided_sign_p(positive, negative),
        }
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
