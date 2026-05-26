import json
import random
from pathlib import Path
from datasets import load_dataset

OUT = Path("sft_data/tr_sft_clean_20k.jsonl")
random.seed(42)

def clean(x):
    if x is None:
        return ""
    return str(x).strip()

def good(ins, out):
    if len(ins) < 5 or len(out) < 10:
        return False
    if len(ins) > 1000 or len(out) > 2500:
        return False
    bad = ["<s>[INST]", "[/INST]", "</s>", "###"]
    text = ins + "\n" + out
    if any(b in text for b in bad):
        return False
    return True

examples = []

# 1) SoAp user/assistant
ds = load_dataset("SoAp9035/turkish_instructions")
for row in ds["train"]:
    ins = clean(row.get("user"))
    out = clean(row.get("assistant"))
    if good(ins, out):
        examples.append({
            "instruction": ins,
            "input": "",
            "output": out,
            "source": "SoAp9035/turkish_instructions",
        })

# 2) Finance
ds = load_dataset("Dbmaxwell/turkish-finance-instruction-dataset")
for row in ds["train"]:
    ins = clean(row.get("instruction"))
    inp = clean(row.get("input"))
    out = clean(row.get("output"))
    if good(ins + "\n" + inp, out):
        examples.append({
            "instruction": ins,
            "input": inp,
            "output": out,
            "source": "Dbmaxwell/turkish-finance-instruction-dataset",
        })

# dedup
seen = set()
dedup = []
for e in examples:
    k = (e["instruction"].lower(), e["input"].lower(), e["output"].lower())
    if k in seen:
        continue
    seen.add(k)
    dedup.append(e)

random.shuffle(dedup)

# maksimum 20K
dedup = dedup[:20000]

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for e in dedup:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print("wrote", len(dedup), "examples to", OUT)
print("sources:")
from collections import Counter
print(Counter(e["source"] for e in dedup))
