import json
from pathlib import Path
from datasets import load_dataset

OUT = Path("sft_data/tr_sft_hf_mix.jsonl")

DATASETS = [
    "NovusResearch/turkish_instructions",
    "SoAp9035/turkish_instructions",
    "Dbmaxwell/turkish-finance-instruction-dataset",
]

BAD_TOKENIZED_COLS = {
    "input_ids",
    "labels",
    "attention_mask",
    "token_type_ids",
    "position_ids",
}

def clean(x):
    if x is None:
        return ""
    return str(x).strip()

def normalize(name, row):
    if name == "NovusResearch/turkish_instructions":
        return {
            "instruction": clean(row.get("instruction")),
            "input": clean(row.get("input")),
            "output": clean(row.get("output")),
        }

    if name == "SoAp9035/turkish_instructions":
        return {
            "instruction": clean(row.get("user")),
            "input": "",
            "output": clean(row.get("assistant")),
        }

    if name == "Dbmaxwell/turkish-finance-instruction-dataset":
        return {
            "instruction": clean(row.get("instruction")),
            "input": clean(row.get("input")),
            "output": clean(row.get("output")),
        }

    return None

def good(obj):
    if not obj:
        return False
    ins = obj["instruction"]
    out = obj["output"]

    if len(ins) < 4 or len(out) < 4:
        return False
    if len(ins) > 4000 or len(out) > 8000:
        return False

    # Çok bariz chat-template artığı varsa temizliği bozmasın.
    bad_markers = ["<s>[INST]", "[/INST]", "</s>"]
    text = ins + "\n" + obj.get("input", "") + "\n" + out
    if any(m in text for m in bad_markers):
        return False

    return True

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    count = 0
    per_dataset = {}

    with OUT.open("w", encoding="utf-8") as w:
        for name in DATASETS:
            print("loading", name, flush=True)
            ds = load_dataset(name)
            per_dataset[name] = 0

            for split in ds.keys():
                cols = set(ds[split].column_names)
                if cols & BAD_TOKENIZED_COLS:
                    print("skip tokenized split", name, split, sorted(cols & BAD_TOKENIZED_COLS), flush=True)
                    continue

                for row in ds[split]:
                    obj = normalize(name, row)
                    if not good(obj):
                        continue

                    key = (
                        obj["instruction"].lower(),
                        obj["input"].lower(),
                        obj["output"].lower(),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    obj["source"] = name
                    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    count += 1
                    per_dataset[name] += 1

    print("wrote", count, "examples to", OUT)
    print("per_dataset:")
    for k, v in per_dataset.items():
        print(k, v)

if __name__ == "__main__":
    main()
