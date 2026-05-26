from datasets import load_dataset

DATASETS = [
    "merve/turkish_instructions",
    "SoAp9035/turkish_instructions",
    "NovusResearch/turkish_instructions",
    "erayalp/turkish-reasoning-instructions",
    "atasoglu/instruction-turkish",
    "Dbmaxwell/turkish-finance-instruction-dataset",
]

BAD_TOKENIZED_COLS = {
    "input_ids",
    "labels",
    "attention_mask",
    "token_type_ids",
    "position_ids",
}

for name in DATASETS:
    print("=" * 100)
    print("DATASET:", name)

    try:
        ds = load_dataset(name)
    except Exception as e:
        print("ERROR loading:", repr(e))
        continue

    print(ds)

    for split in ds.keys():
        cols = set(ds[split].column_names)
        print("split:", split)
        print("columns:", sorted(cols))

        if cols & BAD_TOKENIZED_COLS:
            print("STATUS: SKIP_TOKENIZED")
            print("tokenized columns:", sorted(cols & BAD_TOKENIZED_COLS))
        else:
            print("STATUS: RAW_TEXT_OR_CHECK_MANUALLY")

        for i in range(min(2, len(ds[split]))):
            row = ds[split][i]
            preview = {}
            for k, v in row.items():
                if isinstance(v, str):
                    preview[k] = v[:300]
                else:
                    preview[k] = str(type(v)) + " " + str(v)[:200]
            print("row", i, preview)
