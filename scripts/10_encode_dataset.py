import json
import argparse
from pathlib import Path
import numpy as np
from nedo_turkish_tokenizer import NedoTurkishTokenizer

def load_vocab(path):
    token_to_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            token_to_id[obj["token"]] = int(obj["id"])
    return token_to_id

def typed_token(t):
    typ = t.get("token_type", "UNK")
    tok = t.get("token", "")
    if not tok:
        return None
    return f"<{typ}>{tok}"

def iter_jsonl(path, text_key):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get(text_key)
            if text and len(text) > 50:
                yield text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=100)
    args = ap.parse_args()

    token_to_id = load_vocab(args.vocab)
    bos = token_to_id.get("<bos>", 1)
    eos = token_to_id.get("<eos>", 2)
    unk = token_to_id.get("<unk>", 3)

    tok = NedoTurkishTokenizer()

    train_ids = []
    val_ids = []

    docs = 0
    total = 0
    unk_count = 0

    for text in iter_jsonl(args.corpus, args.text_key):
        if args.max_docs and docs >= args.max_docs:
            break

        try:
            tokens = tok.tokenize(text)
        except Exception:
            continue

        ids = [bos]
        for t in tokens:
            if t.get("token_type") == "SPACE":
                continue
            key = typed_token(t)
            if key is None:
                continue
            idx = token_to_id.get(key, unk)
            if idx == unk:
                unk_count += 1
            total += 1
            ids.append(idx)
        ids.append(eos)

        if len(ids) < 16:
            continue

        if docs % args.val_every == 0:
            val_ids.extend(ids)
        else:
            train_ids.extend(ids)

        docs += 1
        if docs % 1000 == 0:
            print(f"docs={docs} train={len(train_ids)} val={len(val_ids)} unk_ratio={unk_count/max(total,1):.6f}", flush=True)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.uint32 if len(token_to_id) > 65535 else np.uint16
    np.array(train_ids, dtype=dtype).tofile(str(prefix) + "_train.bin")
    np.array(val_ids, dtype=dtype).tofile(str(prefix) + "_val.bin")

    meta = {
        "docs": docs,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "vocab_size": len(token_to_id),
        "dtype": str(dtype),
        "unk_count": unk_count,
        "total_tokens": total,
        "unk_ratio": unk_count / max(total, 1),
    }

    Path(str(prefix) + "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
