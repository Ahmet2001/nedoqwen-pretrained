import os
import json
import time
import argparse
import shutil
import multiprocessing as mp
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
    tok = t.get("token", "")
    typ = t.get("token_type", "UNK")
    if tok == "":
        return None
    return f"<{typ}>{tok}"


def flush_buffer(buf, fh, dtype):
    if buf:
        np.asarray(buf, dtype=dtype).tofile(fh)
        buf.clear()


def process_chunk(args):
    (
        worker_id,
        corpus_path,
        start,
        end,
        vocab_path,
        out_dir,
        text_key,
        val_every,
        flush_tokens,
        dtype_name,
    ) = args

    token_to_id = load_vocab(vocab_path)
    bos = token_to_id.get("<bos>", 1)
    eos = token_to_id.get("<eos>", 2)
    unk = token_to_id.get("<unk>", 3)

    dtype = np.uint16 if dtype_name == "uint16" else np.uint32
    tok = NedoTurkishTokenizer()

    train_path = Path(out_dir) / f"train_part_{worker_id:04d}.bin"
    val_path = Path(out_dir) / f"val_part_{worker_id:04d}.bin"

    docs = 0
    kept = 0
    json_errors = 0
    tok_errors = 0
    total_tokens = 0
    unk_tokens = 0

    train_buf = []
    val_buf = []

    t0 = time.time()

    with open(corpus_path, "rb") as fb, \
         open(train_path, "wb") as train_fh, \
         open(val_path, "wb") as val_fh:

        fb.seek(start)
        if start != 0:
            fb.readline()

        while True:
            pos = fb.tell()
            if pos >= end:
                break

            line = fb.readline()
            if not line:
                break

            try:
                obj = json.loads(line.decode("utf-8", errors="ignore"))
            except Exception:
                json_errors += 1
                continue

            text = obj.get(text_key)
            if not text or len(text) < 50:
                continue

            try:
                tokens = tok.tokenize(text)
            except Exception:
                tok_errors += 1
                continue

            ids = [bos]

            for t in tokens:
                key = typed_token(t)
                if key is None:
                    continue
                idx = token_to_id.get(key, unk)
                if idx == unk:
                    unk_tokens += 1
                total_tokens += 1
                ids.append(idx)

            ids.append(eos)

            if len(ids) < 16:
                continue

            # Worker-local validation split. Full data için yeterli.
            if docs % val_every == 0:
                val_buf.extend(ids)
                if len(val_buf) >= flush_tokens:
                    flush_buffer(val_buf, val_fh, dtype)
            else:
                train_buf.extend(ids)
                if len(train_buf) >= flush_tokens:
                    flush_buffer(train_buf, train_fh, dtype)

            docs += 1
            kept += 1

            if docs % 10000 == 0:
                print(
                    f"[worker {worker_id:04d}] docs={docs} "
                    f"tokens={total_tokens} unk_ratio={unk_tokens/max(total_tokens,1):.6f}",
                    flush=True,
                )

        flush_buffer(train_buf, train_fh, dtype)
        flush_buffer(val_buf, val_fh, dtype)

    return {
        "worker": worker_id,
        "docs": docs,
        "kept": kept,
        "json_errors": json_errors,
        "tok_errors": tok_errors,
        "tokens": total_tokens,
        "unk_tokens": unk_tokens,
        "elapsed_sec": time.time() - t0,
        "train_part": str(train_path),
        "val_part": str(val_path),
    }


def concat_parts(parts, final_path):
    with open(final_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024 * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--flush-tokens", type=int, default=2_000_000)
    args = ap.parse_args()

    token_to_id = load_vocab(args.vocab)
    vocab_size = len(token_to_id)

    # vocab_65536 için uint16 yeterli: id aralığı 0..65535
    dtype_name = "uint16" if vocab_size <= 65536 else "uint32"

    corpus_size = os.path.getsize(args.corpus)
    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent / (out_prefix.name + "_parts")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = corpus_size // args.workers
    job_args = []

    for wid in range(args.workers):
        start = wid * chunk_size
        end = corpus_size if wid == args.workers - 1 else (wid + 1) * chunk_size
        job_args.append((
            wid,
            args.corpus,
            start,
            end,
            args.vocab,
            str(out_dir),
            args.text_key,
            args.val_every,
            args.flush_tokens,
            dtype_name,
        ))

    print("=" * 80)
    print("FULL ENCODE START")
    print("corpus:", args.corpus)
    print("corpus_size_gb:", corpus_size / 1e9)
    print("vocab:", args.vocab)
    print("vocab_size:", vocab_size)
    print("dtype:", dtype_name)
    print("workers:", args.workers)
    print("out_prefix:", str(out_prefix))
    print("=" * 80, flush=True)

    t0 = time.time()

    with mp.Pool(args.workers) as pool:
        stats = pool.map(process_chunk, job_args)

    train_parts = [s["train_part"] for s in sorted(stats, key=lambda x: x["worker"])]
    val_parts = [s["val_part"] for s in sorted(stats, key=lambda x: x["worker"])]

    train_final = str(out_prefix) + "_train.bin"
    val_final = str(out_prefix) + "_val.bin"

    print("Concatenating train parts...", flush=True)
    concat_parts(train_parts, train_final)

    print("Concatenating val parts...", flush=True)
    concat_parts(val_parts, val_final)

    total_docs = sum(s["docs"] for s in stats)
    total_tokens = sum(s["tokens"] for s in stats)
    total_unk = sum(s["unk_tokens"] for s in stats)
    total_json_errors = sum(s["json_errors"] for s in stats)
    total_tok_errors = sum(s["tok_errors"] for s in stats)

    meta = {
        "corpus": args.corpus,
        "vocab": args.vocab,
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "workers": args.workers,
        "docs": total_docs,
        "tokens": total_tokens,
        "unk_tokens": total_unk,
        "unk_ratio": total_unk / max(total_tokens, 1),
        "json_errors": total_json_errors,
        "tokenize_errors": total_tok_errors,
        "elapsed_sec": time.time() - t0,
        "train_bin": train_final,
        "val_bin": val_final,
        "worker_stats": stats,
    }

    meta_path = str(out_prefix) + "_meta.json"
    Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in meta.items() if k != "worker_stats"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
