import sys
import json
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/tokenizer/NedoTurkishTokenizer")
from nedo_turkish_tokenizer import NedoTurkishTokenizer


def import_train_module():
    path = "/workspace/nedo_slm/scripts/20_train_qwen_style.py"
    spec = importlib.util.spec_from_file_location("train_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vocab_jsonl(path):
    token_to_id = {}
    id_to_token = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tok = obj["token"]
            idx = int(obj["id"])
            token_to_id[tok] = idx
            id_to_token[idx] = tok
    return token_to_id, id_to_token


def typed_token(t):
    tok = t.get("token", "")
    typ = t.get("token_type", "UNK")
    if not tok:
        return None
    return f"<{typ}>{tok}"


def encode_prompt(text, token_to_id):
    tok = NedoTurkishTokenizer()
    bos = token_to_id.get("<bos>", 1)
    unk = token_to_id.get("<unk>", 3)

    ids = [bos]
    for t in tok.tokenize(text):
        key = typed_token(t)
        if key is None:
            continue
        ids.append(token_to_id.get(key, unk))
    return ids


def split_typed_token(token):
    if token.startswith("<") and ">" in token:
        typ, surface = token[1:].split(">", 1)
        return typ, surface
    return "UNK", token


def rough_decode(ids, id_to_token):
    out = []
    no_space_types = {"SUFFIX", "PUNCT"}

    for idx in ids:
        token = id_to_token.get(int(idx), "<unk>")

        if token in {"<pad>", "<bos>", "<eos>"}:
            continue

        typ, surface = split_typed_token(token)

        if typ == "SUFFIX":
            out.append(surface)
        elif typ == "PUNCT":
            out.append(surface)
        else:
            if out and not out[-1].endswith((" ", "\n", "'", "’", "(", "[", "{", "/")):
                out.append(" ")
            out.append(surface)

    return "".join(out)


def top_p_filtering(logits, top_p=0.95):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)

    mask = cumprobs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_logits[mask] = -float("inf")

    filtered = torch.full_like(logits, -float("inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


@torch.no_grad()
def ban_repeated_ngrams(logits, generated_ids, no_repeat_ngram_size: int):
    """
    Prevents generating a token that would create an already-seen n-gram.
    Works for batch size 1 or more.
    """
    if no_repeat_ngram_size is None or no_repeat_ngram_size <= 0:
        return logits

    n = no_repeat_ngram_size
    seq_len = generated_ids.size(1)

    if seq_len + 1 < n:
        return logits

    logits = logits.clone()

    for b in range(generated_ids.size(0)):
        seq = generated_ids[b].tolist()
        prefix = tuple(seq[-(n - 1):])

        banned = set()
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i + n])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])

        if banned:
            logits[b, list(banned)] = -float("inf")

    return logits


def apply_repetition_penalty(logits, generated_ids, penalty: float):
    """
    logits: [batch, vocab]
    generated_ids: [batch, seq]
    CTRL-style repetition penalty.
    If a token appeared before, positive logits are divided by penalty,
    negative logits are multiplied by penalty.
    """
    if penalty is None or penalty == 1.0:
        return logits

    logits = logits.clone()

    for b in range(logits.size(0)):
        seen = torch.unique(generated_ids[b])
        for token_id in seen:
            token_id = int(token_id.item())
            if logits[b, token_id] < 0:
                logits[b, token_id] *= penalty
            else:
                logits[b, token_id] /= penalty

    return logits


def generate(model, input_ids, max_new_tokens, temperature, top_p, eos_id, block_size, device, repetition_penalty, no_repeat_ngram_size):
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        ctx = ids[:, -block_size:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :]

        logits = apply_repetition_penalty(
            logits,
            input_ids,
            repetition_penalty,
        )

        logits = logits / max(temperature, 1e-6)

        logits = ban_repeated_ngrams(
            logits,
            input_ids,
            no_repeat_ngram_size,
        )

        if top_p < 1.0:
            logits = top_p_filtering(logits, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)

        if int(next_id.item()) == eos_id:
            break

    return ids[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    args = ap.parse_args()
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        raise ValueError("Either --prompt or --prompt-file must be provided")


    train_mod = import_train_module()
    token_to_id, id_to_token = load_vocab_jsonl(args.vocab)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_mod.QwenStyleLM(
        vocab_size=65536,
        dim=cfg["dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        mlp_dim=cfg["mlp_dim"],
        block_size=cfg["block_size"],
        tie_embeddings=cfg.get("tie_embeddings", False),
    )

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()

    input_ids = encode_prompt(prompt, token_to_id)
    eos_id = token_to_id.get("<eos>", 2)

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_id=eos_id,
        block_size=cfg["block_size"],
        device=device,
    )

    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("=" * 80)
    print("RAW IDS:", output_ids[:80], "...")
    print("=" * 80)
    print("DECODED:")
    print(rough_decode(output_ids, id_to_token))
    print("=" * 80)


if __name__ == "__main__":
    main()
