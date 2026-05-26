import os
import sys
import json
import math
import time
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


def encode_text(text, tokenizer, token_to_id, add_bos=False):
    bos = token_to_id.get("<bos>", 1)
    unk = token_to_id.get("<unk>", 3)

    ids = []
    if add_bos:
        ids.append(bos)

    for t in tokenizer.tokenize(text):
        key = typed_token(t)
        if key is None:
            continue
        ids.append(token_to_id.get(key, unk))

    return ids


def build_prompt(instruction, input_text=""):
    if input_text and input_text.strip():
        return (
            "Kullanıcı talimatı:\n"
            f"{instruction}\n\n"
            "Ek bilgi:\n"
            f"{input_text}\n\n"
            "Asistan cevabı:\n"
        )
    return (
        "Kullanıcı talimatı:\n"
        f"{instruction}\n\n"
        "Asistan cevabı:\n"
    )


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, token_to_id, block_size):
        self.examples = []
        self.tokenizer = tokenizer
        self.token_to_id = token_to_id
        self.block_size = block_size
        self.eos = token_to_id.get("<eos>", token_to_id.get("</s>", 2))
        self.bos = token_to_id.get("<bos>", 1)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                instruction = obj.get("instruction", "")
                input_text = obj.get("input", "")
                output = obj.get("output", "")

                prompt = build_prompt(instruction, input_text)
                prompt_ids = encode_text(prompt, tokenizer, token_to_id, add_bos=True)
                answer_ids = encode_text(output, tokenizer, token_to_id, add_bos=False) + [self.eos]

                ids = prompt_ids + answer_ids
                if len(ids) < 4:
                    continue

                ids = ids[: block_size + 1]

                # x predicts y. We mask prompt targets, train mostly answer part.
                x = ids[:-1]
                y = ids[1:]

                prompt_len = min(len(prompt_ids), len(y))
                labels = y[:]
                for i in range(max(0, prompt_len - 1)):
                    labels[i] = -100

                self.examples.append((x, labels))

                if len(self.examples) % 1000 == 0:
                    print(f"tokenized_sft_examples={len(self.examples)}", flush=True)

        print(f"finished_tokenizing_sft_examples={len(self.examples)}", flush=True)

        if not self.examples:
            raise RuntimeError("No SFT examples loaded.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id, block_size):
    max_len = min(max(len(x) for x, _ in batch), block_size)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (x, y) in enumerate(batch):
        x = x[:max_len]
        y = y[:max_len]
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        labels[i, :len(y)] = torch.tensor(y, dtype=torch.long)

    return input_ids, labels


def build_model(train_mod, ckpt, device):
    """
    Build the same qwen08b architecture used in pretraining.
    Our QwenStyleLM expects explicit constructor args, not a config object.
    """
    if not hasattr(train_mod, "QwenStyleLM"):
        raise RuntimeError("QwenStyleLM not found in 20_train_qwen_style.py")

    model = train_mod.QwenStyleLM(
        vocab_size=65536,
        dim=1536,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        mlp_dim=4096,
        block_size=1024,
        tie_embeddings=False,
    )

    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)

    return model, int(state.get("step", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save-interval", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    token_to_id, _ = load_vocab_jsonl(args.vocab)
    tokenizer = NedoTurkishTokenizer()
    pad_id = token_to_id.get("<pad>", token_to_id.get("<unk>", 3))

    train_mod = import_train_module()
    model, base_step = build_model(train_mod, args.ckpt, device)
    model.train()

    ds = SFTDataset(args.data, tokenizer, token_to_id, args.block_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id, args.block_size),
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    print(f"device={device}", flush=True)
    print(f"base_step={base_step}", flush=True)
    print(f"sft_examples={len(ds)}", flush=True)

    step = 0
    tokens_seen = 0
    t0 = time.time()

    while step < args.max_steps:
        for input_ids, labels in dl:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits, _ = model(input_ids, None)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = loss / args.grad_accum
            loss.backward()

            tokens_seen += int((labels != -100).sum().item())

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1

            if step % 10 == 0:
                dt = max(time.time() - t0, 1e-6)
                print(
                    f"sft_step={step} loss={loss.item() * args.grad_accum:.4f} "
                    f"answer_tokens={tokens_seen} tok/s={tokens_seen/dt:.0f}",
                    flush=True,
                )

            if step % args.save_interval == 0:
                out = Path(args.out_dir) / f"sft_step_{step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "base_ckpt": args.ckpt,
                        "base_step": base_step,
                        "sft_data": args.data,
                    },
                    out,
                )
                print(f"saved {out}", flush=True)

            if step >= args.max_steps:
                break

    final = Path(args.out_dir) / "sft_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "base_ckpt": args.ckpt,
            "base_step": base_step,
            "sft_data": args.data,
        },
        final,
    )
    print(f"saved {final}", flush=True)
    print("sft done", flush=True)


if __name__ == "__main__":
    main()
