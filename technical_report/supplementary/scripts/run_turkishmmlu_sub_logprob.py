import argparse, csv, importlib.util, json, math, os, re, sys, time
from pathlib import Path
from collections import defaultdict

import torch

ABC = "ABCDE"


def softmax_logprob(logits, target_id):
    return float(torch.log_softmax(logits.float(), dim=-1)[int(target_id)].item())


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_prompt(row):
    q = str(row["question"]).strip()
    choices = row["choices"]
    lines = [
        "Aşağıdaki çoktan seçmeli soruyu cevapla.",
        "Sadece doğru seçeneğin harfini yaz: A, B, C, D veya E.",
        "",
        f"Soru: {q}",
    ]
    for i, ch in enumerate(choices):
        lines.append(f"{ABC[i]}) {str(ch).strip()}")
    lines.append("")
    lines.append("Cevap:")
    return "\n".join(lines) + " "


def norm_subject(s):
    return str(s).strip().replace(" ", "_").lower()


# ---------------- NEDO custom model/tokenizer ----------------
def import_train_module():
    path = "${CONTAINER_PROJECT}/scripts/20_train_qwen_style.py"
    spec = importlib.util.spec_from_file_location("train_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vocab_jsonl(path):
    token_to_id, id_to_token = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            token_to_id[obj["token"]] = int(obj["id"])
            id_to_token[int(obj["id"])] = obj["token"]
    return token_to_id, id_to_token


def typed_token(t):
    tok = t.get("token", "")
    typ = t.get("token_type", "UNK")
    if typ == "SPACE":
        return "<SPACE>"
    if not tok:
        return None
    return f"<{typ}>{tok}"


def nedo_encode(text, tokenizer, token_to_id, add_bos=True):
    bos = token_to_id.get("<bos>", 1)
    unk = token_to_id.get("<unk>", 3)
    ids = [bos] if add_bos else []
    for t in tokenizer.tokenize(text):
        key = typed_token(t)
        if key is None:
            continue
        ids.append(token_to_id.get(key, unk))
    return ids


def build_nedo_model(ckpt_path, device):
    train_mod = import_train_module()
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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        step = ckpt.get("step", "unknown")
    else:
        state = ckpt
        step = "model_only"
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARN NEDO load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device)
    model.eval()
    return model, step


@torch.no_grad()
def nedo_continuation_logprob(model, prompt_ids, cont_ids, device, block_size=1024):
    if not cont_ids:
        return -1e30, []
    ids = list(prompt_ids) + list(cont_ids)
    # Keep the full continuation and enough prompt context.
    if len(ids) > block_size:
        overflow = len(ids) - block_size
        ids = ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)
    if prompt_len <= 0:
        return -1e30, []
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x, None)
    token_lps = []
    for pos in range(prompt_len, len(ids)):
        lp = softmax_logprob(logits[0, pos - 1], ids[pos])
        token_lps.append(lp)
    return sum(token_lps) / max(1, len(token_lps)), token_lps


def run_nedo(rows, out_dir, ckpt_path, vocab_path, limit=None):
    sys.path.insert(0, "${CONTAINER_TOKENIZER}")
    from nedo_turkish_tokenizer import NedoTurkishTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("NEDO device", device, flush=True)
    token_to_id, _ = load_vocab_jsonl(vocab_path)
    tokenizer = NedoTurkishTokenizer()
    model, step = build_nedo_model(ckpt_path, device)
    eval_rows = rows[:limit] if limit else rows
    return score_rows("new_step9000_sft_clean20k", step, eval_rows, out_dir, lambda text, add_bos=True: nedo_encode(text, tokenizer, token_to_id, add_bos), lambda p,c: nedo_continuation_logprob(model, p, c, device))


# ---------------- HF/Qwen ----------------
def run_hf(rows, out_dir, model_id, limit=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("HF device", device, "model", model_id, flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    def enc(text, add_bos=True):
        # Plain log-prob scoring, no chat template; add_special_tokens=False keeps prompt length exact.
        return tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def cont_lp(prompt_ids, cont_ids):
        if not cont_ids:
            return -1e30, []
        ids = prompt_ids + cont_ids
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=next(model.parameters()).device)
        logits = model(x).logits
        prompt_len = len(prompt_ids)
        token_lps=[]
        for pos in range(prompt_len, len(ids)):
            token_lps.append(softmax_logprob(logits[0, pos-1], ids[pos]))
        return sum(token_lps) / max(1, len(token_lps)), token_lps

    name = model_id.replace("/", "__")
    eval_rows = rows[:limit] if limit else rows
    return score_rows(name, "hf", eval_rows, out_dir, enc, cont_lp)


# ---------------- shared scoring/reporting ----------------
def score_rows(model_name, model_step, rows, out_dir, encode_fn, cont_lp_fn):
    labels = list(ABC)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{model_name}_turkishmmlu_sub_logprob_outputs.jsonl"
    csv_path = out_dir / f"{model_name}_turkishmmlu_sub_logprob_outputs.csv"
    fields = ["model","model_step","idx","subject","difficulty","grade","correct","pred","is_correct","margin","lp_A","lp_B","lp_C","lp_D","lp_E","question_preview"]
    correct = 0
    by_subject = defaultdict(lambda: {"n":0,"correct":0})
    by_diff = defaultdict(lambda: {"n":0,"correct":0})
    t0 = time.time()
    with open(raw_path, "w", encoding="utf-8") as jf, open(csv_path, "w", encoding="utf-8", newline="") as cf:
        wr = csv.DictWriter(cf, fieldnames=fields)
        wr.writeheader()
        for idx, row in enumerate(rows):
            prompt = format_prompt(row)
            prompt_ids = encode_fn(prompt, add_bos=True)
            lps = {}
            token_lps = {}
            for lab in labels:
                cont_ids = encode_fn(lab, add_bos=False)
                lp, parts = cont_lp_fn(prompt_ids, cont_ids)
                lps[lab] = lp
                token_lps[lab] = parts
            pred = max(labels, key=lambda x: lps[x])
            correct_lab = ABC[int(row["answer"])]
            ok = int(pred == correct_lab)
            correct += ok
            subj = str(row.get("subject", "unknown"))
            md = row.get("metadata") or {}
            diff = str(md.get("difficulty", "unknown"))
            grade = str(md.get("grade", "unknown"))
            by_subject[subj]["n"] += 1; by_subject[subj]["correct"] += ok
            by_diff[diff]["n"] += 1; by_diff[diff]["correct"] += ok
            sorted_lps = sorted(lps.values(), reverse=True)
            margin = sorted_lps[0] - sorted_lps[1] if len(sorted_lps) > 1 else 0.0
            rec = {
                "model": model_name,
                "model_step": model_step,
                "idx": idx,
                "subject": subj,
                "difficulty": diff,
                "grade": grade,
                "correct": correct_lab,
                "pred": pred,
                "is_correct": ok,
                "margin": round(margin, 6),
                "logprobs": {k: round(v, 6) for k,v in lps.items()},
                "token_logprobs": token_lps,
                "prompt": prompt,
                "question": row.get("question"),
                "choices": row.get("choices"),
                "answer_index": row.get("answer"),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wr.writerow({
                "model": model_name, "model_step": model_step, "idx": idx,
                "subject": subj, "difficulty": diff, "grade": grade,
                "correct": correct_lab, "pred": pred, "is_correct": ok,
                "margin": round(margin, 6),
                "lp_A": round(lps["A"],6), "lp_B": round(lps["B"],6), "lp_C": round(lps["C"],6), "lp_D": round(lps["D"],6), "lp_E": round(lps["E"],6),
                "question_preview": str(row.get("question", ""))[:160].replace("\n", " "),
            })
            if (idx + 1) % 50 == 0:
                acc = correct / (idx + 1)
                print(f"{model_name} {idx+1}/{len(rows)} acc={acc:.4f}", flush=True)
    elapsed = time.time() - t0
    summary = {
        "model": model_name,
        "model_step": str(model_step),
        "benchmark": "TurkishMMLU_sub",
        "n": len(rows),
        "accuracy": round(correct / max(1, len(rows)), 6),
        "correct": correct,
        "elapsed_seconds": round(elapsed, 2),
        "by_subject": {k: {"n": v["n"], "correct": v["correct"], "accuracy": round(v["correct"] / max(1, v["n"]), 6)} for k,v in sorted(by_subject.items())},
        "by_difficulty": {k: {"n": v["n"], "correct": v["correct"], "accuracy": round(v["correct"] / max(1, v["n"]), 6)} for k,v in sorted(by_diff.items())},
        "scoring": "zero-shot plain prompt; answer-label log-prob over A/B/C/D/E; accuracy",
    }
    summ_path = out_dir / f"{model_name}_turkishmmlu_sub_summary.json"
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="nedo,qwen", help="comma list: nedo,qwen")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nedo-ckpt", default="${CONTAINER_PROJECT}/models/nedoqwen08b_9000_sft_clean_20k/sft_final.pt")
    ap.add_argument("--vocab", default="/workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl")
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    rows = load_json(args.data)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded rows={len(rows)} cuda={torch.cuda.is_available()} models={args.models}", flush=True)
    summaries = {}
    for m in [x.strip().lower() for x in args.models.split(',') if x.strip()]:
        if m == "nedo":
            s = run_nedo(rows, out_dir, args.nedo_ckpt, args.vocab, None)
        elif m == "qwen":
            s = run_hf(rows, out_dir, args.qwen_model, None)
        else:
            raise ValueError(f"unknown model selector: {m}")
        summaries[s["model"]] = s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    combined = out_dir / "turkishmmlu_sub_combined_summary.json"
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print("COMBINED", json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    print("wrote", combined, flush=True)


if __name__ == "__main__":
    main()
