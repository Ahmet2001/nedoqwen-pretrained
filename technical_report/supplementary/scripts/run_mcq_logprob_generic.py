import argparse, csv, importlib.util, json, sys, time
from pathlib import Path
from collections import defaultdict

import torch

ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_rows(data, benchmark):
    rows=[]
    if benchmark == "tumlu_mini_turkish":
        for r in data:
            ans = str(r.get("answer", "")).strip().upper()
            choices = r.get("choices", [])
            if ans and choices and ans in ABC[:len(choices)]:
                rows.append({
                    "question": r.get("question", ""),
                    "choices": list(choices),
                    "answer_label": ans,
                    "subject": r.get("subject", "unknown"),
                    "source": "TUMLU-mini/turkish/test",
                })
    elif benchmark == "turkishmmlu_sub":
        for r in data:
            choices = r.get("choices", [])
            ans = ABC[int(r.get("answer"))]
            rows.append({
                "question": r.get("question", ""),
                "choices": list(choices),
                "answer_label": ans,
                "subject": r.get("subject", "unknown"),
                "source": "TurkishMMLU-sub",
            })
    else:
        raise ValueError(f"unknown benchmark {benchmark}")
    return rows


def format_prompt(row):
    n = len(row["choices"])
    labels = ABC[:n]
    lines = [
        "Aşağıdaki çoktan seçmeli soruyu cevapla.",
        "Sadece doğru seçeneğin harfini yaz: " + ", ".join(labels) + ".",
        "",
        f"Soru: {str(row['question']).strip()}",
    ]
    for i, ch in enumerate(row["choices"]):
        lines.append(f"{labels[i]}) {str(ch).strip()}")
    lines.append("")
    lines.append("Cevap:")
    return "\n".join(lines) + " "


def softmax_logprob(logits, target_id):
    return float(torch.log_softmax(logits.float(), dim=-1)[int(target_id)].item())


# NEDO
def import_train_module():
    path = "${CONTAINER_PROJECT}/scripts/20_train_qwen_style.py"
    spec = importlib.util.spec_from_file_location("train_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vocab_jsonl(path):
    token_to_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            token_to_id[obj["token"]] = int(obj["id"])
    return token_to_id


def typed_token(t):
    tok = t.get("token", "")
    typ = t.get("token_type", "UNK")
    if typ == "SPACE":
        return "<SPACE>"
    if not tok:
        return None
    return f"<{typ}>{tok}"


def build_nedo_model(ckpt_path, device):
    train_mod = import_train_module()
    model = train_mod.QwenStyleLM(
        vocab_size=65536, dim=1536, n_layers=24, n_heads=16, n_kv_heads=8,
        mlp_dim=4096, block_size=1024, tie_embeddings=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]; step = ckpt.get("step", "unknown")
    else:
        state = ckpt; step = "model_only"
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARN NEDO load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device); model.eval()
    return model, step


def nedo_runner(ckpt, vocab_path):
    sys.path.insert(0, "${CONTAINER_TOKENIZER}")
    from nedo_turkish_tokenizer import NedoTurkishTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = NedoTurkishTokenizer()
    token_to_id = load_vocab_jsonl(vocab_path)
    model, step = build_nedo_model(ckpt, device)
    def enc(text, add_bos=True):
        bos = token_to_id.get("<bos>", 1); unk = token_to_id.get("<unk>", 3)
        ids = [bos] if add_bos else []
        for t in tokenizer.tokenize(text):
            key = typed_token(t)
            if key is not None:
                ids.append(token_to_id.get(key, unk))
        return ids
    @torch.no_grad()
    def lp(prompt_ids, cont_ids):
        ids = list(prompt_ids) + list(cont_ids)
        block_size = 1024
        if len(ids) > block_size:
            overflow = len(ids) - block_size
            ids = ids[overflow:]
            prompt_len = max(0, len(prompt_ids) - overflow)
        else:
            prompt_len = len(prompt_ids)
        if prompt_len <= 0 or not cont_ids:
            return -1e30
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        logits, _ = model(x, None)
        total = 0.0
        for pos in range(prompt_len, len(ids)):
            total += softmax_logprob(logits[0, pos-1], ids[pos])
        return total / max(1, len(ids) - prompt_len)
    return "new_step9000_sft_clean20k", str(step), enc, lp


# HF
def hf_runner(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        return tok.encode(text, add_special_tokens=False)
    @torch.no_grad()
    def lp(prompt_ids, cont_ids):
        if not cont_ids:
            return -1e30
        ids = prompt_ids + cont_ids
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=next(model.parameters()).device)
        logits = model(x).logits
        prompt_len = len(prompt_ids)
        total = 0.0
        for pos in range(prompt_len, len(ids)):
            total += softmax_logprob(logits[0, pos-1], ids[pos])
        return total / max(1, len(ids) - prompt_len)
    return model_id.replace("/", "__"), "hf", enc, lp


def score(model_name, model_step, rows, out_dir, enc, lp_fn, benchmark_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields=["model","model_step","benchmark","idx","subject","correct","pred","is_correct","margin","question_preview"]
    csv_path = out_dir / f"{model_name}_{benchmark_name}_outputs.csv"
    jsonl_path = out_dir / f"{model_name}_{benchmark_name}_outputs.jsonl"
    correct=0; by_subject=defaultdict(lambda:{"n":0,"correct":0})
    t0=time.time()
    with open(csv_path,"w",encoding="utf-8",newline="") as cf, open(jsonl_path,"w",encoding="utf-8") as jf:
        wr=csv.DictWriter(cf, fieldnames=fields); wr.writeheader()
        for i,row in enumerate(rows):
            labels=ABC[:len(row["choices"])]
            pids=enc(format_prompt(row), add_bos=True)
            lps={lab:lp_fn(pids, enc(lab, add_bos=False)) for lab in labels}
            pred=max(labels, key=lambda x:lps[x])
            ok=int(pred==row["answer_label"]); correct+=ok
            subj=str(row.get("subject","unknown")); by_subject[subj]["n"]+=1; by_subject[subj]["correct"]+=ok
            vals=sorted(lps.values(), reverse=True)
            margin=vals[0]-vals[1] if len(vals)>1 else 0.0
            rec={"model":model_name,"model_step":model_step,"benchmark":benchmark_name,"idx":i,"subject":subj,"correct":row["answer_label"],"pred":pred,"is_correct":ok,"margin":round(margin,6),"logprobs":{k:round(v,6) for k,v in lps.items()},"question":row["question"],"choices":row["choices"]}
            jf.write(json.dumps(rec, ensure_ascii=False)+"\n")
            wr.writerow({k: (str(row["question"])[:160].replace("\n"," ") if k=="question_preview" else rec.get(k,"")) for k in fields})
            if (i+1)%50==0:
                print(f"{model_name} {benchmark_name} {i+1}/{len(rows)} acc={correct/(i+1):.4f}", flush=True)
    summary={
        "model":model_name,"model_step":model_step,"benchmark":benchmark_name,"n":len(rows),"accuracy":round(correct/max(1,len(rows)),6),"correct":correct,"elapsed_seconds":round(time.time()-t0,2),
        "by_subject":{k:{"n":v["n"],"correct":v["correct"],"accuracy":round(v["correct"]/max(1,v["n"]),6)} for k,v in sorted(by_subject.items())},
        "scoring":"zero-shot plain prompt; answer-label log-prob over available answer letters; accuracy",
    }
    with open(out_dir / f"{model_name}_{benchmark_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--benchmark", required=True, choices=["tumlu_mini_turkish","turkishmmlu_sub"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="nedo,Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--nedo-ckpt", default="${CONTAINER_PROJECT}/models/nedoqwen08b_9000_sft_clean_20k/sft_final.pt")
    ap.add_argument("--vocab", default="/workspace/tokenizer/results/nedo_vocab_typed_full/vocab_65536.jsonl")
    args=ap.parse_args()
    data=load_json(args.data)
    rows=extract_rows(data, args.benchmark)
    print("loaded", args.benchmark, "rows", len(rows), "cuda", torch.cuda.is_available(), flush=True)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    allsum={}
    for m in [x.strip() for x in args.models.split(',') if x.strip()]:
        if m.lower()=="nedo":
            model_name, step, enc, lp=nedo_runner(args.nedo_ckpt, args.vocab)
        else:
            model_name, step, enc, lp=hf_runner(m)
        s=score(model_name, step, rows, out_dir, enc, lp, args.benchmark)
        allsum[model_name]=s
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    with open(out_dir / f"{args.benchmark}_combined_summary.json", "w", encoding="utf-8") as f:
        json.dump(allsum, f, ensure_ascii=False, indent=2)
    print("COMBINED", json.dumps(allsum, ensure_ascii=False, indent=2), flush=True)

if __name__=="__main__":
    main()
