import json, csv, statistics
from pathlib import Path
from collections import Counter, defaultdict
W=Path('${WORKSPACE}')

def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def dist(path, labels):
    rows=read_jsonl(path)
    pred=Counter(str(r.get('pred','')).strip() for r in rows)
    corr=Counter(str(r.get('correct','')).strip() for r in rows)
    acc_by_pred={}
    for lab in labels:
        rs=[r for r in rows if str(r.get('pred','')).strip()==lab]
        acc_by_pred[lab]=round(sum(int(r.get('is_correct',0)) for r in rs)/len(rs), 4) if rs else None
    out={
        'n': len(rows),
        'accuracy': round(sum(int(r.get('is_correct',0)) for r in rows)/len(rows), 6),
        'pred_counts': {lab: pred.get(lab,0) for lab in labels},
        'pred_pct': {lab: round(pred.get(lab,0)/len(rows), 4) for lab in labels},
        'gold_counts': {lab: corr.get(lab,0) for lab in labels},
        'max_pred_share': round(max((pred.get(lab,0) for lab in labels), default=0)/len(rows), 4),
        'accuracy_by_pred': acc_by_pred,
    }
    return out

paths={
 'turkishmmlu_sub': {
   'labels': list('ABCDE'),
   'pre_repair': W/'turkishmmlu_sub_full_1298743/new_step9000_sft_clean20k_turkishmmlu_sub_logprob_outputs.jsonl',
   'answer_label_only': W/'answer_label_only_600_turkishmmlu_sub_1299407/new_step9000_sft_clean20k_turkishmmlu_sub_logprob_outputs.jsonl',
   'full_repair': W/'repaired_v2_600_turkishmmlu_sub_1298778/new_step9000_sft_clean20k_turkishmmlu_sub_logprob_outputs.jsonl',
 },
 'tumlu_mini_turkish': {
   'labels': list('ABCD'),
   'pre_repair': W/'tumlu_mini_turkish_eval_1298760/new_step9000_sft_clean20k_tumlu_mini_turkish_outputs.jsonl',
   'answer_label_only': W/'answer_label_only_600_tumlu_mini_1299407/new_step9000_sft_clean20k_tumlu_mini_turkish_outputs.jsonl',
   'full_repair': W/'repaired_v2_600_tumlu_mini_1298778/new_step9000_sft_clean20k_tumlu_mini_turkish_outputs.jsonl',
 }
}
summary={}
for bench, cfg in paths.items():
    summary[bench]={}
    for model, path in cfg.items():
        if model=='labels': continue
        summary[bench][model]=dist(path, cfg['labels'])

# Qual examples from 220-prompt diagnostic
pre_rows=read_jsonl(W/'results_expanded_220_1298732/generation_eval_outputs.jsonl')
repair_rows=read_jsonl(W/'results_repair_220_1299380/generation_eval_outputs.jsonl')
pre={(r['prompt_id'], r['model']): r for r in pre_rows}
repair={r['prompt_id']: r for r in repair_rows if r['model']=='repair_v2_600'}
base_name='new_step9000_sft_clean20k'
pairs=[]
for pid, rr in repair.items():
    pr=pre.get((pid, base_name))
    if not pr: continue
    delta=rr['expected_score']-pr['expected_score']
    bad_pre=int(pr['too_short'] or pr['prompt_copy_like'] or pr['repetition_ratio']>0.55)
    bad_rep=int(rr['too_short'] or rr['prompt_copy_like'] or rr['repetition_ratio']>0.55)
    pairs.append({
      'prompt_id': pid,
      'category': rr['category'],
      'prompt': pr['prompt'][:400],
      'expected_contains': pr.get('expected_contains', []),
      'pre_score': pr['expected_score'],
      'repair_score': rr['expected_score'],
      'delta': round(delta,3),
      'pre_rep': pr['repetition_ratio'],
      'repair_rep': rr['repetition_ratio'],
      'pre_bad': bad_pre,
      'repair_bad': bad_rep,
      'pre_output': pr['output_display'][:500],
      'repair_output': rr['output_display'][:500],
    })
# diverse improved examples, then stable/regression examples
chosen=[]; cats=set()
for p in sorted(pairs, key=lambda x:(x['pre_bad']-x['repair_bad'], x['delta'], -x['repair_rep']), reverse=True):
    if p['delta']>0 or (p['pre_bad']==1 and p['repair_bad']==0):
        if p['category'] not in cats or len(chosen)<8:
            chosen.append(p); cats.add(p['category'])
        if len(chosen)>=12: break
for p in sorted(pairs, key=lambda x:x['delta']):
    if len(chosen)>=16: break
    if p['delta']<0 and p['prompt_id'] not in {q['prompt_id'] for q in chosen}:
        chosen.append(p)
summary['qualitative_examples_220'] = chosen[:16]

out=W/'ablation_distribution_qualitative_summary_v14.json'
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('WROTE', out)
