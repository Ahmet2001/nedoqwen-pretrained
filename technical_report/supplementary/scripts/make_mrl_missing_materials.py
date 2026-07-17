import json, csv, math, re
from pathlib import Path
from collections import Counter, defaultdict
W=Path('${WORKSPACE}')
OUT=W/'mrl_missing_materials'
OUT.mkdir(exist_ok=True)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows
# Inputs
sft_tok = read_json(W/'tokenizer_sft_large_1298683/tokenizer_sft_large_summary.json')
bench_tok = read_json(W/'tokenizer_compare_turkishmmlu_sub_1298744/tokenizer_compare_summary.json')
pre_tmmlu = read_json(W/'turkishmmlu_sub_full_1298743/new_step9000_sft_clean20k_turkishmmlu_sub_summary.json')
repair_tmmlu = read_json(W/'repaired_v2_600_turkishmmlu_sub_1298778/new_step9000_sft_clean20k_turkishmmlu_sub_summary.json')
pre_tumlu = read_json(W/'tumlu_mini_turkish_eval_1298760/new_step9000_sft_clean20k_tumlu_mini_turkish_summary.json')
repair_tumlu = read_json(W/'repaired_v2_600_tumlu_mini_1298778/new_step9000_sft_clean20k_tumlu_mini_turkish_summary.json')
repair220 = read_json(W/'results_repair_220_1299380/generation_eval_summary.json')['repair_v2_600']
pre220 = read_json(W/'results_expanded_220_1298732/generation_eval_summary.json')['new_step9000_sft_clean20k']
abldist = read_json(W/'ablation_distribution_qualitative_summary_v14.json')
abl_tmmlu = read_json(W/'answer_label_only_600_turkishmmlu_sub_1299407/new_step9000_sft_clean20k_turkishmmlu_sub_summary.json')
abl_tumlu = read_json(W/'answer_label_only_600_tumlu_mini_1299407/new_step9000_sft_clean20k_tumlu_mini_turkish_summary.json')
repair_mix = read_json(W/'repair_mix_v2_summary.json')
abl_data = read_json(W/'answer_label_ablation_data_summary.json')
# Tokenizer coverage table
coverage=[]
coverage.append({'text_source':'Clean SFT sample (5k examples)','tokenizer':'NEDO 65K typed','tokens_per_word':sft_tok['avg_token_per_word'],'unk_rate':sft_tok['unk_rate_micro'],'unk_count':sft_tok['unk_total'],'notes':'in-domain SFT text'})
for name,vals in bench_tok['summary'].items():
    if name in ['NEDO_65K_typed','Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-0.5B-Instruct','Trendyol/Trendyol-LLM-7b-chat-v1.0','microsoft/Phi-3.5-mini-instruct']:
        coverage.append({'text_source':'TurkishMMLU-sub question+choices','tokenizer':name,'tokens_per_word':vals['token_per_word'],'unk_rate':vals['unk_rate'],'unk_count':vals['unk_count'],'notes':'public benchmark text'})
# Tokenizer-failure correlation: correlate unk_count / token_per_word with pre-repair and repair correctness on TMMLU-sub
rows_tok=[]
with (W/'tokenizer_compare_turkishmmlu_sub_1298744/tokenizer_compare_rows.csv').open(encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['tokenizer']=='NEDO_65K_typed':
            rows_tok.append({k:(int(v) if k in ['idx','n_tokens','n_words','n_chars','unk_count'] else v) for k,v in r.items()})
pre_rows=read_jsonl(W/'turkishmmlu_sub_full_1298743/new_step9000_sft_clean20k_turkishmmlu_sub_logprob_outputs.jsonl')
rep_rows=read_jsonl(W/'repaired_v2_600_turkishmmlu_sub_1298778/new_step9000_sft_clean20k_turkishmmlu_sub_logprob_outputs.jsonl')
pre_by={int(r['idx']):r for r in pre_rows}
rep_by={int(r['idx']):r for r in rep_rows}
merged=[]
for t in rows_tok:
    idx=t['idx']; pr=pre_by[idx]; rr=rep_by[idx]
    merged.append({**t,'token_per_word':t['n_tokens']/max(1,t['n_words']),'pre_correct':int(pr['is_correct']),'repair_correct':int(rr['is_correct']),'subject':pr.get('subject'),'difficulty':pr.get('difficulty')})
def acc(xs, field): return sum(x[field] for x in xs)/len(xs) if xs else None
bins=[]
for name, pred in [('unk=0', lambda x:x['unk_count']==0),('unk=1-2', lambda x:1<=x['unk_count']<=2),('unk=3-5', lambda x:3<=x['unk_count']<=5),('unk>=6', lambda x:x['unk_count']>=6)]:
    xs=[x for x in merged if pred(x)]
    bins.append({'bucket':name,'n':len(xs),'pre_repair_acc':round(acc(xs,'pre_correct'),4),'repair600_acc':round(acc(xs,'repair_correct'),4),'avg_tpw':round(sum(x['token_per_word'] for x in xs)/len(xs),3) if xs else None})
# top examples with high unk and incorrect after repair: summarize question snippet without answers
high_unk=[]
for x in sorted(merged, key=lambda z:z['unk_count'], reverse=True)[:20]:
    r=rep_by[x['idx']]
    high_unk.append({'idx':x['idx'],'subject':x['subject'],'difficulty':x['difficulty'],'unk_count':x['unk_count'],'tokens_per_word':round(x['token_per_word'],3),'pre_correct':x['pre_correct'],'repair_correct':x['repair_correct'],'pred':r.get('pred'),'gold':r.get('correct'),'question_snippet':re.sub(r'\s+',' ',r.get('question',''))[:220]})
# Tables and markdown snippets
summary={
 'tokenizer_coverage': coverage,
 'tokenizer_failure_correlation_turkishmmlu_sub': bins,
 'high_unk_examples': high_unk,
 'internal_diagnostic_220': {
   'pre_repair': {'expected':pre220['avg_expected_score'],'repetition':pre220['avg_repetition_ratio'],'bad_outputs':pre220['bad_output_count'],'n':pre220['n_prompts']},
   'repair600': {'expected':repair220['avg_expected_score'],'repetition':repair220['avg_repetition_ratio'],'bad_outputs':repair220['bad_output_count'],'n':repair220['n_prompts']},
   'interpretation':'No aggregate open-ended regression: expected match is stable and bad-output count decreases, but qualitative examples show local regressions.'
 },
 'external_ablation': {
   'pre_repair': {'turkishmmlu_sub':pre_tmmlu['accuracy'],'tumlu_mini_turkish':pre_tumlu['accuracy']},
   'answer_label_only': {'turkishmmlu_sub':abl_tmmlu['accuracy'],'tumlu_mini_turkish':abl_tumlu['accuracy'],'repair_rows':abl_data['repair_rows'],'clean_rows':abl_data['clean_rows_added']},
   'full_repair600': {'turkishmmlu_sub':repair_tmmlu['accuracy'],'tumlu_mini_turkish':repair_tumlu['accuracy'],'repair_rows':repair_mix['repair_rows'],'clean_rows':repair_mix['clean_rows_added']}
 },
 'prediction_label_distribution': {
   'turkishmmlu_sub': abldist['turkishmmlu_sub'],
   'tumlu_mini_turkish': abldist['tumlu_mini_turkish']
 },
 'repair_mix': repair_mix,
 'qualitative_examples': abldist['qualitative_examples_220']
}
(OUT/'mrl_results_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
# Markdown appendix
md=[]
md.append('# MRL Missing Materials for NEDOQwen\n')
md.append('## Tokenizer and Representation Coverage\n')
md.append('| Text source | Tokenizer | Tok/word | UNK rate | UNK count | Notes |\n|---|---:|---:|---:|---:|---|')
for r in coverage:
    md.append(f"| {r['text_source']} | {r['tokenizer']} | {r['tokens_per_word']:.4f} | {r['unk_rate']:.4%} | {r['unk_count']} | {r['notes']} |")
md.append('\n**Interpretation.** The typed Turkish tokenizer is language-aware, but it is not automatically more compact than mature multilingual tokenizers on TurkishMMLU-sub. It has a nonzero UNK rate, while the Qwen tokenizer has zero UNKs in this slice. This should be presented as a representation-coverage finding rather than a tokenizer superiority claim.\n')
md.append('## Tokenizer failure buckets on TurkishMMLU-sub\n')
md.append('| NEDO UNK bucket | n | Pre-repair acc | Repair-600 acc | Avg tok/word |\n|---|---:|---:|---:|---:|')
for b in bins:
    md.append(f"| {b['bucket']} | {b['n']} | {b['pre_repair_acc']:.2%} | {b['repair600_acc']:.2%} | {b['avg_tpw']:.3f} |")
md.append('\nThese buckets should be treated as diagnostic correlations, not causal proof.\n')
md.append('## Post-repair 220-prompt regression check\n')
md.append('| Checkpoint | Expected match | Repetition ratio | Bad outputs |\n|---|---:|---:|---:|')
md.append(f"| Pre-repair clean SFT | {pre220['avg_expected_score']:.3f} | {pre220['avg_repetition_ratio']:.3f} | {pre220['bad_output_count']} / {pre220['n_prompts']} |")
md.append(f"| Full repair-600 | {repair220['avg_expected_score']:.3f} | {repair220['avg_repetition_ratio']:.3f} | {repair220['bad_output_count']} / {repair220['n_prompts']} |")
md.append('\nAggregate regression is not observed: expected match is essentially stable and bad-output count decreases. However, local qualitative regressions remain and should be disclosed.\n')
md.append('## Repair ablation\n')
md.append('| Model | TurkishMMLU-sub | TUMLU-mini Turkish |\n|---|---:|---:|')
md.append(f"| Pre-repair | {pre_tmmlu['accuracy']:.2%} | {pre_tumlu['accuracy']:.2%} |")
md.append(f"| Answer-label-only repair | {abl_tmmlu['accuracy']:.2%} | {abl_tumlu['accuracy']:.2%} |")
md.append(f"| Full repair-600 | {repair_tmmlu['accuracy']:.2%} | {repair_tumlu['accuracy']:.2%} |")
md.append('\nThe answer-label-only ablation improves both external benchmarks, but full repair is better, suggesting that gains are not only from learning to emit answer letters.\n')
md.append('## Prediction label distribution\n')
for bench in ['turkishmmlu_sub','tumlu_mini_turkish']:
    labs=list(summary['prediction_label_distribution'][bench]['full_repair']['pred_counts'].keys())
    md.append(f'### {bench}\n')
    md.append('| Model | ' + ' | '.join(labs) + ' | Max share | Accuracy |\n|---|' + '|'.join(['---:']*(len(labs)+2)) + '|')
    for model in ['pre_repair','answer_label_only','full_repair']:
        d=summary['prediction_label_distribution'][bench][model]
        vals=[f"{100*d['pred_pct'][lab]:.1f}%" for lab in labs]
        md.append(f"| {model} | " + ' | '.join(vals) + f" | {100*d['max_pred_share']:.1f}% | {100*d['accuracy']:.2f}% |")
md.append('\nThe repaired model still has a strong C/D prediction skew, so MRL should frame this as partial calibration rather than solved multiple-choice reasoning.\n')
md.append('## Contamination and leakage control appendix text\n')
md.append('The repair data was constructed without using TurkishMMLU-sub or TUMLU-mini Turkish test rows. We did not copy benchmark question text, answer text, answer keys, or row identifiers into the repair mixture. The repair set targets category-level behaviors revealed by evaluation: answer-letter formatting, arithmetic, Turkish grammar, exact formatting, uncertainty calibration, identity/scope, short explanations, and stable school-level facts. We distinguish row-level leakage from style-level transfer: the repair intentionally teaches generic answer-label calibration and short-form task behavior, so it is benchmark-safe with respect to test-row content but not independent of the broad multiple-choice format. To make this auditable, we report repair category counts and include data-construction scripts/log summaries in the anonymous artifact package.\n')
(OUT/'mrl_missing_materials.md').write_text('\n'.join(md), encoding='utf-8')
# Artifact structure files
artifact=OUT/'artifact'
for sub in ['configs','tokenizer','data_cards','scripts','results','logs','docs']:
    (artifact/sub).mkdir(parents=True, exist_ok=True)
(artifact/'README.md').write_text('# Anonymous NEDOQwen MRL Artifact\n\nThis anonymized artifact summarizes configuration, tokenizer coverage, diagnostics, repair data categories, benchmark-safe contamination controls, and aggregate results for review. It intentionally excludes user names, institutional account names, private paths, and non-anonymous repository links.\n', encoding='utf-8')
(artifact/'configs/model_config.json').write_text(json.dumps({'parameters':824256000,'layers':24,'vocab_size':65536,'hidden_size':1536,'attention_heads':16,'kv_heads':8,'mlp_dim':4096,'context_length':1024,'position_encoding':'RoPE','norm':'RMSNorm','activation':'SwiGLU','embedding_tying':False,'implementation':'custom PyTorch'}, indent=2), encoding='utf-8')
(artifact/'results/mrl_results_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
(artifact/'tokenizer/tokenizer_coverage.md').write_text('\n'.join(md[1:md.index('## Post-repair 220-prompt regression check\n')]), encoding='utf-8')
(artifact/'docs/contamination_check.md').write_text(md[-1], encoding='utf-8')
(artifact/'docs/limitations.md').write_text('Limitations: not production-ready; not state-of-the-art; diagnostic rather than official leaderboard protocol; no full TurkishMMLU; TR-MMLU not run due access; human evaluation not completed; repair data is synthetic and small; tokenizer has nonzero UNK rate; benchmark gains may not transfer to real users.\n', encoding='utf-8')
print('WROTE', OUT)
