import json, random
from pathlib import Path
from collections import Counter
random.seed(424242)
base = Path('${PROJECT_ROOT}')
repair_in = base / 'sft_data/tr_repair_mix_v2.jsonl'
clean_in = base / 'sft_data/tr_sft_clean_20k.jsonl'
out_repair = base / 'sft_data/tr_repair_answer_label_only_v1.jsonl'
out_mix = base / 'sft_data/tr_repair_answer_label_only_v1_plus_clean.jsonl'
rows=[]
with repair_in.open(encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        r=json.loads(line)
        if r.get('source') == 'repair_v2_format_letter':
            rows.append(r)
clean=[]
with clean_in.open(encoding='utf-8') as f:
    for i,line in enumerate(f):
        if not line.strip():
            continue
        if i % 10 == 0:
            clean.append(json.loads(line))
        if len(clean) >= 2000:
            break
mix = rows + clean
random.shuffle(mix)
out_repair.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
out_mix.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in mix)+'\n', encoding='utf-8')
summary = {
  'ablation': 'answer_label_only_plus_clean',
  'repair_rows': len(rows),
  'clean_rows_added': len(clean),
  'mixed_rows': len(mix),
  'repair_sources': Counter(r.get('source') for r in rows).most_common(),
  'input_repair_file': str(repair_in),
  'output_repair_file': str(out_repair),
  'output_mix_file': str(out_mix)
}
Path('${WORKSPACE}/answer_label_ablation_data_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
