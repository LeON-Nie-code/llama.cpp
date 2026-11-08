import json
from pathlib import Path
import re
import statistics
import os

root = Path(os.environ.get('ROLL_BENCH_RESULTS_DIR', str(Path(__file__).resolve().parent)))
rows = json.loads((root / 'model-results.json').read_text())
for row in rows:
    tag = f"{row['scenario']}-{row['variant']}-{row['trial']}"
    log = (root / (tag + '.log')).read_text()
    # The CLI logger can insert newlines between numeric token IDs.
    tokens = re.findall(r'BENCH_TOKENS([\s\d]+)BENCH_INFERENCE', log)
    assert len(tokens) == 7, tag
    row['output_tokens'] = [int(x) for x in tokens[-1].split()]
(root / 'model-results.json').write_text(json.dumps(rows, indent=2))

summary = []
for scenario in sorted({x['scenario'] for x in rows}):
    r = {'scenario': scenario}
    for variant in ['baseline', 'candidate']:
        selected = [x for x in rows if x['scenario'] == scenario and x['variant'] == variant and x['trial'] >= 0]
        r[variant] = {'n': len(selected)}
        for metric in ['total_s', 'audio_encoding_s']:
            values = [x[metric] for x in selected]
            r[variant][metric] = {'mean': statistics.mean(values), 'sd': statistics.stdev(values)}
    r['speedup'] = r['baseline']['total_s']['mean'] / r['candidate']['total_s']['mean']
    r['tokens_identical'] = len({tuple(i['tokens']) for x in rows if x['scenario'] == scenario for i in x['iterations']}) == 1
    diagnostics = [x for x in rows if x['scenario'] == scenario and x['trial'] == -1]
    r['embeddings_identical'] = len({h for x in diagnostics for h in x['embedding_sha256']}) == 1
    r['placement'] = {x['variant']: x['placement'] for x in diagnostics}
    r['generated_tokens'] = len(diagnostics[0]['output_tokens'])
    summary.append(r)
(root / 'model-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
