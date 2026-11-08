import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

root = Path(__file__).resolve().parent
candidate = Path('/mnt/data2/rsi/llama-webgpu-roll-build/bin')
baseline = Path('/mnt/data2/rsi/llama-webgpu-roll-baseline-bin')
models = Path('/mnt/data2/rsi/llama-webgpu-roll-research/models')
audio = {
    'speech-3s': Path('/tmp/llama-arange-fullmodel-models/reference.wav'),
    'repeated-speech-30s': Path('/home/leon/code/tmp/penguin/ce-roll-research/reference-repeated-30s.wav'),
}
rows = []
for scenario, path in audio.items():
    for trial in [-1, 0, 1, 2, 3, 4]:
        order = ['baseline', 'candidate'] if trial % 2 else ['candidate', 'baseline']
        for variant in order:
            tag = f'{scenario}-{variant}-{trial}'
            cmd = [str(root / 'mtmd-bench'), '-m', str(models / 'LFM2.5-Audio-1.5B-Q8_0.gguf'),
                   '--mmproj', str(models / 'mmproj-LFM2.5-Audio-1.5B-F16.gguf'),
                   '--audio', str(path), '-p', 'Transcribe this audio.', '-n', '64',
                   '-c', '2048', '-ub', '512', '-b', '512', '-ngl', '99', '-t', '4',
                   '--no-warmup', '--jinja', '--seed', '42', '--temp', '0']
            overrides = {
                'VK_ICD_FILENAMES': '/usr/share/vulkan/icd.d/nvidia_icd.json',
                'XDG_RUNTIME_DIR': '/tmp',
                'LD_LIBRARY_PATH': (str(baseline) + ':' if variant == 'baseline' else '') + str(candidate) + ':/tmp/llama-dawn/lib64',
                'LD_PRELOAD': '/tmp/llama-webgpu-count-equal-evidence/fullmodel/driver-pin.so',
            }
            if trial == -1:
                overrides['LD_PRELOAD'] += ':' + str(root / 'trace-roll.so')
                overrides['ROLL_BENCH_EMBEDDINGS'] = str(root / (tag + '-embeddings'))
            env = os.environ.copy()
            env.update(overrides)
            (root / (tag + '-command.json')).write_text(json.dumps({'argv': cmd, 'env': overrides}, indent=2))
            start = time.monotonic()
            with (root / (tag + '.log')).open('w') as f:
                result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            if result.returncode:
                raise RuntimeError((tag, result.returncode))
            log = (root / (tag + '.log')).read_text()
            times = re.findall(r'BENCH_INFERENCE trial=(-?\d+) total_s=([\d.]+)', log)
            encodes = re.findall(r'BENCH_AUDIO encoding_s=([\d.]+)', log)
            tokens = re.findall(r'BENCH_TOKENS([\s\d]+)BENCH_INFERENCE', log)
            assert len(times) == len(encodes) == len(tokens) == 7, tag
            assert [int(x[0]) for x in times] == list(range(-2, 5))
            row = {'scenario': scenario, 'variant': variant, 'trial': trial,
                   'total_s': statistics.mean(float(x[1]) for x in times[2:]), 'audio_encoding_s': statistics.mean(float(x) for x in encodes[2:]),
                   'iterations': [{'total_s': float(t[1]), 'audio_encoding_s': float(a), 'tokens': [int(x) for x in k.split()]} for t, a, k in zip(times[2:], encodes[2:], tokens[2:])],
                   'output_tokens': [int(x) for x in tokens[-1].split()],
                   'process_wall_s': time.monotonic() - start}
            if trial == -1:
                row['placement'] = re.search(r'BACKEND_OP op=ROLL cpu=\d+ gpu=\d+', log)[0]
                row['embedding_sha256'] = [hashlib.sha256((root / (tag + f'-embeddings-{i}.bin')).read_bytes()).hexdigest() for i in range(7)]
            rows.append(row)
            (root / 'model-results.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k != 'output_tokens'}), flush=True)

summary = []
for scenario in audio:
    r = {'scenario': scenario}
    for variant in ['baseline', 'candidate']:
        selected = [x for x in rows if x['scenario'] == scenario and x['variant'] == variant and x['trial'] >= 0]
        r[variant] = {'n': len(selected)}
        for metric in ['total_s', 'audio_encoding_s']:
            values = [x[metric] for x in selected]
            r[variant][metric] = {'mean': statistics.mean(values), 'sd': statistics.stdev(values)}
    r['speedup'] = r['baseline']['total_s']['mean'] / r['candidate']['total_s']['mean']
    r['tokens_identical'] = len({tuple(x['output_tokens']) for x in rows if x['scenario'] == scenario}) == 1
    diagnostics = [x for x in rows if x['scenario'] == scenario and x['trial'] == -1]
    r['embeddings_identical'] = len({h for x in diagnostics for h in x['embedding_sha256']}) == 1
    r['placement'] = {x['variant']: x['placement'] for x in diagnostics}
    summary.append(r)
(root / 'model-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
