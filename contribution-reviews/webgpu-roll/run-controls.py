import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
bin_dir = Path('/mnt/data2/rsi/llama-webgpu-roll-build/bin')
model = '/tmp/llama-count-equal-models/frequent/SmolLM2-135M-Instruct-Q4_K_M.gguf'
dataset = '/tmp/llama-count-equal-models/frequent/wikitext-2-raw/wiki.valid.raw'
for variant in ['baseline', 'candidate']:
    env = os.environ.copy()
    overrides = {
        'VK_ICD_FILENAMES': '/usr/share/vulkan/icd.d/nvidia_icd.json',
        'XDG_RUNTIME_DIR': '/tmp',
        'LD_LIBRARY_PATH': ('/mnt/data2/rsi/llama-webgpu-roll-baseline-bin:' if variant == 'baseline' else '') + str(bin_dir) + ':/tmp/llama-dawn/lib64',
        'LD_PRELOAD': '/tmp/llama-webgpu-count-equal-evidence/fullmodel/driver-pin.so',
    }
    env.update(overrides)
    commands = {
        'bench': [str(bin_dir / 'llama-bench'), '-m', model, '-ngl', '99', '-t', '4', '-p', '128', '-n', '32', '-r', '3', '-o', 'json'],
        'ppl': [str(bin_dir / 'llama-perplexity'), '-m', model, '-ngl', '99', '-t', '4', '-c', '128', '-b', '128', '-ub', '128', '--chunks', '2', '-f', dataset],
    }
    loader_env = dict(env, LD_TRACE_LOADED_OBJECTS='1')
    with (root / ('libraries-' + variant + '.log')).open('w') as f:
        subprocess.run([str(root / 'mtmd-bench')], env=loader_env, stdout=f, stderr=subprocess.STDOUT, check=True)
    for test, cmd in commands.items():
        tag = 'control-' + variant + '-' + test
        (root / (tag + '-command.json')).write_text(json.dumps({'argv': cmd, 'env': overrides}, indent=2))
        with (root / (tag + '.log')).open('w') as f:
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
        print(tag, r.returncode, flush=True)
        if r.returncode:
            raise RuntimeError(tag)
