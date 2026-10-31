import hashlib, json, os, re, statistics, subprocess, time, wave
from pathlib import Path

out = Path(__file__).resolve().parent
old = Path('/home/leon/code/tmp/penguin/llama-webgpu-arange-research/experiments/fullmodel')
base_libs = '/tmp/llama-count-equal-baseline-build/bin'
pr_libs = '/tmp/llama-count-equal-build/bin'
models = [
 ('pocket', json.loads((old/'pocket-pr-trial0.command.json').read_text())),
 ('qwen06', json.loads((old/'qwen-pr-trial0.command.json').read_text())),
 ('qwen17', json.loads((old/'qwen-pr-trial0.command.json').read_text())),
]
models[1][1][models[1][1].index('-m')+1]='/tmp/llama-arange-fullmodel-models/Qwen3-TTS-12Hz-0.6B-Base.Q4_K_M.gguf'
models[1][1][models[1][1].index('-mm')+1]='/tmp/llama-arange-fullmodel-models/Qwen3-TTS-12Hz-0.6B-Base.mmproj-Q8_0.gguf'
results=[]
for model, template in models:
    for trial in [-2, -1, 0, 1, 2, 3, 4]:
        variants = ['pr'] if trial == -2 else (['baseline','pr'] if trial % 2 == 0 else ['pr','baseline'])
        for variant in variants:
            tag=f'{model}-{variant}-trial{trial}'
            cmd=template.copy()
            cmd[0]=pr_libs+'/llama-tts'
            cmd[cmd.index('-n')+1]='16'
            cmd[cmd.index('-o')+1]=str(out/(tag+'.wav'))
            env=os.environ.copy()
            env['VK_ICD_FILENAMES']='/usr/share/vulkan/icd.d/nvidia_icd.json'
            env['XDG_RUNTIME_DIR']='/tmp'
            env['LD_LIBRARY_PATH']=(base_libs+':' if variant=='baseline' else '')+pr_libs+':/tmp/llama-dawn/lib64'
            env['LD_PRELOAD']=str(out/'driver-pin.so')
            if trial == -2: env['LD_PRELOAD']+=':'+str(out/'trace-ops.so')
            (out/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'environment':{k:v for k,v in env.items() if k in ['LD_LIBRARY_PATH','LD_PRELOAD','VK_ICD_FILENAMES','XDG_RUNTIME_DIR']}},indent=2))
            with (out/(tag+'.log')).open('w') as f:
                t=time.perf_counter();p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);wall=time.perf_counter()-t
            log=(out/(tag+'.log')).read_text()
            if p.returncode: raise RuntimeError(f'{tag} exit {p.returncode}; see log')
            match=re.search(r'timings: prompt eval ([\d.]+)s \+ generation ([\d.]+)s \+ vocoder ([\d.]+)s = total ([\d.]+)s',log)
            if not match: raise RuntimeError(f'missing timing: {tag}')
            wav_path=out/(tag+'.wav')
            with wave.open(str(wav_path),'rb') as w:
                pcm=w.readframes(w.getnframes());audio_s=w.getnframes()/w.getframerate()
            row={'model':model,'variant':variant,'trial':trial,'wall_s':wall,'prompt_s':float(match[1]),'generation_s':float(match[2]),'vocoder_s':float(match[3]),'inference_s':float(match[4]),'audio_s':audio_s,'pcm_sha256':hashlib.sha256(pcm).hexdigest()}
            trace=re.search(r'OP_TRACE .*',log)
            if trace: row['trace']=trace[0]
            results.append(row)
            (out/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)
summary=[]
for model,_ in models:
    row={'model':model}
    for variant in ['baseline','pr']:
        values=[r['inference_s'] for r in results if r['model']==model and r['variant']==variant and r['trial']>=0]
        row[variant]={'mean_s':statistics.mean(values),'sd_s':statistics.stdev(values),'n':len(values)}
    row['speedup']=row['baseline']['mean_s']/row['pr']['mean_s']
    row['pcm_identical']=len({r['pcm_sha256'] for r in results if r['model']==model})==1
    row['trace']=[r['trace'] for r in results if r['model']==model and 'trace' in r][0]
    summary.append(row)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
