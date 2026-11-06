import hashlib, json, os, re, statistics, subprocess, time, wave
from pathlib import Path
out=Path(__file__).resolve().parent
src=Path('/home/leon/code/tmp/penguin/llama-webgpu-arange-research/experiments/fullmodel')
bin=Path('/tmp/llama-pad-build/bin')
models={'qwen06':json.loads((src/'qwen-pr-trial0.command.json').read_text())}
q=models['qwen06'];q[q.index('-m')+1]='/tmp/llama-arange-fullmodel-models/Qwen3-TTS-12Hz-0.6B-Base.Q4_K_M.gguf';q[q.index('-mm')+1]='/tmp/llama-arange-fullmodel-models/Qwen3-TTS-12Hz-0.6B-Base.mmproj-Q8_0.gguf'
rows=[]
for name,template in models.items():
 for trial in [-1,0,1,2,3,4]:
  for variant in (['baseline','candidate'] if trial%2 else ['candidate','baseline']):
   tag=f'{name}-{variant}-{trial}'
   cmd=template.copy();cmd[0]=str(out/'tts-bench');cmd[cmd.index('-n')+1]='16';cmd[cmd.index('-o')+1]=str(out/(tag+'.wav'))
   env=os.environ.copy();env.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-pad-baseline-build/bin:' if variant=='baseline' else '')+str(bin)+':/tmp/llama-dawn/lib64',LD_PRELOAD='/tmp/llama-webgpu-count-equal-evidence/fullmodel/driver-pin.so')
   if trial==-1:env['LD_PRELOAD']+=':'+'/tmp/pad-trace.so'
   (out/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'env':{k:env[k] for k in ['VK_ICD_FILENAMES','XDG_RUNTIME_DIR','LD_LIBRARY_PATH','LD_PRELOAD']}},indent=2))
   t=time.monotonic()
   with (out/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   assert p.returncode==0,(tag,p.returncode)
   s=(out/(tag+'.log')).read_text();m=re.search(r'timings: prompt eval ([\d.]+)s \+ generation ([\d.]+)s \+ vocoder ([\d.]+)s = total ([\d.]+)s',s);assert m,tag
   with wave.open(str(out/(tag+'.wav')),'rb') as w:
    pcm=w.readframes(w.getnframes());samples=w.getnframes();rate=w.getframerate()
   mfull=re.search(r'BENCH_TTS input_s=([\d.]+) full_inference_s=([\d.]+)',s);assert mfull,tag
   r={'input_s':float(mfull[1]),'full_inference_s':float(mfull[2]),'model':name,'variant':variant,'trial':trial,'wall_s':time.monotonic()-t,'prompt_s':float(m[1]),'generation_s':float(m[2]),'vocoder_s':float(m[3]),'inference_s':float(m[4]),'samples':samples,'rate':rate,'pcm_sha256':hashlib.sha256(pcm).hexdigest()}
   if trial==-1:
    trace=re.search(r'OP_TRACE .*',s);assert trace,tag;r['trace']=trace[0];r['pad_details']=re.findall(r'PAD_DETAIL .*',s)
    r['shapes']=re.findall(r'MEAN_NODE .*',s)
   rows.append(r);(out/'model-results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(r),flush=True)
summary=[]
for name in models:
 row={'model':name}
 for variant in ['baseline','candidate']:
  vals=[r['full_inference_s'] for r in rows if r['model']==name and r['variant']==variant and r['trial']>=0]
  row[variant]={'mean_s':statistics.mean(vals),'sd_s':statistics.stdev(vals),'n':len(vals)}
 row['speedup']=row['baseline']['mean_s']/row['candidate']['mean_s'];row['traces']={r['variant']:r['trace'] for r in rows if r['model']==name and r['trial']==-1};row['pcm_identical']=len({r['pcm_sha256'] for r in rows if r['model']==name})==1
 summary.append(row)
(out/'model-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
