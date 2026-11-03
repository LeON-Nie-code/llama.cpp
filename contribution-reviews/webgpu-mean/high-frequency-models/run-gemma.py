import json,os,re,statistics,subprocess,time,sys
from pathlib import Path
out=Path(__file__).resolve().parent
models={'e2b':'/mnt/data2/rsi/llama-webgpu-mean-models/google_gemma-3n-E2B-it-Q4_K_M.gguf','e4b':'/mnt/data2/rsi/llama-webgpu-mean-models/google_gemma-3n-E4B-it-Q4_K_M.gguf'}
model=sys.argv[1]
rows=[]
for trial in [-1,0,1,2,3,4]:
 for variant in (['baseline','candidate'] if trial%2 else ['candidate','baseline']):
  tag=f'{model}-{variant}-{trial}'
  cmd=['/tmp/llama-mean-build/bin/llama-bench','-m',models[model],'-p','0' if trial==-1 else '128,512','-n','128','-r','1','-ngl','99','-dev','WebGPU','-fa','off','-t','2','-b','512','-ub','512','-o','json']
  e=os.environ.copy();e.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-mean-baseline-bin:' if variant=='baseline' else '')+'/tmp/llama-mean-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD='/tmp/llama-webgpu-count-equal-evidence/fullmodel/driver-pin.so')
  if trial==-1:e['LD_PRELOAD']+=':/tmp/llama-mean-evidence/trace-mean.so'
  (out/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'env':{k:e[k] for k in ['LD_LIBRARY_PATH','LD_PRELOAD','XDG_RUNTIME_DIR','VK_ICD_FILENAMES']}},indent=2))
  t=time.monotonic()
  with (out/(tag+'.json')).open('w') as f,(out/(tag+'.log')).open('w') as log:r=subprocess.run(cmd,env=e,stdout=f,stderr=log)
  assert r.returncode==0,(tag,r.returncode)
  measurements=json.loads((out/(tag+'.json')).read_text())
  row={'model':model,'variant':variant,'trial':trial,'wall_s':time.monotonic()-t,'measurements':[{k:x[k] for k in ['model_type','n_prompt','n_gen','avg_ns','avg_ts','samples_ns']} for x in measurements]}
  if trial==-1:
   log=(out/(tag+'.log')).read_text();m=re.search(r'OP_TRACE .*',log);assert m;row['trace']=m[0]
   field='cpu_mean' if variant=='baseline' else 'webgpu_mean'
   assert field+'=1290' in row['trace'],row['trace']
  rows.append(row);(out/f'{model}-results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
summary=[]
for pp,tg in [(128,0),(512,0),(0,128)]:
 s={'model':model,'n_prompt':pp,'n_gen':tg}
 for variant in ['baseline','candidate']:
  ms=[x for r in rows if r['trial']>=0 and r['variant']==variant for x in r['measurements'] if x['n_prompt']==pp and x['n_gen']==tg]
  times=[x['avg_ns']/1e9 for x in ms];rates=[x['avg_ts'] for x in ms];assert len(times)==5
  s[variant]={'mean_s':statistics.mean(times),'sd_s':statistics.stdev(times),'mean_tokens_s':statistics.mean(rates),'sd_tokens_s':statistics.stdev(rates),'n':5}
 s['speedup']=s['baseline']['mean_s']/s['candidate']['mean_s'];s['latency_reduction_percent']=(1-1/s['speedup'])*100;summary.append(s)
(out/f'{model}-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
