import hashlib,json,os,re,statistics,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent
while not (out/'summary.json').exists(): time.sleep(2)
base='/tmp/llama-count-equal-baseline-build/bin'
pr='/tmp/llama-count-equal-build/bin'
rows=[]
for model in ['stories260K','stories15M']:
 for trial in [-1,0,1,2,3,4]:
  for variant in (['baseline','pr'] if trial%2==0 else ['pr','baseline']):
   tag=f'eval-{model}-{variant}-{trial}'
   env=os.environ.copy()
   env.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=(base+':' if variant=='baseline' else '')+pr+':/tmp/llama-dawn/lib64',LD_PRELOAD=str(out/'driver-pin.so'))
   if trial==-1:env['LD_PRELOAD']+=':'+str(out/'trace-ops.so')
   cmd=[str(out/'eval-llm'),'/tmp/llama-count-equal-models/'+model+'.gguf',str(out/(tag+'.pred.bin')),'1' if trial==-1 else '6']
   (out/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['LD_LIBRARY_PATH','LD_PRELOAD','VK_ICD_FILENAMES','XDG_RUNTIME_DIR']}},indent=2))
   with (out/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   if p.returncode: raise RuntimeError(f'{tag}: exit {p.returncode}')
   log=(out/(tag+'.log')).read_text()
   values=[];loss=[];accuracy=[]
   for m in re.finditer(r'EVAL epoch=(-?\d+) tokens=(\d+) ms=([\d.]+) loss=([\d.e+-]+) accuracy=([\d.e+-]+)',log):
    if int(m[1])>=0:values.append(float(m[3]));loss.append(float(m[4]));accuracy.append(float(m[5]))
   trace=re.search(r'OP_TRACE .*',log)
   row={'model':model,'variant':variant,'trial':trial,'epochs_ms':values,'mean_ms':statistics.mean(values),'loss':loss,'accuracy':accuracy,'pred_sha256':hashlib.sha256((out/(tag+'.pred.bin')).read_bytes()).hexdigest()}
   if trace:row['trace']=trace[0]
   rows.append(row);(out/'eval-results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
summary=[]
for model in ['stories260K','stories15M']:
 row={'model':model}
 for variant in ['baseline','pr']:
  vals=[r['mean_ms'] for r in rows if r['model']==model and r['variant']==variant and r['trial']>=0]
  row[variant]={'mean_ms':statistics.mean(vals),'sd_ms':statistics.stdev(vals),'n_processes':len(vals),'epochs_per_process':6}
 row['speedup']=row['baseline']['mean_ms']/row['pr']['mean_ms']
 row['predictions_identical']=len({r['pred_sha256'] for r in rows if r['model']==model and r['trial']>=0})==1
 row['loss_identical']=len({tuple(r['loss']) for r in rows if r['model']==model and r['trial']>=0})==1
 row['accuracy_identical']=len({tuple(r['accuracy']) for r in rows if r['model']==model and r['trial']>=0})==1
 row['traces']={r['variant']:r['trace'] for r in rows if r['model']==model and r['trial']==-1}
 summary.append(row)
(out/'eval-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
