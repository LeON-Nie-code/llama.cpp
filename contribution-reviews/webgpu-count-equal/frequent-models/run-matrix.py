from pathlib import Path
import os,subprocess,json,re,hashlib,statistics
root=Path(__file__).resolve().parent;old=root.parent/'fullmodel'
if not old.is_dir():old=root.parent
models={'smol135':'/tmp/llama-count-equal-models/frequent/SmolLM2-135M-Instruct-Q4_K_M.gguf','qwen05':'/tmp/llama-count-equal-models/frequent/qwen2.5-0.5b-instruct-q4_k_m.gguf'}
rows=[]
for name,path in models.items():
 for context in [256,1024]:
  contexts=4096//context
  for trial in [-1,0,1,2,3,4]:
   for variant in (['baseline','pr'] if trial%2==0 else ['pr','baseline']):
    tag=f'{name}-c{context}-{variant}-{trial}'
    env=os.environ.copy();env.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-count-equal-baseline-build/bin:' if variant=='baseline' else '')+'/tmp/llama-count-equal-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD=str(old/'driver-pin.so'))
    if trial==-1:env['LD_PRELOAD']+=':'+str(old/'trace-ops.so')
    epochs=1 if trial==-1 else 2
    cmd=[str(root/'eval-llm'),path,str(root/(tag+'.bin')),str(epochs),str(context),str(contexts),'/tmp/llama-count-equal-models/frequent/wikitext-2-raw/wiki.valid.raw']
    (root/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['LD_LIBRARY_PATH','LD_PRELOAD','VK_ICD_FILENAMES','XDG_RUNTIME_DIR']}},indent=2))
    with (root/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{tag}: exit {p.returncode}')
    log=(root/(tag+'.log')).read_text();trace=re.search(r'OP_TRACE .*',log)
    timings=[];loss=[];accuracy=[]
    for m in re.finditer(r'EVAL epoch=(-?\d+) tokens=(\d+) ms=([\d.]+) loss=([\d.e+-]+) accuracy=([\d.e+-]+)',log):
     assert int(m[2])==4096
     if int(m[1])>=0:timings.append(float(m[3]));loss.append(float(m[4]));accuracy.append(float(m[5]))
    assert len(timings)==epochs
    row={'model':name,'context':context,'batches_per_epoch':contexts,'count_equal_per_epoch':contexts,'variant':variant,'trial':trial,'epochs_ms':timings,'mean_ms':statistics.mean(timings),'loss':loss,'accuracy':accuracy,'pred_sha256':hashlib.sha256((root/(tag+'.bin')).read_bytes()).hexdigest()}
    if trial==-1:
     assert trace,tag
     calls=contexts*2
     expected=f'webgpu_count_equal={calls} cpu_count_equal=0' if variant=='pr' else f'webgpu_count_equal=0 cpu_count_equal={calls}'
     assert expected in trace[0],(tag,trace[0])
     row['trace']=trace[0]
    rows.append(row);(root/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    measured=[r for r in rows if r['model']==name and r['context']==context and r['trial']>=0]
    if len(measured)>=2:
     assert len({r['pred_sha256'] for r in measured})==1,(name,context,'predictions')
     assert len({tuple(r['loss']) for r in measured})==1,(name,context,'loss')
     assert len({tuple(r['accuracy']) for r in measured})==1,(name,context,'accuracy')
summary=[]
for name in models:
 for context in [256,1024]:
  selected=[r for r in rows if r['model']==name and r['context']==context]
  row={'model':name,'context':context,'tokens_per_epoch':4096,'count_equal_per_epoch':4096//context,'traces':{r['variant']:r['trace'] for r in selected if r['trial']==-1}}
  for variant in ['baseline','pr']:
   vals=[r['mean_ms'] for r in selected if r['variant']==variant and r['trial']>=0]
   row[variant]={'mean_ms':statistics.mean(vals),'sd_ms':statistics.stdev(vals),'n_processes':len(vals),'epochs_per_process':2}
  row['speedup']=row['baseline']['mean_ms']/row['pr']['mean_ms']
  row['predictions_identical']=len({r['pred_sha256'] for r in selected if r['trial']>=0})==1
  row['loss_identical']=len({tuple(r['loss']) for r in selected if r['trial']>=0})==1
  row['accuracy_identical']=len({tuple(r['accuracy']) for r in selected if r['trial']>=0})==1
  summary.append(row)
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
