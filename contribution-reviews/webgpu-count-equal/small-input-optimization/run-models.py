from pathlib import Path
import os,subprocess,json,re,hashlib,statistics,itertools
root=Path(__file__).resolve().parent;helpers=root.parent/'fullmodel'
if not helpers.is_dir():helpers=root.parent
exe=root/'eval-llm'
models={'stories260K':'/tmp/llama-count-equal-models/stories260K.gguf','smol135':'/tmp/llama-count-equal-models/frequent/SmolLM2-135M-Instruct-Q4_K_M.gguf','qwen05':'/tmp/llama-count-equal-models/frequent/qwen2.5-0.5b-instruct-q4_k_m.gguf'}
variants=['upstream','before','optimized']
extra={'upstream':'/tmp/llama-count-equal-baseline-build/bin:','before':'/tmp/llama-count-equal-original-backend:','optimized':''}
orders=[['upstream','before','optimized'],['before','optimized','upstream'],['optimized','upstream','before'],['optimized','before','upstream'],['upstream','optimized','before']]
configs=[('stories260K',256),('smol135',256),('smol135',1024),('qwen05',256),('qwen05',1024)]
rows=[]
for model,context in configs:
 batches=4096//context
 for trial in [-1,0,1,2,3,4]:
  for variant in (variants if trial==-1 else orders[trial]):
   tag=f'{model}-c{context}-{variant}-{trial}'
   env=os.environ.copy();env.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=extra[variant]+'/tmp/llama-count-equal-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD=str(helpers/'driver-pin.so'))
   if trial==-1:env['LD_PRELOAD']+=':'+str(helpers/'trace-ops.so')
   cmd=[str(exe),models[model],str(root/(tag+'.bin')),'1',str(context),str(batches),'/tmp/llama-count-equal-models/frequent/wikitext-2-raw/wiki.valid.raw']
   (root/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['LD_LIBRARY_PATH','LD_PRELOAD','VK_ICD_FILENAMES','XDG_RUNTIME_DIR']}},indent=2))
   with (root/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   assert p.returncode==0,(tag,p.returncode)
   log=(root/(tag+'.log')).read_text()
   m=re.search(r'EVAL epoch=0 tokens=4096 ms=([\d.]+) loss=([\d.e+-]+) accuracy=([\d.e+-]+)',log)
   assert m,tag
   row={'model':model,'context':context,'tokens':4096,'count_equal_calls_per_epoch':batches,'variant':variant,'trial':trial,'ms':float(m[1]),'loss':float(m[2]),'accuracy':float(m[3]),'pred_sha256':hashlib.sha256((root/(tag+'.bin')).read_bytes()).hexdigest()}
   if trial==-1:
    trace=re.search(r'OP_TRACE .*',log);assert trace,tag;row['trace']=trace[0]
    n=2*batches
    expected=f'webgpu_count_equal=0 cpu_count_equal={n}' if variant=='upstream' else f'webgpu_count_equal={n} cpu_count_equal=0'
    assert expected in trace[0],(tag,trace[0])
   rows.append(row);(root/'model-results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
   selected=[r for r in rows if r['model']==model and r['context']==context]
   for field in ['loss','accuracy','pred_sha256']:assert len({r[field] for r in selected})==1,(model,context,field)
summary=[]
for model,context in configs:
 selected=[r for r in rows if r['model']==model and r['context']==context]
 row={'model':model,'context':context,'tokens':4096,'count_equal_calls_per_epoch':4096//context,'traces':{r['variant']:r['trace'] for r in selected if r['trial']==-1},'results_identical':True}
 for variant in variants:
  vals=[r['ms'] for r in selected if r['variant']==variant and r['trial']>=0];assert len(vals)==5
  row[variant]={'mean_ms':statistics.mean(vals),'sd_ms':statistics.stdev(vals),'processes':len(vals)}
 row['speedup_vs_upstream']=row['upstream']['mean_ms']/row['optimized']['mean_ms']
 row['speedup_vs_before']=row['before']['mean_ms']/row['optimized']['mean_ms'];summary.append(row)
(root/'model-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
