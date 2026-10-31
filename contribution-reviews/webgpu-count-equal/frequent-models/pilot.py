from pathlib import Path
import os,subprocess,json,re,hashlib
root=Path(__file__).resolve().parent;old=root.parent/'fullmodel'
if not old.is_dir():old=root.parent
models={'smol135':'/tmp/llama-count-equal-models/frequent/SmolLM2-135M-Instruct-Q4_K_M.gguf','qwen05':'/tmp/llama-count-equal-models/frequent/qwen2.5-0.5b-instruct-q4_k_m.gguf'}
rows=[]
for name,path in models.items():
 for variant in ['baseline','pr']:
  tag=f'pilot-{name}-{variant}'
  env=os.environ.copy();env.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-count-equal-baseline-build/bin:' if variant=='baseline' else '')+'/tmp/llama-count-equal-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD=str(old/'driver-pin.so')+':'+str(old/'trace-ops.so'))
  cmd=[str(root/'eval-llm'),path,str(root/(tag+'.bin')),'1','256','8','/tmp/llama-count-equal-models/frequent/wikitext-2-raw/wiki.valid.raw']
  (root/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['LD_LIBRARY_PATH','LD_PRELOAD','VK_ICD_FILENAMES','XDG_RUNTIME_DIR']}},indent=2))
  with (root/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError(f'{tag}: exit {p.returncode}')
  log=(root/(tag+'.log')).read_text();trace=re.search(r'OP_TRACE .*',log)
  assert trace,tag
  expected='webgpu_count_equal=16 cpu_count_equal=0' if variant=='pr' else 'webgpu_count_equal=0 cpu_count_equal=16'
  assert expected in trace[0],(tag,trace[0])
  row={'model':name,'variant':variant,'trace':trace[0],'eval':[x for x in log.splitlines() if x.startswith('EVAL')],'pred_sha256':hashlib.sha256((root/(tag+'.bin')).read_bytes()).hexdigest()}
  rows.append(row);(root/'pilot-results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
for name in models:
 assert len({r['pred_sha256'] for r in rows if r['model']==name})==1,name
print('GATE PASSED: both complete models execute COUNT_EQUAL once per validation batch, CPU in baseline and GPU in candidate; identical predictions.',flush=True)
