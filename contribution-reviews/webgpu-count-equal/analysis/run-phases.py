from pathlib import Path
import json,os,re,subprocess
root=Path(__file__).resolve().parent;helpers=root.parent/'fullmodel'
if not helpers.is_dir():helpers=root.parent
models={'stories260K':'/tmp/llama-count-equal-models/stories260K.gguf','smol135':'/tmp/llama-count-equal-models/frequent/SmolLM2-135M-Instruct-Q4_K_M.gguf','qwen05':'/tmp/llama-count-equal-models/frequent/qwen2.5-0.5b-instruct-q4_k_m.gguf'}
rows=[]
for model,path in models.items():
 for variant in ['baseline','pr']:
  tag=model+'-'+variant
  env=os.environ.copy();env.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-count-equal-baseline-build/bin:' if variant=='baseline' else '')+'/tmp/llama-count-equal-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD=str(helpers/'driver-pin.so')+':'+str(root/'phases.so'))
  cmd=[str(root/'eval-phases'),path,str(root/(tag+'.bin')),'1','256','16','/tmp/llama-count-equal-models/frequent/wikitext-2-raw/wiki.valid.raw']
  (root/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['LD_LIBRARY_PATH','LD_PRELOAD','VK_ICD_FILENAMES','XDG_RUNTIME_DIR']}},indent=2))
  with (root/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError((tag,p.returncode))
  log=(root/(tag+'.log')).read_text()
  row={'model':model,'variant':variant,'meta':re.search(r'PHASE_META .*',log)[0], 'epoch_ms':float(re.search(r'EVAL epoch=0 tokens=4096 ms=([\d.]+)',log)[1]),'phases':{m[1]:{'ms':float(m[2]),'calls':int(m[3]),'bytes':int(m[4])} for m in re.finditer(r'PHASE name=(\S+) ms=([\d.]+) calls=(\d+) bytes=(\d+)',log)}}
  assert ('cpu_count=16 gpu_count=0' if variant=='baseline' else 'cpu_count=0 gpu_count=16') in row['meta']
  rows.append(row);(root/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
