import subprocess,os,json,time,re
from pathlib import Path
p=Path(__file__).resolve().parent;rows=[]
for model in ['e2b','e4b']:
 m=json.loads((p.parent/'high-frequency-models'/f'{model}-model.json').read_text())
 for variant in ['baseline','candidate']:
  e=os.environ.copy();e.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-mean-baseline-bin:' if variant=='baseline' else '')+'/tmp/llama-mean-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD='/tmp/llama-webgpu-count-equal-evidence/fullmodel/driver-pin.so')
  cmd=['/tmp/llama-mean-build/bin/llama-perplexity','-m',m['path'],'-f','/tmp/llama-count-equal-models/frequent/wikitext-2-raw/wiki.valid.raw','--chunks','4','-c','512','-b','512','-ub','512','-ngl','99','-dev','WebGPU','-fa','off','-t','2','-tb','2']
  tag=f'{model}-{variant}-ppl';(p/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'env':{k:e[k] for k in ['LD_LIBRARY_PATH','LD_PRELOAD','XDG_RUNTIME_DIR','VK_ICD_FILENAMES']}},indent=2))
  t=time.monotonic()
  with (p/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,(tag,r.returncode)
  s=(p/(tag+'.log')).read_text();matches=re.findall(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)',s);assert matches,tag
  row={'model':model,'variant':variant,'ppl':float(matches[-1][0]),'reported_uncertainty':float(matches[-1][1]),'wall_s':time.monotonic()-t};rows.append(row);(p/'ppl-summary.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
