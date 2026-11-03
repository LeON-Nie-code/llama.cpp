import os,subprocess,json,hashlib
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
summary=[]
for model in ['e2b','e4b']:
 m=json.loads((p/f'{model}-model.json').read_text())
 for variant in ['baseline','candidate']:
  tag=f'{model}-{variant}-generation'
  e=os.environ.copy();e.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-mean-baseline-bin:' if variant=='baseline' else '')+'/tmp/llama-mean-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD='/tmp/llama-webgpu-count-equal-evidence/fullmodel/driver-pin.so')
  cmd=[str(p/'check-generation'),m['path'],str(p/tag)]
  with (p/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,(tag,r.returncode)
  print(tag,'passed',flush=True)
 a,b=[np.fromfile(p/f'{model}-{v}-generation.tokens.bin',dtype=np.int32) for v in ['baseline','candidate']]
 x,y=[np.fromfile(p/f'{model}-{v}-generation.logits.bin',dtype=np.float32).astype(float) for v in ['baseline','candidate']]
 assert len(a)==len(b)==128
 finite=np.isfinite(x)&np.isfinite(y)
 assert np.array_equal(np.isneginf(x),np.isneginf(y))
 row={'model':model,'generated_tokens':128,'identical_tokens':bool(np.array_equal(a,b)),'token_mismatches':int(np.count_nonzero(a!=b)),'logits_samples':'after prompt and before final generated token','finite_logit_values':int(finite.sum()),'max_abs_logit_difference':float(abs(x[finite]-y[finite]).max()),'logits_nmse':float(np.sum((x[finite]-y[finite])**2)/np.sum(x[finite]**2))}
 summary.append(row);(p/'generation-check-summary.json').write_text(json.dumps(summary,indent=2));print(row,flush=True)
