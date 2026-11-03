from pathlib import Path
import os,subprocess,json,numpy as np
p=Path(__file__).resolve().parent;rows=[]
for model in ['e2b','e4b']:
 m=json.loads((p/f'{model}-model.json').read_text());prefix=str(p/f'{model}-candidate-teacher')
 e=os.environ.copy();e.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH='/tmp/llama-mean-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD='/tmp/llama-webgpu-count-equal-evidence/fullmodel/driver-pin.so')
 cmd=[str(p/'check-generation'),m['path'],prefix,str(p/f'{model}-baseline-generation.tokens.bin')]
 with (p/f'{model}-candidate-teacher.log').open('w') as f:r=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0
 a,b=[np.fromfile(x,dtype=np.int32) for x in [p/f'{model}-baseline-generation.tokens.bin',p/f'{model}-candidate-teacher.tokens.bin']]
 x,y=[np.fromfile(path,dtype=np.float32).reshape(2,-1).astype(float) for path in [p/f'{model}-baseline-generation.logits.bin',p/f'{model}-candidate-teacher.logits.bin']]
 row={'model':model,'teacher_forced_tokens':128,'top1_matches':int((a==b).sum()),'compared_logit_positions':[]}
 for i in range(2):
  mask=np.isfinite(x[i])&np.isfinite(y[i]);d=x[i][mask]-y[i][mask]
  row['compared_logit_positions'].append({'generated_position':0 if i==0 else 127,'max_abs_difference':float(abs(d).max()),'nmse':float(np.sum(d*d)/np.sum(x[i][mask]**2))})
 rows.append(row);(p/'teacher-check-summary.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
