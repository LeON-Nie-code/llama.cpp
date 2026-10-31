from pathlib import Path
import os,subprocess,json,re,statistics
root=Path(__file__).resolve().parent
rows=[]
for trial in range(3):
 for variant in (['before','optimized'] if trial%2==0 else ['optimized','before']):
  env=os.environ.copy();env.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-count-equal-original-backend:' if variant=='before' else '')+'/tmp/llama-count-equal-build/bin')
  cmd=['/tmp/llama-count-equal-build/bin/test-backend-ops','perf','-b','WebGPU','-o','COUNT_EQUAL']
  tag=f'operator-{variant}-{trial}'
  (root/(tag+'.command.json')).write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['LD_LIBRARY_PATH','VK_ICD_FILENAMES','XDG_RUNTIME_DIR']}},indent=2))
  with (root/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,tag
  log=(root/(tag+'.log')).read_text()
  found=list(re.finditer(r'COUNT_EQUAL\(ne=\[(\d+),1,1,1\].*?:\s+(\d+) runs -\s+([\d.]+) us/run',log))
  assert len(found)==6,(tag,len(found))
  for m in found:rows.append({'variant':variant,'trial':trial,'n':int(m[1]),'runs':int(m[2]),'us':float(m[3])})
  (root/'operator-results.json').write_text(json.dumps(rows,indent=2));print(tag,[(int(m[1]),float(m[3])) for m in found],flush=True)
summary=[]
for n in [256,1024,1025,4096,65536,1048576]:
 row={'n':n}
 for variant in ['before','optimized']:
  vals=[r['us'] for r in rows if r['n']==n and r['variant']==variant]
  row[variant]={'mean_us':statistics.mean(vals),'sd_us':statistics.stdev(vals),'trials':len(vals)}
 row['speedup']=row['before']['mean_us']/row['optimized']['mean_us'];summary.append(row)
(root/'operator-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
