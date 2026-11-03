from pathlib import Path
import os,subprocess,json,time,statistics,re
root=Path('/home/leon/code/tmp/penguin/llama-webgpu-count-equal/current-preview');b=Path('/tmp/llama-count-equal-current-build/bin')
e=os.environ.copy();e.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=str(b)+':/tmp/llama-dawn/lib64',LD_PRELOAD=str(root.parent/'fullmodel/driver-pin.so'),GGML_NLOOP='3',GGML_N_THREADS='1')
jobs=[('webgpu',[str(b/'test-backend-ops'),'test','-b','WebGPU','-o','COUNT_EQUAL,ARGMAX,SUM,SUM_ROWS','-j','2']),('cpu',[str(b/'test-backend-ops'),'test','-b','CPU','-o','COUNT_EQUAL']),('exact',[str(root/'validate-count-equal')]),('ctest-opt-and-tensors',['ctest','--test-dir',str(b.parent),'-C','Release','--output-on-failure','-R','^(test-opt|test-backend-tensor)$','--timeout','120'])]
rows=[]
for tag,cmd in jobs:
 start=time.monotonic()
 with (root/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
 rows.append({'check':tag,'argv':cmd,'returncode':p.returncode,'seconds':time.monotonic()-start});(root/'checks.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
 assert p.returncode==0,tag
perfs=[]
for trial in range(3):
 cmd=[str(b/'test-backend-ops'),'perf','-b','WebGPU','-o','COUNT_EQUAL']
 with (root/f'perf-{trial}.log').open('w') as f:p=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0
 s=(root/f'perf-{trial}.log').read_text()
 matches=list(re.finditer(r'COUNT_EQUAL\(ne=\[(\d+),1,1,1\].*?:\s+(\d+) runs -\s+([\d.]+) us/run',s));assert len(matches)==6
 perfs.extend({'trial':trial,'n':int(x[1]),'runs':int(x[2]),'us':float(x[3])} for x in matches)
(root/'perf-results.json').write_text(json.dumps(perfs,indent=2))
summary=[]
for n in sorted({x['n'] for x in perfs}):
 vals=[x['us'] for x in perfs if x['n']==n];summary.append({'n':n,'mean_us':statistics.mean(vals),'sd_us':statistics.stdev(vals)})
(root/'perf-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
