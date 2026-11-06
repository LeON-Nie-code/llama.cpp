import json,os,subprocess,time,statistics,hashlib,wave
from pathlib import Path
p=Path(__file__).parent;root='/mnt/data2/rsi/llama-webgpu-pad-models/'
files=['chatterbox-turbo-Q8_0.gguf','mmproj-chatterbox-turbo-F16.gguf']
while not (Path(root)/'download.complete').exists():time.sleep(2)
rows=[]
for trial in [-1,0,1,2,3,4]:
 for variant in (['baseline','candidate'] if trial%2 else ['candidate','baseline']):
  tag=f'{variant}-{trial}';e=os.environ.copy();e.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_RUNTIME_DIR='/tmp',LD_LIBRARY_PATH=('/tmp/llama-pad-baseline-build/bin:' if variant=='baseline' else '')+'/tmp/llama-pad-build/bin:/tmp/llama-dawn/lib64',LD_PRELOAD='/tmp/llama-webgpu-count-equal-evidence/fullmodel/driver-pin.so')
  if trial==-1:e['LD_PRELOAD']+=':/tmp/pad-trace.so'
  cmd=['/tmp/llama-pad-evidence/tts-bench','-m',root+files[0],'-mm',root+files[1],'-p','Open source software helps researchers understand and improve the systems they use.','--tts-voice','default','-n','64','-ngl','99','-dev','WebGPU','-mmdev','WebGPU','-fa','off','-t','2','-tb','2','-c','1024','-b','512','-ub','512','-s','1234','--temp','0','-o',str(p/(tag+'.wav'))]
  with (p/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:print(tag,'FAILED',r.returncode,flush=True);raise SystemExit(r.returncode)
  import re
  s=(p/(tag+'.log')).read_text();m=re.search(r'BENCH_TTS input_s=([\d.]+) full_inference_s=([\d.]+)',s);assert m
  with wave.open(str(p/(tag+'.wav')),'rb') as w:pcm=w.readframes(w.getnframes());frames=w.getnframes()
  row={'variant':variant,'trial':trial,'input_s':float(m[1]),'full_s':float(m[2]),'samples':frames,'pcm_sha256':hashlib.sha256(pcm).hexdigest()};rows.append(row);(p/'results.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
summary={v:{'mean_s':statistics.mean(r['full_s'] for r in rows if r['variant']==v and r['trial']>=0),'sd_s':statistics.stdev(r['full_s'] for r in rows if r['variant']==v and r['trial']>=0)} for v in ['baseline','candidate']};(p/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
