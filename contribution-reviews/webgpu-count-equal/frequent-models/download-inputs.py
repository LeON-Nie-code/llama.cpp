from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request,zipfile

root=Path('/tmp/llama-count-equal-models/frequent')
root.mkdir(parents=True,exist_ok=True)
manifest=json.loads((Path(__file__).resolve().parent/'inputs.json').read_text())
checks={x['file']:x['sha256'] for x in manifest['files']}
models={
 'stories260K.gguf':'https://huggingface.co/ggml-org/tiny-llamas/resolve/def3e2dd70df35ecbf6403ea347de4c5977220c1/stories260K.gguf',
 'SmolLM2-135M-Instruct-Q4_K_M.gguf':'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q4_K_M.gguf',
 'qwen2.5-0.5b-instruct-q4_k_m.gguf':'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf',
}
def download(item):
 name,url=item
 target=(root.parent if name=='stories260K.gguf' else root)/name
 if not target.exists():urllib.request.urlretrieve(url,target)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==checks[name],name
 print(name,'verified',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,models.items()))
archive=root/'wikitext-2-raw-v1.zip'
if not archive.exists():urllib.request.urlretrieve(manifest['dataset_url'],archive)
name='wikitext-2-raw/wiki.valid.raw'
with zipfile.ZipFile(archive) as z:data=z.read(name)
assert hashlib.sha256(data).hexdigest()==checks[name],name
(root/name).parent.mkdir(parents=True,exist_ok=True)
(root/name).write_bytes(data)
print(name,'verified')
