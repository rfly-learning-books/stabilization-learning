"""Download pinned public weights into an isolated experiment asset folder."""
from pathlib import Path
import json
import fnmatch
import hashlib
import requests
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/book_generation_20261029'
SPECS=[
    ('sd15','stable-diffusion-v1-5/stable-diffusion-v1-5','451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
     ['model_index.json','scheduler/*','tokenizer/*','text_encoder/config.json','text_encoder/model.fp16.safetensors',
      'unet/config.json','unet/diffusion_pytorch_model.fp16.safetensors','vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors','LICENSE*','README.md']),
    ('clip','openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
     ['config.json','pytorch_model.bin','preprocessor_config.json','tokenizer*.json','vocab.json','merges.txt','special_tokens_map.json','README.md']),
    ('siglip','google/siglip-base-patch16-224','7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed',
     ['config.json','model.safetensors','preprocessor_config.json','tokenizer*.json','spiece.model','special_tokens_map.json','README.md'])]

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    for name,repo,revision,patterns in SPECS:
        print(json.dumps(dict(event='downloading',name=name,repo=repo,revision=revision)),flush=True)
        # Direct pinned GET also works through proxies that strip the Hub's
        # optional HEAD metadata. TLS validation remains enabled.
        response=requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=30)
        response.raise_for_status();metadata=response.json()
        if metadata['sha'] != revision:
            raise RuntimeError('Pinned model revision mismatch')
        for entry in metadata['siblings']:
            relative=entry['rfilename']
            if not any(fnmatch.fnmatch(relative,pattern) for pattern in patterns):
                continue
            root=(OUT/name).resolve();path=(root/relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Invalid model asset path')
            path.parent.mkdir(parents=True,exist_ok=True)
            expected=entry.get('lfs',{}).get('sha256')
            if path.exists() and (not expected or hashlib.sha256(path.read_bytes()).hexdigest()==expected):
                continue
            temporary=path.with_name(path.name+'.download')
            print(json.dumps(dict(event='file',name=name,file=relative,bytes=entry.get('size'))),flush=True)
            with requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/{relative}?download=true',stream=True,timeout=(30,120)) as response:
                response.raise_for_status();digest=hashlib.sha256();count=0;tick=time.time()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(4*1024*1024):
                        if not block:continue
                        stream.write(block);digest.update(block);count+=len(block)
                        if time.time()-tick>20:
                            print(json.dumps(dict(event='progress',name=name,file=relative,mib=count/2**20)),flush=True);tick=time.time()
            if expected and digest.hexdigest()!=expected:
                raise RuntimeError('Downloaded weight checksum mismatch: '+relative)
            if entry.get('size') is not None and count!=entry['size']:
                raise RuntimeError('Downloaded asset size mismatch: '+relative)
            temporary.replace(path)
        print(json.dumps(dict(event='downloaded',name=name)),flush=True)
    (OUT/'MODEL_MANIFEST.json').write_text(json.dumps([dict(name=n,repo=r,revision=v,patterns=p) for n,r,v,p in SPECS],indent=2),encoding='utf-8')
