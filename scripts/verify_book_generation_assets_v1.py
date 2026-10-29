"""Load pinned generation assets on CPU without using the active RGB GPU."""
from pathlib import Path
import gc
import json
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
from diffusers import StableDiffusionPipeline,DDIMScheduler
from transformers import CLIPModel,CLIPTokenizer,SiglipModel,AutoProcessor
root=Path(__file__).resolve().parents[1]
assets=root/'assets/book_generation_20261029'
output=root/'artifacts/book_generation_assets_cpu_20261029'
output.mkdir(exist_ok=False)
torch.set_num_threads(1)
result={}
pipe=StableDiffusionPipeline.from_pretrained(str(assets/'sd15'),torch_dtype=torch.float16,variant='fp16',
    local_files_only=True,safety_checker=None,feature_extractor=None,requires_safety_checker=False)
scheduler=DDIMScheduler.from_config(pipe.scheduler.config);scheduler.set_timesteps(20)
result['sd15_components_loaded']=True
result['ddim_timesteps']=scheduler.timesteps.tolist()
del pipe;gc.collect()
model=CLIPModel.from_pretrained(str(assets/'clip'),local_files_only=True).eval()
tokenizer=CLIPTokenizer.from_pretrained(str(assets/'clip'),local_files_only=True)
with torch.no_grad():
    features=model.get_text_features(**tokenizer(['A green bowl.'],return_tensors='pt'))
result['clip_text_shape']=list(features.shape)
del model,tokenizer;gc.collect()
model=SiglipModel.from_pretrained(str(assets/'siglip'),local_files_only=True).eval()
processor=AutoProcessor.from_pretrained(str(assets/'siglip'),local_files_only=True)
from PIL import Image
inputs=processor(text=['A green bowl.'],images=Image.new('RGB',(224,224),(0,128,0)),padding='max_length',return_tensors='pt')
with torch.no_grad():score=model(**inputs).logits_per_image
result['siglip_forward_finite']=bool(torch.isfinite(score).all())
result['scope']='CPU load and API validation only; this is not a generation-quality experiment'
(output/'VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
