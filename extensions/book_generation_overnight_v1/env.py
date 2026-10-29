"""A deterministic DDIM environment with explicit local x0 corrections."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPTokenizer


class GenerationEnv:
    """The action corrects predicted x0 while the current epsilon is held fixed.

    z_next = sqrt(alpha_next) * (x0 + action) + sqrt(1-alpha_next)*epsilon.
    This local proxy is not a certificate for the subsequent DDIM trajectory.
    """

    def __init__(self, assets: Path, size=512, steps=20, cfg=7.5, device="cuda:0"):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The generation experiment requires an available CUDA GPU")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            str(assets/"sd15"), torch_dtype=torch.float16, variant="fp16",
            local_files_only=True, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.enable_attention_slicing("auto")
        self.clip = CLIPModel.from_pretrained(str(assets/"clip"), local_files_only=True).to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(str(assets/"clip"), local_files_only=True)
        for model in (self.pipe.unet, self.pipe.vae, self.pipe.text_encoder, self.clip):
            model.eval().requires_grad_(False)
        self.size, self.steps, self.cfg = size, steps, cfg
        self.mean = torch.tensor([.48145466,.4578275,.40821073],device=self.device).view(1,3,1,1)
        self.std = torch.tensor([.26862954,.26130258,.27577711],device=self.device).view(1,3,1,1)
        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(steps, device=self.device)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.counts = {}
        self.reset_counts()

    def reset_counts(self):
        self.counts = dict(unet_calls=0, vae_decode_calls=0, proxy_calls=0, gradient_calls=0)

    @torch.no_grad()
    def reset(self, prompt, seed):
        self.reset_counts()
        self.prompt, self.seed = prompt, seed
        tokens = self.pipe.tokenizer(["",prompt], padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        self.embedding = self.pipe.text_encoder(tokens.input_ids.to(self.device))[0]
        tokens = self.tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(self.device)
        self.text_features = F.normalize(self.clip.get_text_features(**tokens),dim=-1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn((1,4,self.size//8,self.size//8),generator=generator,device=self.device,dtype=torch.float32)

    @torch.no_grad()
    def predict(self, latent, index):
        timestep = self.scheduler.timesteps[index]
        model_input = latent.repeat(2,1,1,1).to(torch.float16)
        epsilon = self.pipe.unet(model_input,timestep,encoder_hidden_states=self.embedding).sample.float()
        self.counts["unet_calls"] += 1
        unconditional, conditional = epsilon.chunk(2)
        epsilon = unconditional + self.cfg*(conditional-unconditional)
        alpha = self.alphas[int(timestep)]
        x0 = (latent-(1-alpha).sqrt()*epsilon)/alpha.sqrt()
        prev_t = int(timestep)-self.scheduler.config.num_train_timesteps//self.steps
        previous_alpha = self.alphas[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod.to(self.device)
        return x0, epsilon, previous_alpha

    def decode(self, x0):
        self.counts["vae_decode_calls"] += 1
        return (self.pipe.vae.decode(x0.float()/self.pipe.vae.config.scaling_factor).sample*.5+.5).clamp(0,1)

    def proxy(self, x0):
        pixels = self.decode(x0)
        pixels = F.interpolate(pixels, size=(224,224),mode="bicubic",align_corners=False,antialias=True)
        features = F.normalize(self.clip.get_image_features(pixel_values=(pixels-self.mean)/self.std),dim=-1)
        self.counts["proxy_calls"] += 1
        return (1-(features*self.text_features).sum(-1)).mean()

    def local_gradient(self, x0):
        point = x0.detach().requires_grad_(True)
        with torch.enable_grad():
            value = self.proxy(point)
            gradient = torch.autograd.grad(value,point)[0]
        self.counts["gradient_calls"] += 1
        rms = gradient.square().mean().sqrt()
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite local semantic gradient")
        direction = -gradient/rms.clamp_min(1e-12)
        return float(value), direction.detach(), float(rms)

    @staticmethod
    def advance(corrected_x0, epsilon, previous_alpha):
        return (previous_alpha.sqrt()*corrected_x0+(1-previous_alpha).sqrt()*epsilon).detach()

    @staticmethod
    def features(index, steps, proxy, grad_rms, x0, epsilon):
        return np.array([index/max(1,steps-1), proxy, np.log10(max(grad_rms,1e-12)),
                         float(x0.square().mean().sqrt()), float(epsilon.square().mean().sqrt())],np.float32)

    def sync(self):
        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def image(self, latent):
        array = self.decode(latent)[0].permute(1,2,0).cpu().numpy()
        return np.uint8(np.clip(array*255+.5,0,255))

    def close(self):
        del self.pipe, self.clip, self.tokenizer
        torch.cuda.empty_cache()
