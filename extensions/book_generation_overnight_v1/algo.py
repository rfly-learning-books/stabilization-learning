"""Frozen-step guidance policies and a supervised local improvement critic."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn


class LocalImprovementCritic(nn.Module):
    """Predict all four local proxy improvements; no return or test labels."""
    def __init__(self, actions):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,actions))
        self.register_buffer("mean",torch.zeros(5))
        self.register_buffer("scale",torch.ones(5))
        self.register_buffer("target_scale",torch.ones(1))

    def forward(self, value):
        return self.net((value-self.mean)/self.scale)*self.target_scale


def fit(data, validation, action_count, steps, seed, device, guard):
    torch.manual_seed(seed)
    model=LocalImprovementCritic(action_count).to(device)
    x=torch.as_tensor(data["features"],device=device); y=torch.as_tensor(data["improvements"],device=device)
    vx=torch.as_tensor(validation["features"],device=device); vy=torch.as_tensor(validation["improvements"],device=device)
    model.mean.copy_(x.mean(0)); model.scale.copy_(x.std(0,unbiased=False).clamp_min(1e-4))
    model.target_scale.copy_(y.std(unbiased=False).clamp_min(1e-5))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    generator=torch.Generator(device=device).manual_seed(seed+771)
    history=[]; best=None; best_loss=float("inf"); best_update=0
    for update in range(1,steps+1):
        guard()
        indices=torch.randint(len(x),(min(64,len(x)),),generator=generator,device=device)
        predicted=model(x[indices]); loss=((predicted-y[indices])/model.target_scale).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        if update==1 or update%10==0 or update==steps:
            with torch.no_grad():
                train_mse=float((model(x)-y).square().mean())
                val_mse=float((model(vx)-vy).square().mean())
                val_regret=float((vy.max(1).values-vy.gather(1,model(vx).argmax(1)[:,None]).squeeze(1)).mean())
            history.append(dict(update=update,train_mse=train_mse,val_mse=val_mse,val_local_regret=val_regret))
            if val_mse<best_loss:
                best=copy.deepcopy(model.state_dict());best_loss=val_mse;best_update=update
    model.load_state_dict(best);model.eval().requires_grad_(False)
    return model,history,dict(selected_update=best_update,selection="minimum held-out prompt validation MSE",val_mse=best_loss)


def choose(method, env, x0, index, epsilon, strengths, critic=None):
    """Return corrected x0 plus an auditable bounded local action."""
    if method=="original":
        with torch.no_grad(): proxy=float(env.proxy(x0))
        direction=torch.zeros_like(x0);rms=0.
    else:
        proxy,direction,rms=env.local_gradient(x0)
    feature=env.features(index,env.steps,proxy,rms,x0,epsilon)
    attempts=[]
    if method=="original":
        strength=0.;after=proxy
    elif method=="fixed":
        strength=strengths[2]
        with torch.no_grad(): after=float(env.proxy(x0+strength*direction))
        attempts.append(dict(strength=strength,proxy=after))
    elif method=="backtracking":
        strength=0.;after=proxy
        for candidate in (strengths[-1],strengths[-1]/2,strengths[-1]/4,strengths[-1]/8):
            with torch.no_grad(): value=float(env.proxy(x0+candidate*direction))
            attempts.append(dict(strength=candidate,proxy=value))
            if value<=proxy-1e-5:
                strength=candidate;after=value;break
    elif method=="learned":
        with torch.no_grad():
            prediction=critic(torch.as_tensor(feature,device=env.device)[None])[0]
            # Zero is a known no-op, even when its learned estimate is imperfect.
            prediction[0]=0.
            selected=int(prediction.argmax())
            strength=strengths[selected]
            after=proxy if strength==0 else float(env.proxy(x0+strength*direction))
        attempts.append(dict(strength=strength,proxy=after,predicted_improvements=prediction.cpu().tolist()))
    else:
        raise ValueError(method)
    correction=strength*direction
    return (x0+correction).detach(),dict(proxy_before=proxy,proxy_after=after,local_delta=after-proxy,
        gradient_rms=rms,strength=strength,correction_rms=float(correction.square().mean().sqrt()),
        attempts=attempts,local_descent=bool(after<=proxy+1e-7)),feature
