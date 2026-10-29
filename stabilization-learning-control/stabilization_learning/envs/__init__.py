"""Compatibility facade plus learner-owned representation loading."""
from stabilization_gym.envs import *
from stabilization_gym.envs import make as _make, available_envs, __getattr__


def make(env_id, **kwargs):
    checkpoint = kwargs.pop("representation_checkpoint", None)
    if checkpoint is not None:
        if env_id != "visual_servo" or "representation" in kwargs:
            raise ValueError("A representation checkpoint is only supported for visual_servo, without another phi")
        import torch
        from ..representations import RGBAutoEncoder
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        phi = RGBAutoEncoder(state["latent_dim"], state["image_size"])
        phi.load_state_dict(state["autoencoder"])
        kwargs.setdefault("image_size", state["image_size"])
        kwargs.setdefault("error_dim", state["latent_dim"])
        kwargs["representation"] = phi.eval().requires_grad_(False)
    return _make(env_id, **kwargs)
