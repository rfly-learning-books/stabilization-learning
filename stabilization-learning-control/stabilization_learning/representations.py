"""Representations owned by the learning library, passed into image tasks."""
import torch
from torch import nn


class IdentityRepresentation(nn.Module):
    preserves_physical_coordinates = True
    def forward(self, observation):
        return observation


class RGBAutoEncoder(nn.Module):
    """Small RGB AE; accepts NCHW floats in [0,1], outputs shared latent z."""
    def __init__(self, latent_dim=16, image_size=64):
        super().__init__()
        if image_size % 8:
            raise ValueError("image_size must be divisible by 8")
        self.latent_dim, self.image_size = latent_dim, image_size
        width = image_size // 8
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, 32, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(32 * width * width, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32 * width * width), nn.SiLU(),
            nn.Unflatten(1, (32, width, width)), nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, images):
        return self.encoder(images)

    def reconstruct(self, images):
        return self.decoder(self(images))


class DerivativeHead(nn.Module):
    """Signed instantaneous physical-cost derivative: D=a(z,z*) dot v.

    A shared difference makes a(z,z)=0 exactly. Targets must come from known
    kinematics or measured transitions, not from a learned certificate itself.
    """
    def __init__(self, latent_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))

    def coefficients(self, z, goal):
        return self.net(torch.cat((z, goal), -1)) - self.net(torch.cat((goal, goal), -1))

    def forward(self, z, goal, velocity):
        return (self.coefficients(z, goal) * velocity).sum(-1)


class RepresentationLearner:
    """AE-dominated representation training between complete policy batches.

    The auxiliary gradient on encoder parameters is norm-capped relative to the
    AE gradient. Decoder receives only reconstruction; head only supervision.
    No representation update is made inside a control episode or PPO batch.
    """
    def __init__(self, autoencoder, derivative_head=None, lr=1e-3, auxiliary_cap=.25):
        if not 0 <= auxiliary_cap <= 1 or lr <= 0:
            raise ValueError("Require auxiliary_cap in [0,1] and positive learning rate")
        self.autoencoder, self.head = autoencoder, derivative_head
        self.auxiliary_cap = float(auxiliary_cap)
        params = list(autoencoder.parameters()) + ([] if derivative_head is None else list(derivative_head.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def update(self, images, goals=None, velocities=None, derivative_targets=None):
        ae, head = self.autoencoder, self.head
        encoder, decoder = list(ae.encoder.parameters()), list(ae.decoder.parameters())
        z = ae(images)
        rec = torch.nn.functional.mse_loss(ae.decoder(z), images)
        rec_grads = torch.autograd.grad(rec, encoder + decoder, retain_graph=True)
        auxiliary = rec.new_zeros(())
        aux_grads, head_grads = [torch.zeros_like(p) for p in encoder], []
        if derivative_targets is not None:
            if head is None or goals is None or velocities is None:
                raise ValueError("Derivative supervision requires goals, velocities, targets and a head")
            auxiliary = torch.nn.functional.smooth_l1_loss(head(z, ae(goals), velocities), derivative_targets)
            grads = torch.autograd.grad(auxiliary, encoder + list(head.parameters()))
            aux_grads, head_grads = grads[:len(encoder)], grads[len(encoder):]
        ae_norm = torch.sqrt(sum(g.square().sum() for g in rec_grads[:len(encoder)]))
        aux_norm = torch.sqrt(sum(g.square().sum() for g in aux_grads))
        scale = torch.clamp(self.auxiliary_cap * ae_norm / (aux_norm + 1e-12), max=1.)
        self.optimizer.zero_grad(set_to_none=True)
        for p, g, ag in zip(encoder, rec_grads[:len(encoder)], aux_grads):
            p.grad = g + scale * ag
        for p, g in zip(decoder, rec_grads[len(encoder):]):
            p.grad = g
        if head is not None:
            for p, g in zip(head.parameters(), head_grads):
                p.grad = g
        self.optimizer.step()
        return {"reconstruction_loss": rec.item(), "derivative_loss": auxiliary.item(),
                "encoder_ae_grad_norm": ae_norm.item(), "encoder_aux_grad_norm": aux_norm.item(),
                "auxiliary_scale": scale.item(), "effective_aux_ratio": (scale * aux_norm / (ae_norm + 1e-12)).item()}
