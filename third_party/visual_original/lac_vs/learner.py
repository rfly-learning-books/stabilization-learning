from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.nn import functional as F

from . import legacy
from .controller import candidate_residuals, certificate_control
from .models import CostCritic, MetricActor, ema, frozen_parameters
from .vision import dct_reconstruction, frozen_features, image_tensor


def cost_td_target(cost, terminal, tail, gamma):
    return cost + gamma * (1 - terminal) * tail


class Learner:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.ae = legacy.autoencoder(cfg, self.device)
        latent = cfg["encoder"]["latent_dim"]
        self.actor = MetricActor(latent, cfg["metric"]["hidden"], cfg["metric"]["eigen_floor"],
                                 identity=cfg["ablation"] == "identity_m").to(device)
        self.critic = CostCritic(latent).to(device)
        self.target_ae = copy.deepcopy(self.ae).eval().requires_grad_(False)
        self.target_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        train = cfg["training"]
        self.representation_optimizer = torch.optim.Adam([
            {"params": self.ae.parameters(), "lr": train["encoder_lr"]},
            {"params": self.critic.parameters(), "lr": train["critic_lr"]}])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=train["actor_lr"])
        self.dual = float(train["dual_initial"])
        self.counters = {"warmup": 0, "representation": 0, "actor": 0}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _ae_loss(self, batch):
        x = image_tensor(self.cfg, batch["rgb"], self.device)
        goal = image_tensor(self.cfg, batch["goal_rgb"], self.device)
        both = torch.cat([x, goal], dim=0)
        encoded, reconstruction = self.ae(both)
        count = len(x)
        rec = F.mse_loss(reconstruction, both)
        dct = dct_reconstruction(reconstruction, both)
        latent = encoded.square().mean()
        weights = self.cfg["loss"]
        loss = weights["reconstruction"] * rec + weights["dct_reconstruction"] * dct + weights["latent_l2"] * latent
        return loss, encoded[:count], encoded[count:], {"reconstruction": rec, "dct": dct, "latent_l2": latent}

    def _target(self, batch):
        z, zg, jacobian = frozen_features(self.cfg, self.target_ae, batch["next_rgb"], batch["goal_rgb"],
                                          batch["next_depth"], batch["camera_from_body"], self.device)
        action = certificate_control(self.target_actor, z, zg, jacobian, **self.cfg["controller"])["action"].detach()
        with torch.no_grad():
            tail = self.target_critic(z, zg, action)
            # Timeout is a continuing-task sampling cutoff, not an absorbing failure.
            return cost_td_target(self.tensor(batch["cost"]), self.tensor(batch["terminal"]), tail,
                                  self.cfg["training"]["gamma"])

    def representation_loss(self, batch, warmup=False):
        ae_loss, z, zg, logs = self._ae_loss(batch)
        if warmup:
            return ae_loss, logs
        target = self._target(batch)
        # AE-only ablation still trains its critic; only critic -> encoder is cut.
        qz, qg = (z.detach(), zg.detach()) if self.cfg["ablation"] == "ae_only" else (z, zg)
        prediction = self.critic(qz, qg, self.tensor(batch["action"]))
        fit = F.mse_loss(prediction, target.detach())
        anchor = self.critic(qg, qg, torch.zeros(len(z), 4, device=self.device)).square().mean()
        loss = ae_loss + self.cfg["loss"]["critic"] * fit + self.cfg["loss"]["critic_goal_anchor"] * anchor
        return loss, {**logs, "critic_fit": fit, "critic_goal_anchor": anchor,
                      "critic_prediction": prediction.mean(), "critic_target": target.mean()}

    def actor_loss(self, batch):
        cfg = self.cfg
        z, zg, jacobian = frozen_features(cfg, self.ae, batch["rgb"], batch["goal_rgb"],
                                         batch["depth"], batch["camera_from_body"], self.device)
        zn, zgn, jn = frozen_features(cfg, self.ae, batch["next_rgb"], batch["goal_rgb"],
                                     batch["next_depth"], batch["camera_from_body"], self.device)
        with frozen_parameters(self.critic):
            current = certificate_control(self.actor, z, zg, jacobian, create_graph=True, **cfg["controller"])
            following = certificate_control(self.actor, zn, zgn, jn, create_graph=True, **cfg["controller"])
            current_cost = self.critic(z, zg, current["action"])
            following_cost = self.critic(zn, zgn, following["action"])
            with torch.no_grad():
                recorded_cost = self.critic(z, zg, self.tensor(batch["action"]))
            drift = ((1 - self.tensor(batch["terminal"])) * following_cost - recorded_cost
                     + cfg["training"]["alpha_cost"] * self.tensor(batch["cost"]))
            predicted_error = current["error"] + cfg["plant"]["dt"] * (jacobian @ current["action"].unsqueeze(-1)).squeeze(-1)
            future_value, _, _ = self.actor.certificate(predicted_error, zg)
            residual = (future_value - current["V"] + cfg["loss"]["decay_per_second"] * cfg["plant"]["dt"] * current["V"])
            descent = (F.relu(residual) / (current["V"].detach() + 1e-6)).square().mean()
            candidates = candidate_residuals(self.actor, current, zg, jacobian, cfg["plant"]["dt"])
            existence = F.relu(candidates.amin(-1) + cfg["loss"]["existence_margin"]).square().mean()
            perturbation = 0.01 * torch.randn_like(current["error"])
            _, nearby = self.actor.metric(current["error"] + perturbation, zg)
            metric_smooth = (nearby - current["M"]).square().mean()
            action_smooth = (current["action"] - self.tensor(batch["previous_action"])).square().mean()
            weights = cfg["loss"]
            dual = 0.0 if cfg["ablation"] == "critic_ae_no_drift" else self.dual
            loss = (weights["actor_cost"] * current_cost.mean() + dual * drift.mean()
                    + weights["predicted_descent"] * descent + weights["metric_smoothness"] * metric_smooth
                    + weights["action_smoothness"] * action_smooth + weights["descent_existence"] * existence)
        return loss, {"actor_cost": current_cost.mean(), "lac_drift": drift.mean(), "predicted_descent": descent,
                      "metric_smoothness": metric_smooth, "action_smoothness": action_smooth,
                      "descent_existence": existence,
                      "V": current["V"].mean(), "positive_Vdot": (current["Vdot"] > 1e-8).float().mean()}

    def update(self, replay, phase):
        actor_phase = phase == "actor"
        optimizer = self.actor_optimizer if actor_phase else self.representation_optimizer
        # Clear all modules to make leakage audits meaningful, including after phase switches.
        for module in (self.ae, self.actor, self.critic):
            module.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        self.ae.eval() if actor_phase else self.ae.train()
        self.actor.train()
        logs = []
        accumulation = self.cfg["training"]["accumulation"]
        for _ in range(accumulation):
            batch = replay.sample(self.cfg["training"]["batch_size"])
            loss, row = self.actor_loss(batch) if actor_phase else self.representation_loss(batch, warmup=phase == "warmup")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {phase} loss")
            (loss / accumulation).backward()
            logs.append({"total": float(loss.detach()), **{k: float(v.detach()) for k, v in row.items()}})
        parameters = list(self.actor.parameters()) if actor_phase else list(self.ae.parameters()) + list(self.critic.parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg["training"]["gradient_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite {phase} gradient")
        if actor_phase:
            if any(p.grad is not None for p in self.ae.parameters()) or any(p.grad is not None for p in self.critic.parameters()):
                raise RuntimeError("Actor loss leaked into encoder or critic parameters")
        elif any(p.grad is not None for p in self.actor.parameters()):
            raise RuntimeError("Representation loss leaked into certificate actor")
        optimizer.step()
        means = {key: float(np.mean([row[key] for row in logs])) for key in logs[0]}
        for name, module in (("encoder", self.ae.encoder), ("critic", self.critic), ("actor", self.actor)):
            terms = [p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
            means[f"{name}_gradient_norm"] = float(torch.stack(terms).sum().sqrt()) if terms else 0.0
        if actor_phase and self.cfg["ablation"] not in ("critic_ae_no_drift", "identity_m"):
            self.dual = float(np.clip(self.dual + self.cfg["training"]["dual_lr"] * means["lac_drift"],
                                      0.0, self.cfg["training"]["dual_max"]))
        rate = self.cfg["training"]["target_ema"]
        if actor_phase:
            ema(self.target_actor, self.actor, rate)
        else:
            ema(self.target_ae, self.ae, rate)
            if phase != "warmup":
                ema(self.target_critic, self.critic, rate)
        self.counters[phase] += 1
        return {**means, "gradient_norm": float(norm), "dual": self.dual}

    def payload(self):
        return {name: getattr(self, name).state_dict() for name in
                ("ae", "actor", "critic", "target_ae", "target_actor", "target_critic",
                 "representation_optimizer", "actor_optimizer")} | {"dual": self.dual, "counters": self.counters}

    def restore(self, payload):
        for name in ("ae", "actor", "critic", "target_ae", "target_actor", "target_critic",
                     "representation_optimizer", "actor_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        self.dual, self.counters = payload["dual"], payload["counters"]
