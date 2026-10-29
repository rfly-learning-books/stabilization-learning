"""Fresh RGB representation, finite-transition response, twin Q and goal metric.

Only training batches carry pose labels. Online control consumes encoded RGB
and a learned local response. M is optimized through its constrained candidate
distribution, as in the project's discrete Push certificate policy.
"""
from dataclasses import dataclass, asdict
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from stabilization_gym.tasks_gpu_rgb.certificate_control import (
    CertificateControlConfig, candidates, choose, value)


@dataclass(frozen=True)
class LearnerConfig:
    latent: int = 16
    hidden: int = 256
    learning_rate: float = .0003
    gamma: float = .995
    tau: float = .01
    entropy: float = .01
    calibration_weight: float = .2
    feasibility_weight: float = 1.
    decay: float = .05


def mlp(n, out, hidden):
    return nn.Sequential(nn.Linear(n, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out))


class Representation(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64*8*8, 256), nn.SiLU(), nn.Linear(256, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 256), nn.SiLU(), nn.Linear(256, 64*8*8), nn.SiLU(),
            nn.Unflatten(1, (64, 8, 8)), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())
        self.pose_head = mlp(latent*2, 4, 128)
        self.register_buffer('center', torch.zeros(latent))
        self.register_buffer('scale', torch.ones(()))
        self.register_buffer('calibrated', torch.tensor(False))

    def raw(self, image):
        return self.encoder(image.float()/255. if image.dtype == torch.uint8 else image)

    def forward(self, image):
        return (self.raw(image)-self.center)/self.scale


class GoalMetric(nn.Module):
    def __init__(self, d=16, hidden=128):
        super().__init__()
        self.d = d
        rows, cols = torch.tril_indices(d, d)
        self.register_buffer('rows', rows); self.register_buffer('cols', cols)
        self.register_buffer('diagonal', rows == cols)
        self.net = mlp(d, len(rows), hidden)
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[self.diagonal] = math.log(math.expm1(1.-.001))

    def forward(self, goal):
        raw = self.net(goal)
        entries = torch.where(self.diagonal, F.softplus(raw)+.001, raw)
        lower = raw.new_zeros(len(raw), self.d, self.d)
        lower[:, self.rows, self.cols] = entries
        gram = lower @ lower.transpose(-1, -2)
        tr = gram.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
        return .1*torch.eye(self.d, device=goal.device) + .9*self.d*gram/tr


class Response(nn.Module):
    def __init__(self, d, hidden):
        super().__init__(); self.d = d
        self.net = mlp(2*d, 4*d, hidden)

    def forward(self, error, goal):
        return self.net(torch.cat((error, goal), -1)).reshape(-1, self.d, 4)


class ActionValue(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        # Remaining time is public. Previous commands describe the rollout;
        # this first training protocol is P0, with no hidden actuator memory.
        self.net = mlp(2*d+4+1+4, 1, hidden)

    def forward(self, error, goal, previous, remaining, action):
        return self.net(torch.cat((error, goal, previous, remaining[:, None], action), -1)).squeeze(-1)


class RGBCertificateLearner:
    def __init__(self, config=None, device='cuda:0'):
        self.config = config or LearnerConfig(); c = self.config
        self.device = torch.device(device)
        self.features = Representation(c.latent).to(device)
        self.metric = GoalMetric(c.latent).to(device)
        self.initial_metric = deepcopy(self.metric).requires_grad_(False)
        self.response = Response(c.latent, c.hidden).to(device)
        self.q1 = ActionValue(c.latent, c.hidden).to(device)
        self.q2 = ActionValue(c.latent, c.hidden).to(device)
        self.target1 = deepcopy(self.q1).requires_grad_(False)
        self.target2 = deepcopy(self.q2).requires_grad_(False)
        self.ae_opt = torch.optim.Adam(self.features.parameters(), lr=c.learning_rate)
        self.model_opt = torch.optim.Adam(self.response.parameters(), lr=c.learning_rate)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=c.learning_rate)
        self.m_opt = torch.optim.Adam(self.metric.parameters(), lr=c.learning_rate)
        self.control_config = CertificateControlConfig(decay=c.decay)
        self.counts = dict(representation=0, response=0, critic=0, metric=0)

    @staticmethod
    def _step(loss, optimizer, parameters):
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(parameters), 5., error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    def warmup(self, batch):
        if bool(self.features.calibrated): raise RuntimeError('Representation version is frozen')
        x, goal = batch['image'].float()/255., batch['goal_image'].float()/255.
        raw, zg = self.features.raw(x), self.features.raw(goal)
        reconstruction = (F.mse_loss(self.features.decoder(raw), x)+F.mse_loss(self.features.decoder(zg), goal))*.5
        physical = batch['offset'].clone(); physical[:, 3] /= math.radians(25.)
        prediction = self.features.pose_head(torch.cat((raw-zg, zg), -1))
        pose = F.smooth_l1_loss(prediction, physical)
        radius = (raw-zg).square().sum(-1)
        physical_radius = physical.square().sum(-1)
        calibration = ((torch.log(radius+1e-4)-torch.log(physical_radius+1e-4)).square()).mean()
        spread = F.relu(.05-raw.std(0, unbiased=False)).square().mean()
        loss = reconstruction + .5*pose + .02*calibration + spread
        norm = self._step(loss, self.ae_opt, self.features.parameters())
        self.counts['representation'] += 1
        return dict(ae_loss=float(loss), reconstruction=float(reconstruction), pose=float(pose), feature_calibration=float(calibration), gradient=norm)

    @torch.no_grad()
    def freeze_features(self, batch):
        if bool(self.features.calibrated): raise RuntimeError('Already calibrated')
        z, zg = self.features.raw(batch['image']), self.features.raw(batch['goal_image'])
        energy = .5*(z-zg).square().sum(-1).mean()
        physical = batch['energy'].mean()
        if energy < 1e-12 or physical < 1e-12: raise ValueError('Degenerate feature calibration')
        self.features.center.copy_(torch.cat((z, zg)).mean(0))
        self.features.scale.copy_((energy/physical).sqrt().clamp_min(1e-4))
        self.features.calibrated.fill_(True)
        self.features.eval().requires_grad_(False)

    def distribution(self, error, goal, *, identity=False, initial_metric=False):
        model = self.initial_metric if initial_metric else self.metric
        return candidates(error, model(goal), self.response(error, goal), self.control_config, identity)

    @torch.no_grad()
    def act(self, image, goal_image, explore=False, variant='selected'):
        z, zg = self.features(image), self.features(goal_image)
        error = z-zg
        result = self.distribution(error, zg, identity=variant=='identity', initial_metric=variant=='initial_metric')
        action, diagnostics = choose(result, explore)
        return action, dict(**diagnostics, error=error, goal=zg, matrix=result['matrix'], value=result['before'])

    def response_update(self, b):
        prediction = .1*torch.einsum('bdi,bi->bd', self.response(b['error'], b['goal']), b['action'])
        actual = b['next_error']-b['error']
        loss = (prediction-actual).square().mean() / actual.detach().square().mean().clamp_min(1e-5)
        norm = self._step(loss, self.model_opt, self.response.parameters())
        cos = F.cosine_similarity(prediction.detach(), actual, dim=-1)
        self.counts['response'] += 1
        return dict(response_loss=float(loss), response_cosine=float(cos.mean()), response_opposite=float((cos<0).float().mean()), gradient=norm)

    def _all_q(self, network, b, result, successor=False):
        error = b['next_error'] if successor else b['error']
        previous = b['action'] if successor else b['previous']
        remaining = b['next_remaining'] if successor else b['remaining']
        k = result['actions'].shape[1]
        expand = lambda x: x[:, None].expand(-1, k, *x.shape[1:]).reshape(-1, *x.shape[1:])
        return network(expand(error), expand(b['goal']), expand(previous), expand(remaining), result['actions'].reshape(-1, 4)).reshape(len(error), k)

    def critic_update(self, b):
        with torch.no_grad():
            result = self.distribution(b['next_error'], b['goal'])
            q = torch.minimum(self._all_q(self.target1, b, result, True), self._all_q(self.target2, b, result, True))
            soft_value = (result['probability']*(q-self.config.entropy*result['logp'])).sum(-1)
            target = b['reward'] + self.config.gamma*(~b['terminal']).float()*soft_value
        args = (b['error'], b['goal'], b['previous'], b['remaining'], b['action'])
        first, second = self.q1(*args), self.q2(*args)
        loss = F.smooth_l1_loss(first, target)+F.smooth_l1_loss(second, target)
        norm = self._step(loss, self.q_opt, list(self.q1.parameters())+list(self.q2.parameters()))
        with torch.no_grad():
            for source, target_network in ((self.q1, self.target1), (self.q2, self.target2)):
                for p, target_p in zip(source.parameters(), target_network.parameters()): target_p.lerp_(p, self.config.tau)
        self.counts['critic'] += 1
        return dict(q_loss=float(loss), q_mean=float(first.mean()), target_mean=float(target.mean()), gradient=norm)

    def metric_update(self, b):
        # The response and Q models are held fixed during this block. Candidate
        # probabilities and nominal commands retain their derivatives through M.
        with torch.no_grad(): response = self.response(b['error'], b['goal'])
        matrix = self.metric(b['goal'])
        result = candidates(b['error'], matrix, response, self.control_config)
        # Q parameters stay fixed, while gradients through the M-dependent
        # nominal candidate actions remain part of the performance derivative.
        self.q1.requires_grad_(False); self.q2.requires_grad_(False)
        q = torch.minimum(self._all_q(self.q1, b, result), self._all_q(self.q2, b, result))
        performance = (result['probability']*(self.config.entropy*result['logp']-q)).sum(-1).mean()
        predicted_energy = value(b['error'], matrix)
        ratio = predicted_energy / b['energy'].clamp_min(1e-5)
        calibration = (F.relu(.2-ratio).square()+F.relu(ratio-5.).square()).mean()
        feasibility = (F.relu(result['residual'].min(-1).values) / predicted_energy.detach().clamp_min(1e-4)).square().mean()
        loss = performance + self.config.calibration_weight*calibration + self.config.feasibility_weight*feasibility
        try:
            norm = self._step(loss, self.m_opt, self.metric.parameters())
        finally:
            self.q1.requires_grad_(True); self.q2.requires_grad_(True)
        self.counts['metric'] += 1
        return dict(m_loss=float(loss), performance=float(performance), calibration=float(calibration), feasibility_loss=float(feasibility),
                    model_feasible_fraction=float(result['has_feasible'].float().mean()), VE_median=float(ratio.median()), gradient=norm)

    def payload(self):
        names = ('features','metric','initial_metric','response','q1','q2','target1','target2')
        return dict(schema='rgb_certificate_replay_v1', config=asdict(self.config), counts=self.counts.copy(),
                    models={k:getattr(self,k).state_dict() for k in names},
                    optimizers={k:getattr(self,k).state_dict() for k in ('ae_opt','model_opt','q_opt','m_opt')},
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [])

    def restore_models(self, payload):
        for k, state in payload['models'].items(): getattr(self,k).load_state_dict(state)
        self.features.eval().requires_grad_(False)
