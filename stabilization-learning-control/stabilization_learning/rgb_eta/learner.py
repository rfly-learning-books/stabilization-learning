"""Twin discounted-return critics Q(context, eta), with a goal-metric actor.

Eta is the raw Cholesky parameter vector, never a physical command. The
deterministic actor and Gaussian behavior exploration follow a TD3-style
off-policy update (no SAC entropy term). The fixed encoder version is part of
the replay contract because changing it also changes eta -> physical action.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from stabilization_gym.tasks_gpu_rgb.eta_control import metric_from_eta, command, value, p0_step


@dataclass(frozen=True)
class EtaConfig:
    dim: int = 16
    hidden: int = 256
    q_lr: float = 1e-4
    m_lr: float = 1e-4
    gamma: float = .995
    tau: float = .005
    exploration_sigma: float = .08
    performance_weight: float = .2
    calibration_weight: float = 1.
    replay_descent_weight: float = .5
    physical_descent_weight: float = 2.
    direction_weight: float = .5
    smoothness_weight: float = .01
    matrix_regularization: float = .001
    value_lower: float = .25
    value_upper: float = 4.
    objective_floor: float = .0004
    decay_per_second: float = .1


def mlp(n, out, hidden):
    return nn.Sequential(nn.Linear(n, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out))


class EtaMetric(nn.Module):
    def __init__(self, dim=16):
        super().__init__(); self.dim = dim
        self.net = mlp(dim, dim*(dim+1)//2, 128)
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)
        rows, cols = torch.tril_indices(dim, dim)
        with torch.no_grad():
            self.net[-1].bias[rows == cols] = math.log(math.expm1(1.-1e-3))

    def forward(self, goal):
        return self.net(goal)


class EtaReturnCritic(nn.Module):
    def __init__(self, dim=16, hidden=256):
        super().__init__(); self.eta_dim = dim*(dim+1)//2
        # J specifies the fixed visual controller geometry; previous command
        # is public. Pose labels, reward and executed velocity are not inputs.
        self.net = mlp(2*dim + 4*dim + 4 + self.eta_dim, 1, hidden)

    def forward(self, error, goal, interaction, previous, eta):
        if eta.shape[-1] != self.eta_dim:
            raise ValueError('Critic action is eta, not four-dimensional velocity')
        x = torch.cat((error, goal/4., interaction.flatten(-2), previous, eta), -1)
        return self.net(x).squeeze(-1)


def td_target(reward, failure, next_q, gamma):
    # Rollout time limits and the end of the extra holding observation are
    # sampling cutoffs of a continuing task. They retain the actual successor
    # and bootstrap. Only a physical absorbing failure ends this return.
    return reward + gamma*(~failure).to(reward.dtype)*next_q


class EtaReturnLearner:
    def __init__(self, config=None, device='cuda:0', feature_version='unbound'):
        self.config = config or EtaConfig(); c = self.config
        self.device, self.feature_version = torch.device(device), feature_version
        self.metric = EtaMetric(c.dim).to(device)
        self.initial_metric = deepcopy(self.metric).requires_grad_(False)
        self.target_metric = deepcopy(self.metric).requires_grad_(False)
        self.q1 = EtaReturnCritic(c.dim, c.hidden).to(device)
        self.q2 = EtaReturnCritic(c.dim, c.hidden).to(device)
        self.target1 = deepcopy(self.q1).requires_grad_(False)
        self.target2 = deepcopy(self.q2).requires_grad_(False)
        self.q_opt = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=c.q_lr)
        self.m_opt = torch.optim.Adam(self.metric.parameters(), lr=c.m_lr)
        self.counts = dict(critic=0, metric=0)

    def _check(self, batch):
        if batch.get('feature_version') != self.feature_version:
            raise ValueError('Replay encoder version changes the eta-to-command mapping')
        if not bool(batch['eta_valid'].all()):
            raise ValueError('Random physical-action warmup is not an eta-action transition')

    @staticmethod
    def _step(loss, optimizer, parameters):
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite objective')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(parameters), 1., error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    @torch.no_grad()
    def act(self, error, goal, interaction, explore=False, variant='selected'):
        net = self.initial_metric if variant == 'initial_metric' else self.metric
        eta = net(goal)
        if explore: eta = eta + torch.randn_like(eta)*self.config.exploration_sigma
        _, matrix = metric_from_eta(eta, self.config.dim)
        if variant == 'identity': matrix = torch.eye(self.config.dim, device=eta.device).expand_as(matrix)
        action, h = command(error, interaction, matrix)
        return action, dict(eta=eta, matrix=matrix, value=value(error, matrix), hessian=h)

    def critic_update(self, b):
        self._check(b)
        with torch.no_grad():
            eta_next = self.target_metric(b['goal'])
            args = (b['next_error'], b['goal'], b['interaction'], b['action'], eta_next)
            qn = torch.minimum(self.target1(*args), self.target2(*args))
            target = td_target(b['reward'], b['failure'], qn, self.config.gamma)
        args = (b['error'], b['goal'], b['interaction'], b['previous'], b['eta'])
        q1, q2 = self.q1(*args), self.q2(*args)
        loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        norm = self._step(loss, self.q_opt, [*self.q1.parameters(), *self.q2.parameters()])
        self.counts['critic'] += 1
        self._polyak(self.q1, self.target1); self._polyak(self.q2, self.target2)
        return dict(q_loss=float(loss), q_mean=float(q1.mean()), target_mean=float(target.mean()),
            q_disagreement=float((q1-q2).abs().mean()), gradient=norm,
            physical_failures=int(b['failure'].sum()), sampling_cutoffs=int(b['cutoff'].sum()))

    @torch.no_grad()
    def _polyak(self, source, target):
        for p, q in zip(source.parameters(), target.parameters()): q.lerp_(p, self.config.tau)

    def objective_terms(self, b):
        c = self.config
        eta = self.metric(b['goal']); _, m = metric_from_eta(eta, c.dim)
        e, en, E, En = b['error'], b['next_error'], b['energy'], b['next_energy']
        v, vn = value(e, m), value(en, m)
        def cal(vv, ee):
            den = ee.detach()+c.objective_floor
            return ((F.relu(c.value_lower*ee-vv)/den).square() + (F.relu(vv-c.value_upper*ee)/den).square()).mean()
        calibration = .5*(cal(v,E)+cal(vn,En))
        descent = (F.relu(vn-v+c.decay_per_second*.1*E)/(E.detach()+c.objective_floor)).square().mean()
        u, h = command(e, b['interaction'], m)
        nxt = p0_step(b['offset'], u)
        scale = nxt.new_tensor([1.,1.,1.,math.radians(25.)])
        Ep = (nxt/scale).square().sum(-1)
        rate = c.decay_per_second*.1
        physical = (F.relu(Ep-E+rate*E)/(E+c.objective_floor)).square().mean()
        p, pn = b['offset'][:,:3].square().sum(-1), nxt[:,:3].square().sum(-1)
        position = (F.relu(pn-p+rate*p)/(p+c.objective_floor)).square().mean()
        r, dr = b['offset']/scale, (nxt-b['offset'])/scale
        cosine = (r*dr).sum(-1)/(r.norm(dim=-1)*dr.norm(dim=-1)+1e-6)
        direction = F.relu(cosine+.15).square().mean()
        q = torch.minimum(self.q1(e,b['goal'],b['interaction'],b['previous'],eta),
                          self.q2(e,b['goal'],b['interaction'],b['previous'],eta))
        terms = dict(performance=-q.mean(), calibration=calibration, replay_descent=descent,
            physical_descent=physical+position, direction=direction,
            smoothness=(u-b['previous']).square().mean(),
            matrix_regularization=(m-torch.eye(c.dim,device=m.device)).square().mean())
        weights = dict(performance=c.performance_weight, calibration=c.calibration_weight,
            replay_descent=c.replay_descent_weight, physical_descent=c.physical_descent_weight,
            direction=c.direction_weight, smoothness=c.smoothness_weight, matrix_regularization=c.matrix_regularization)
        weighted = {k:weights[k]*v for k,v in terms.items()}
        auxiliary = dict(V=float(v.mean()), VE_median=float((v/(E+1e-8)).median()),
            physical_delta=float((Ep-E).mean()), action_replay_distance=float((u-b['action']).norm(dim=-1).mean()),
            eta_replay_distance=float((eta-b['eta']).norm(dim=-1).mean()),
            eigen_min=float(torch.linalg.eigvalsh(m.detach()).min()), eigen_max=float(torch.linalg.eigvalsh(m.detach()).max()),
            hessian_condition=float(torch.linalg.cond(h.detach()).max()))
        return terms, weighted, auxiliary

    def metric_update(self, b, gradient_audit=False):
        self._check(b)
        self.q_opt.zero_grad(set_to_none=True)
        self.q1.requires_grad_(False); self.q2.requires_grad_(False)
        try:
            terms, weighted, extra = self.objective_terms(b)
            if gradient_audit:
                params = list(self.metric.parameters()); vectors = {}
                for name, loss in weighted.items():
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    vectors[name] = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(params,grads)])
                ref = vectors['performance']
                for name, grad in vectors.items():
                    extra['gradient_'+name] = float(grad.norm())
                    extra['cosine_to_return_'+name] = float(F.cosine_similarity(ref[None],grad[None]))
            loss = sum(weighted.values())
            norm = self._step(loss, self.m_opt, self.metric.parameters())
        finally:
            self.q1.requires_grad_(True); self.q2.requires_grad_(True)
        self._polyak(self.metric, self.target_metric)
        self.counts['metric'] += 1
        return dict(m_loss=float(loss), gradient=norm, **{k:float(v) for k,v in terms.items()}, **extra)

    def payload(self):
        names = ('metric','initial_metric','target_metric','q1','q2','target1','target2')
        return dict(schema='rgb_eta_return_v1', config=asdict(self.config), feature_version=self.feature_version,
            counts=self.counts.copy(), models={k:getattr(self,k).state_dict() for k in names},
            optimizers={k:getattr(self,k).state_dict() for k in ('q_opt','m_opt')},
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [])

    def restore(self, payload):
        if payload['feature_version'] != self.feature_version: raise ValueError('Frozen coordinate mismatch')
        for k, state in payload['models'].items(): getattr(self,k).load_state_dict(state)
