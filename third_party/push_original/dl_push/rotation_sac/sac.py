"""Small GPU-native SAC implementation for the rotation PhysX task."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.distributions import Normal


LOG_TWO_PI = math.log(2.0 * math.pi)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim=2):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.SiLU(),
                                  nn.Linear(256, 256), nn.SiLU())
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)

    def distribution(self, obs):
        h = self.body(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5.0, 1.5)
        return Normal(mean, log_std.exp())

    def sample(self, obs):
        dist = self.distribution(obs)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(-1), dist

    def deterministic(self, obs):
        return torch.tanh(self.mean(self.body(obs)))


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, 256), nn.LayerNorm(256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, obs, action):
        return self.net(torch.cat((obs, action), -1)).squeeze(-1)


class Replay:
    def __init__(self, capacity, obs_dim, device='cpu'):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity, self.obs_dim, self.device = int(capacity), int(obs_dim), device
        self.obs = torch.empty((capacity, obs_dim), dtype=torch.float32)
        self.next_obs = torch.empty_like(self.obs)
        self.action = torch.empty((capacity, 2), dtype=torch.float32)
        self.reward = torch.empty(capacity, dtype=torch.float32)
        self.terminated = torch.empty(capacity, dtype=torch.float32)
        self.truncated = torch.empty(capacity, dtype=torch.float32)
        self.success = torch.empty(capacity, dtype=torch.float32)
        self.cursor = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated, truncated, success=None):
        n = len(action)
        if n > self.capacity:
            raise ValueError('batch larger than replay capacity')
        ids = (torch.arange(n) + self.cursor) % self.capacity
        self.obs[ids] = obs.detach().cpu()
        self.next_obs[ids] = next_obs.detach().cpu()
        self.action[ids] = action.detach().cpu()
        self.reward[ids] = reward.detach().cpu()
        self.terminated[ids] = terminated.detach().float().cpu()
        self.truncated[ids] = truncated.detach().float().cpu()
        self.success[ids] = (torch.zeros(n) if success is None else success.detach().float().cpu())
        self.cursor = int((self.cursor + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size, device):
        # Rare successful terminals carry the only strong discrete signal. Keep
        # them visible without turning this into behavior cloning.
        ids = torch.randint(self.size, (batch_size,))
        positive = self.success[:self.size].nonzero().flatten()
        if len(positive) and batch_size >= 4:
            count = max(1, batch_size // 4)
            ids[:count] = positive[torch.randint(len(positive), (count,))]
        return tuple(x[ids].to(device, non_blocking=True) for x in
                     (self.obs, self.action, self.reward, self.next_obs, self.terminated, self.truncated))


@dataclass
class SACConfig:
    gamma: float = .99
    tau: float = .005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float = -2.0
    min_alpha: float = .05
    batch_size: int = 512
    updates_per_vector_step: int = 8
    learning_starts: int = 10_000
    gradient_clip: float = 5.0


class SAC:
    def __init__(self, obs_dim, device, config=None, seed=20260911):
        torch.manual_seed(seed)
        self.device = device
        self.config = config or SACConfig()
        c = self.config
        if c.min_alpha <= 0 or c.batch_size < 1 or c.updates_per_vector_step < 1:
            raise ValueError('Invalid SAC configuration')
        self.actor = Actor(obs_dim).to(device)
        self.q1 = Critic(obs_dim).to(device)
        self.q2 = Critic(obs_dim).to(device)
        self.q1_target = Critic(obs_dim).to(device)
        self.q2_target = Critic(obs_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.actor_lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=c.critic_lr)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=c.alpha_lr)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().clamp(self.config.min_alpha, 10.)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        return self.actor.deterministic(obs) if deterministic else self.actor.sample(obs)[0]

    def update(self, replay):
        c = self.config
        obs, action, reward, next_obs, terminated, truncated = replay.sample(c.batch_size, self.device)
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            target_q = torch.minimum(self.q1_target(next_obs, next_action),
                                     self.q2_target(next_obs, next_action)) - self.alpha.detach() * next_logp
            # Time-limit truncation bootstraps; success/failure termination does not.
            target = reward + c.gamma * (1.0 - terminated) * target_q
        q1, q2 = self.q1(obs, action), self.q2(obs, action)
        q_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True); q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), c.gradient_clip)
        self.q_opt.step()
        new_action, logp, _ = self.actor.sample(obs)
        actor_q = torch.minimum(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (self.alpha.detach() * logp - actor_q).mean()
        self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.gradient_clip); self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (logp + c.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(self.config.min_alpha), math.log(10.))
        with torch.no_grad():
            for source, target_network in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for source_param, target_param in zip(source.parameters(), target_network.parameters()):
                    target_param.lerp_(source_param, c.tau)
        self.updates += 1
        return dict(q_loss=float(q_loss.detach()), actor_loss=float(actor_loss.detach()),
                    alpha_loss=float(alpha_loss.detach()), alpha=float(self.alpha.detach()),
                    mean_q=float(torch.minimum(q1, q2).mean().detach()), updates=self.updates)

    def checkpoint(self, step, env_steps, shape, args, reward_history):
        return dict(format='rotation_sac_physx_v1', route='rotation_sac_physx_v1',
                    from_scratch=True, planner_guided=False, demonstrations=None,
                    shape=shape, step=int(step), env_steps=int(env_steps), updates=self.updates,
                    actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
                    q1_target=self.q1_target.state_dict(), q2_target=self.q2_target.state_dict(),
                    actor_optimizer=self.actor_opt.state_dict(), q_optimizer=self.q_opt.state_dict(),
                    log_alpha=self.log_alpha.detach().cpu(), alpha_optimizer=self.alpha_opt.state_dict(),
                    sac_config=self.config.__dict__, args=args, reward_history=reward_history,
                    certified=False)
