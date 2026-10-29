"""Physical one-sided quasi-static disk; no learned parameters live here."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class DiskConfig:
    radius: float = .05
    action_max: float = .05
    dt: float = .1
    kappa: float = 1.
    response: str = "constant"
    kappa_upper_factor: float = 1.5
    tolerance: float = .001
    hold_seconds: float = 1.
    horizon_seconds: float = 20.

    def __post_init__(self):
        values = [self.radius, self.action_max, self.dt, self.kappa,
                  self.kappa_upper_factor, self.tolerance, self.hold_seconds, self.horizon_seconds]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Physical parameters must be positive and finite")
        if self.response not in ("constant", "varying"):
            raise ValueError("Unknown response model")
        if self.kappa_upper_factor < (1.4 if self.response == "varying" else 1.):
            raise ValueError("Declared response upper bound does not cover this plant")

    @property
    def kappa_max(self):
        return self.kappa*self.kappa_upper_factor

    def asdict(self):
        return asdict(self)


def value(x):
    return .5*np.asarray(x, dtype=float)**2


class DiskEnv:
    """Exact ZOH transitions; contact side is fixed throughout a rollout.

    The varying-response experiment freezes the sampled state/action gain
    over each hold. This is a declared bounded discrete-model mismatch, not
    a rigid-body contact or image-based simulator.
    """
    def __init__(self, config=DiskConfig()):
        self.config = config

    def safe_upper(self, x):
        c = self.config
        return np.maximum(0., np.minimum(c.action_max, -np.asarray(x)/(c.kappa_max*c.dt)))

    def gain(self, x, action):
        c = self.config
        x, action = np.broadcast_arrays(np.asarray(x, float), np.asarray(action, float))
        if c.response == "constant":
            return np.full_like(x, c.kappa)
        return c.kappa*(1.+.25*np.sin(2*np.pi*(-x/c.radius))+.15*action/c.action_max)

    def step(self, x, action):
        x, action = np.broadcast_arrays(np.asarray(x, float), np.asarray(action, float))
        if not (np.isfinite(x).all() and np.isfinite(action).all()):
            raise FloatingPointError("Nonfinite physical state/action")
        if np.any(action < -1e-12) or np.any(action > self.config.action_max+1e-12):
            raise ValueError("Physical input constraint violated")
        return x+self.config.dt*self.gain(x, action)*action

    def in_domain(self, x):
        return (np.asarray(x) >= -self.config.radius-1e-12) & (np.asarray(x) <= 1e-12)


def transitions(env, count, rng, excitation="random", *, center=None, noise=0.):
    """Exactly count labeled transitions from the training-only physical model."""
    if count < 1 or noise < 0:
        raise ValueError("Positive sample count and nonnegative noise required")
    c = env.config
    if center is None:
        x = -rng.uniform(0., c.radius, count)
        near = rng.random(count) < .2
        x[near] = -rng.uniform(1e-7*c.radius, .08*c.radius, near.sum())
        boundary = rng.random(count) < .1
        x[boundary] = -rng.uniform(.9*c.radius, c.radius, boundary.sum())
    else:
        x = np.clip(rng.choice(np.asarray(center), count)+rng.normal(0., .06*c.radius, count), -c.radius, -1e-8*c.radius)
    upper = env.safe_upper(x)
    if excitation == "feedback":
        a = np.minimum(-x, upper)
    elif excitation == "multigain":
        a = np.minimum(-x*rng.choice([.2,.5,1.,2.,4.], count), upper)
    elif excitation == "random":
        a = upper*rng.uniform(.01, 1., count)
    else:
        raise ValueError(excitation)
    following = env.step(x, a)
    return dict(x=x, action=a, next_x=following,
                observed_x=x+rng.normal(0., noise, count),
                observed_next_x=following+rng.normal(0., noise, count),
                true_D=value(following)-value(x), gain=env.gain(x,a))


def concatenate(a, b):
    return {key: np.concatenate([a[key], b[key]]) for key in a}


def fixed_starts(config, count, rng):
    # Include the exact goal, near-goal and far boundary in every split;
    # only random interior starts are independent across val/test splits.
    fixed = np.array([0., -config.tolerance/2, -config.tolerance*1.1, -config.radius])
    return np.r_[fixed[:count], -rng.uniform(config.tolerance*1.2, config.radius, max(0,count-len(fixed)))]
