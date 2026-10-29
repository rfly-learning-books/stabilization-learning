"""Strict, versioned experiment configuration; paths resolve from project root."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

from .runtime import ROOT, json_hash, read_json


@dataclass
class ObserverConfig:
    feature_dim: int = 64
    geometry_dim: int = 32
    hidden_dim: int = 64
    gain: str = "constant"
    eta: float = 0.8
    eta_min: float = 0.05
    eta_max: float = 1.5
    epsilon: float = 0.1
    radius: float = 20.0
    static_residual_head: bool = False
    static_residual_limit: float = 4.0
    adaptation: bool = True


@dataclass
class ControlConfig:
    window: int = 10
    delta: float = 0.001
    alpha: float = 0.001
    beta: float = 0.01
    smooth_weight: float = 0.01
    slack_budget: float = 0.0
    slack_strategy: str = "scheduled"
    slack_per_step_max: float = 0.0
    slack_activation_energy: float = 0.0
    terminal_instantaneous_energy: float = 0.0
    contact_target_weight: float = 0.0
    contact_target_scale: float = 0.10
    contact_target_turn_gain: float = 0.15
    contact_target_activation_energy: float = 0.0
    contact_target_score_weight: float = 0.0
    contact_target_score_activation_energy: float = 0.0
    samples_per_prior: int = 4
    prior_std_scale: float = 1.0
    prior_std_floor: float = 0.0
    prior_std_scales: list[float] = field(default_factory=list)
    anchor_speed: float = 0.06
    warmup_steps: int = 40
    contact_gap: float = 0.008
    position_scale: float = 0.10
    orientation_weight: float = 1.0
    linear_velocity_weight: float = 0.02
    angular_velocity_weight: float = 0.01
    pusher_velocity_weight: float = 0.02
    radial_speeds: list[float] = field(default_factory=list)
    radial_directions: int = 16
    geometric_candidates: bool = False
    selection: str = "action"
    reach_weight: float = 0.0
    start_when_feasible: bool = False
    refinement_rounds: int = 0
    refinement_sigma: float = 0.05
    multistep_horizon: int = 1
    multistep_horizons: list[int] = field(default_factory=list)
    multistep_samples_per_prior: int = 0
    multistep_proposal: str = "prior"
    multistep_random_sequences: int = 0
    multistep_random_blocks: int = 4
    multistep_robust_margin: float = 0.0
    multistep_correction_mode: str = "full"
    contact_recovery_steps: int = 0
    contact_recovery_clearance: float = 0.01
    contact_recovery_speed: float = 0.15
    contact_recovery_turn_gain: float = 0.06
    contact_recovery_tolerance: float = 0.015
    settle_object_speed: float = 0.03
    settle_object_omega: float = 0.30
    settle_pusher_speed: float = 0.03
    stagnation_recovery: bool = False
    stagnation_min_drop: float = 0.0
    stagnation_cooldown_steps: int = 20
    stagnation_activation_energy: float = 0.0
    stagnation_min_admitted_steps: int = 0
    # Phase-based control (new controller path; disabled by default).
    phase_control_enabled: bool = False
    phase_contact_force_threshold: float = 0.05
    phase_terminal_energy: float = 0.02
    phase_terminal_pose_m: float = 0.02
    phase_terminal_angle_rad: float = 0.10
    phase_transition_hold: int = 2
    phase_approach_gap: float = 0.008
    phase_contact_lookback: int = 4
    # CEM proposal for the push phase.
    cem_sequences: int = 256
    cem_iterations: int = 4
    cem_elite_fraction: float = 0.10
    cem_horizon: int = 8
    cem_random_blocks: int = 3
    cem_sigma: float = 0.05
    cem_min_sigma: float = 0.002
    cem_input_weight: float = 0.10
    cem_smooth_weight: float = 0.05
    cem_contact_weight: float = 0.0
    cem_clf_weight: float = 10.0

    @property
    def candidate_count(self):
        return (5 * (1 + self.samples_per_prior) + 7
                + self.radial_directions * len(self.radial_speeds)
                + (4 if self.geometric_candidates else 0))


@dataclass
class EnvironmentConfig:
    num_envs: int = 2
    reset_mode: str = "contact_pose"
    episode_steps: int = 120
    # Explicit domain variations; never silently change reference defaults.
    masses: list[float] = field(default_factory=lambda: [0.8])
    frictions: list[float] = field(default_factory=lambda: [3.0])
    contact_threshold_n: float = 0.05
    terminate_on_success: bool = True


@dataclass
class TrainingConfig:
    iterations: int = 10
    episodes_per_shape: int = 4
    updates_per_iteration: int = 40
    batch_episodes: int = 4
    learning_rate: float = 0.0003
    gradient_clip: float = 5.0
    tbptt_steps: int = 64
    contact_weight: float = 4.0
    validation_fraction: float = 0.25
    behavior_switch_steps: int = 8
    prior_sample_probability: float = 0.25
    initial_collection: str = "prior"
    later_collection: str = "clf"
    lk_weight: float = 0.0
    lk_window: int = 10
    lk_contraction: float = 0.95
    lk_tolerance: float = 0.0
    zero_action_blocks: bool = False
    vectorized_replay: bool = True
    lyapunov_difference_weight: float = 0.0
    lyapunov_difference_scale: float = 0.10
    clf_boundary_weight: float = 0.0
    clf_error_scale: float = 0.05
    clf_boundary_epsilon: float = 0.05
    selection_metric: str = "state"
    realizability_weight: float = 0.0
    realizability_ridge: float = 0.01
    realizability_norm_weight: float = 0.01
    dissipation_weight: float = 0.0
    allow_observer_upgrade: bool = False
    trainable_modules: list[str] = field(default_factory=lambda: ["all"])


@dataclass
class EvaluationConfig:
    position_tolerance: float = 0.001
    angle_tolerance: float = 0.017453292519943295
    velocity_tolerance: float = 0.005
    angular_velocity_tolerance: float = 0.02
    input_tolerance: float = 0.005
    hold_steps: int = 100


@dataclass
class Config:
    schema: int = 1
    seed: int = 20260909
    device: str = "cpu"
    mjx_backend: str = "cpu"
    asset_root: str = "assets/native_v1"
    train_shapes: list[str] = field(default_factory=lambda: [
        "T", "Square", "RectWide", "Plus", "H", "C_wide", "C_narrow"
    ])
    validation_shapes: list[str] = field(default_factory=lambda: ["L", "Z"])
    test_shapes: list[str] = field(default_factory=list)
    # Optional JSON defining extra independent box-compound truth geometries.
    shape_catalog: str | None = None
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self):
        if self.schema != 1:
            raise ValueError("Unsupported configuration schema")
        if self.mjx_backend not in {"cpu", "cuda"}:
            raise ValueError("mjx_backend must be cpu or cuda")
        if not self.train_shapes or len(set(self.train_shapes)) != len(self.train_shapes):
            raise ValueError("Training shapes must be nonempty and unique")
        if any(len(s) != len(set(s)) for s in (self.validation_shapes, self.test_shapes)):
            raise ValueError("Duplicate split entries")
        sets = list(map(set, (self.train_shapes, self.validation_shapes, self.test_shapes)))
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Training, validation and test geometry splits must be disjoint")
        o, c, e, t = self.observer, self.control, self.environment, self.training
        if o.gain not in {"constant", "learned"}:
            raise ValueError("gain must be constant or learned")
        if not (0 < o.eta_min <= o.eta <= o.eta_max < 2):
            raise ValueError("Require 0 < eta_min <= eta <= eta_max < 2")
        if min(o.feature_dim, o.geometry_dim, o.hidden_dim) < 1 or min(
            o.epsilon, o.radius, o.static_residual_limit
        ) <= 0:
            raise ValueError("Invalid observer dimensions or normalization")
        if c.window < 1 or c.warmup_steps < c.window - 1 or c.samples_per_prior < 0:
            raise ValueError("Invalid history, warmup or candidate budget")
        if c.prior_std_scale <= 0 or c.prior_std_floor < 0:
            raise ValueError("Prior sampling scale must be positive and floor nonnegative")
        if any(value <= 0 for value in c.prior_std_scales):
            raise ValueError("Mixture standard-deviation multipliers must be positive")
        if min(c.alpha, c.beta, c.delta, c.position_scale) <= 0 or c.slack_budget < 0:
            raise ValueError("Invalid CLF coefficients")
        if c.slack_strategy not in {"scheduled", "ledger"}:
            raise ValueError("Unknown slack strategy")
        if min(
            c.slack_per_step_max, c.slack_activation_energy,
            c.terminal_instantaneous_energy,
            c.contact_target_weight, c.contact_target_turn_gain,
            c.contact_target_activation_energy,
            c.contact_target_score_weight,
            c.contact_target_score_activation_energy,
        ) < 0:
            raise ValueError("Slack limits must be nonnegative")
        if c.contact_target_scale <= 0:
            raise ValueError("Contact-target scale must be positive")
        if (c.contact_target_weight > 0
                and c.contact_target_activation_energy <= 0):
            raise ValueError("Contact-target shaping needs a positive activation energy")
        if (c.contact_target_score_weight > 0
                and c.contact_target_score_activation_energy <= 0):
            raise ValueError("Contact-target scoring needs a positive activation energy")
        if (c.slack_strategy == "ledger" and c.slack_budget > 0
                and (c.slack_per_step_max <= 0 or c.slack_activation_energy <= 0)):
            raise ValueError("Ledger slack needs positive per-step and activation limits")
        if min(c.orientation_weight, c.linear_velocity_weight,
               c.angular_velocity_weight, c.pusher_velocity_weight) <= 0:
            raise ValueError("Task and terminal-velocity weights must be positive")
        if not 0 < c.anchor_speed <= 0.30 or c.smooth_weight < 0 or c.contact_gap < 0:
            raise ValueError("Invalid candidate settings")
        if c.radial_directions < 1 or any(not 0 < v <= .30 for v in c.radial_speeds):
            raise ValueError("Invalid radial action coverage")
        if c.selection not in {"action", "energy"} or c.reach_weight < 0:
            raise ValueError("Invalid control selection or reach energy")
        if c.refinement_rounds < 0 or c.refinement_sigma <= 0:
            raise ValueError("Invalid same-state candidate refinement")
        if (c.multistep_horizon < 1 or c.multistep_samples_per_prior < 0
                or c.multistep_robust_margin < 0):
            raise ValueError("Invalid finite-step CLF configuration")
        if (any(value <= 1 for value in c.multistep_horizons)
                or c.multistep_horizons != sorted(set(c.multistep_horizons))
                or (c.multistep_horizons
                    and c.multistep_horizons[-1] != c.multistep_horizon)):
            raise ValueError(
                "Finite-step horizons must be sorted unique values above one "
                "and end at multistep_horizon"
            )
        if c.multistep_correction_mode not in {"full", "first", "none"}:
            raise ValueError("Unknown finite-step correction mode")
        if c.multistep_proposal not in {"prior", "random"}:
            raise ValueError("Unknown finite-step proposal family")
        if c.multistep_random_sequences < 0 or c.multistep_random_blocks < 1:
            raise ValueError("Invalid random-shooting configuration")
        if (c.multistep_proposal == "random" and c.multistep_horizon > 1
                and c.multistep_random_sequences < 2):
            raise ValueError("Random shooting needs at least two sequences")
        if c.contact_recovery_steps < 0 or c.contact_recovery_clearance < 0:
            raise ValueError("Invalid contact-recovery budget or clearance")
        if (not 0 < c.contact_recovery_speed <= .30
                or c.contact_recovery_turn_gain < 0
                or c.contact_recovery_tolerance <= 0):
            raise ValueError("Invalid contact-recovery action settings")
        if min(
            c.settle_object_speed, c.settle_object_omega,
            c.settle_pusher_speed,
        ) <= 0:
            raise ValueError("Settling thresholds must be positive")
        if min(
            c.phase_contact_force_threshold,
            c.phase_terminal_energy,
            c.phase_terminal_pose_m,
            c.phase_terminal_angle_rad,
            c.phase_approach_gap,
        ) <= 0:
            raise ValueError("Phase thresholds must be positive")
        if c.phase_transition_hold < 1 or c.phase_contact_lookback < 1:
            raise ValueError("Phase transition/lookback counts must be positive")
        if c.phase_control_enabled:
            if min(
                c.cem_iterations, c.cem_horizon, c.cem_random_blocks,
            ) < 1 or c.cem_sequences < 2:
                raise ValueError("CEM dimensions must be positive")
            if not 0 < c.cem_elite_fraction <= 1:
                raise ValueError("CEM elite fraction must be in (0,1]")
            if min(
                c.cem_sigma, c.cem_min_sigma,
                c.cem_input_weight, c.cem_smooth_weight,
                c.cem_clf_weight,
            ) <= 0:
                raise ValueError("CEM cost weights and scales must be positive")
            if c.cem_contact_weight < 0:
                raise ValueError("CEM contact weight must be nonnegative")
            if c.cem_min_sigma > c.cem_sigma:
                raise ValueError("CEM minimum sigma cannot exceed initial sigma")
        if (min(c.stagnation_min_drop, c.stagnation_activation_energy) < 0
                or min(c.stagnation_cooldown_steps,
                       c.stagnation_min_admitted_steps) < 0):
            raise ValueError("Stagnation progress threshold must be nonnegative")
        if c.stagnation_recovery and c.stagnation_activation_energy <= 0:
            raise ValueError("Stagnation recovery needs a positive activation energy")
        ev = self.evaluation
        if min(ev.position_tolerance, ev.angle_tolerance, ev.velocity_tolerance,
               ev.angular_velocity_tolerance, ev.input_tolerance, ev.hold_steps) <= 0:
            raise ValueError("Evaluation thresholds must be positive")
        if e.num_envs < 1 or e.episode_steps < 2:
            raise ValueError("Need positive lane count and at least two steps")
        if e.reset_mode not in {"full", "contact_pose", "contact_translation"}:
            raise ValueError("Unknown reset distribution")
        if not e.masses or not e.frictions or min(e.masses) <= 0 or min(e.frictions) < 0:
            raise ValueError("Invalid physical parameter distribution")
        if min(t.iterations, t.episodes_per_shape, t.updates_per_iteration,
               t.batch_episodes, t.tbptt_steps, t.behavior_switch_steps) < 1:
            raise ValueError("Invalid training counts")
        if not 0 < t.validation_fraction < 1 or min(t.learning_rate, t.gradient_clip, t.contact_weight) <= 0:
            raise ValueError("Invalid training hyperparameters")
        if not 0 <= t.prior_sample_probability <= 1:
            raise ValueError("prior_sample_probability must be in [0,1]")
        if t.initial_collection not in {"prior", "phase"} or t.later_collection not in {"prior", "clf", "phase"}:
            raise ValueError("Collection behavior must be prior, clf, or phase")
        if t.lk_window < 1 or t.lk_weight < 0 or not 0 < t.lk_contraction <= 1 or t.lk_tolerance < 0:
            raise ValueError("Invalid optional empirical window regularizer")
        if min(t.lyapunov_difference_weight, t.clf_boundary_weight) < 0:
            raise ValueError("Joint observer loss weights must be nonnegative")
        if min(t.lyapunov_difference_scale, t.clf_error_scale, t.clf_boundary_epsilon) <= 0:
            raise ValueError("Joint observer loss scales must be positive")
        if t.selection_metric not in {"state", "joint"}:
            raise ValueError("selection_metric must be state or joint")
        if (t.lyapunov_difference_weight or t.clf_boundary_weight) and not t.vectorized_replay:
            raise ValueError("Joint observer loss currently requires vectorized_replay")
        if t.selection_metric == "joint" and not (
            t.lyapunov_difference_weight or t.clf_boundary_weight
            or t.realizability_weight or t.dissipation_weight
        ):
            raise ValueError("Joint selection requires a nonzero joint observer loss")
        if min(
            t.realizability_weight, t.realizability_norm_weight,
            t.dissipation_weight,
        ) < 0:
            raise ValueError("Realizability loss weights must be nonnegative")
        if t.realizability_ridge <= 0:
            raise ValueError("Realizability ridge must be positive")
        allowed_modules = {
            "all", "image_encoder", "geometry_encoder", "feature_network",
            "static_residual_network", "feedback_gru", "gain_head",
        }
        if (not t.trainable_modules or len(set(t.trainable_modules)) != len(t.trainable_modules)
                or set(t.trainable_modules) - allowed_modules
                or ("all" in t.trainable_modules and len(t.trainable_modules) != 1)):
            raise ValueError("Invalid trainable_modules selection")
        # Reject NaN/Inf recursively rather than letting comparisons silently pass.
        def finite(v):
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError("Configuration must be finite")
            if isinstance(v, dict):
                for x in v.values():
                    finite(x)
            if isinstance(v, list):
                for x in v:
                    finite(x)
        finite(self.to_dict())
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return json_hash(self.to_dict())

    def path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path


def load_config(path):
    values = read_json(path)
    for name, cls in (("observer", ObserverConfig), ("control", ControlConfig),
                      ("environment", EnvironmentConfig), ("training", TrainingConfig),
                      ("evaluation", EvaluationConfig)):
        values[name] = cls(**values.get(name, {}))
    return Config(**values).validate()
