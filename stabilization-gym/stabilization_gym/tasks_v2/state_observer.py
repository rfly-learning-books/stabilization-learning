"""State control and observation adapters with an explicit physical boundary.

The four existing state plants retain their equations and integration.  The
observation physical backend is a replay of a declared exogenous forcing and
sensor trajectory, not a plant accepting estimator corrections as actuators.
Observer prediction/correction state lives only in ObserverTaskAdapter memory.
Privileged estimation error is computed by a separate supervision method.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
import torch

from .contracts import ActionPlan, Capabilities, TaskContext
from ..envs.classic_control.sdc_metric import SDCMetricTask
from ..observation.environment import ObservationEnv
from ..observation.models import vector


def _tensor(value, *, like=None):
    if torch.is_tensor(value):
        return value if like is None else value.to(like)
    return torch.as_tensor(value, dtype=torch.float64 if like is None else like.dtype,
                           device=None if like is None else like.device)


def _numpy(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.array(value, copy=True)


def _finite_vector(value, size, name):
    result = _tensor(value)
    if result.shape != (size,) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def _checked_matrix(matrix, prototype, size):
    matrix = _tensor(matrix, like=prototype)
    if matrix.shape[-2:] != (size, size) or not torch.isfinite(matrix).all():
        raise ValueError("Certificate coordinates or entries are invalid")
    if not torch.allclose(matrix, matrix.transpose(-1, -2), rtol=1e-7, atol=1e-10):
        raise ValueError("Certificate matrix must be symmetric")
    if torch.any(torch.linalg.eigvalsh(matrix) <= 0):
        raise ValueError("Certificate matrix must be positive definite")
    return matrix


class StatePhysicalEnv(gym.Wrapper):
    """Convert the legacy positive-cost return to standard Gymnasium reward.

    No controller, certificate, feature provider or optimizer is passed to step.
    Legacy termination/hold semantics are preserved, not upgraded silently.
    """
    def step(self, action):
        action = _numpy(action)
        observation, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info, physical_cost=float(cost), reward_convention="negative_physical_cost",
                    legacy_cost_convention=info.get("cost_convention"))
        info["cost_convention"] = "physical_cost_in_info"
        info["dt"] = self.env.dt
        return observation, -float(cost), bool(terminated), bool(truncated), info


class StateControlAdapter:
    """M and the declared input model yield feedback in physical units.

    The inherited model provider has historical identifiers.  Its role here is
    error/model/control construction, not a mandatory factorization paradigm.
    A replacement model supplies error, coefficients, feedforward and R.
    """
    capabilities = Capabilities(action_gradient=True, prediction_gradient=True,
                                physical_step_gradient=False, action_kind="physical",
                                note="Differentiable prediction model; actual Gym step crosses NumPy.")

    def __init__(self, physical, name):
        raw = physical.unwrapped if isinstance(physical, gym.Wrapper) else physical
        self.model = SDCMetricTask(raw, name)
        self.name = name
        self.action_low = self.model.action_low
        self.action_high = self.model.action_high

    def make_context(self, observation, reference=None, *, features=None, model=None,
                     memory=None, binding=None):
        if features is not None and not getattr(features, "preserves_physical_coordinates", False):
            raise ValueError("This adapter uses declared physical state coordinates")
        state = _tensor(observation)
        active = self.model if model is None else model
        if not torch.isfinite(state).all() or state.shape[-1] != active.error_dim:
            raise ValueError("Invalid public state observation")
        if reference is not None:
            target = reference.get("goal") if isinstance(reference, Mapping) else reference
            if target is not None and not torch.equal(_tensor(target, like=state), state.new_tensor(active.goal)):
                raise ValueError("Set the physical task reference before constructing context")
        error = active.error(state)
        _, B = active.coefficients(error)
        return TaskContext(error=error, metric_context=None,
            model_context={"state": state, "B": B, "R": state.new_tensor(active.R),
                           "feedforward": active.feedforward(state), "model": active},
            memory={} if memory is None else dict(memory), binding=binding)

    def construct_action(self, context, certificate):
        if context.error is None:
            raise ValueError("State feedback requires observed state error")
        result = certificate.evaluate(context.error, context.metric_context, binding=context.binding)
        M = _checked_matrix(result.M, context.error, context.error.shape[-1])
        model = context.model_context
        weighted = (M @ context.error.unsqueeze(-1))
        feedback = torch.linalg.solve(model["R"], model["B"].transpose(-1, -2) @ weighted).squeeze(-1)
        requested = model["feedforward"] - feedback
        action = torch.maximum(self.action_low.to(requested), torch.minimum(self.action_high.to(requested), requested))
        return ActionPlan(action, memory=dict(context.memory), diagnostics={
            "requested_action": requested, "M": M, "V": result.value,
            "certificate_claim": "candidate_only", "action_kind": "physical",
            "saturated": bool(torch.any(action.detach() != requested.detach()))})

    def predict(self, state, action, *, model=None):
        return (self.model if model is None else model).step_model(state, action)


def make_state_control(name, **kwargs):
    """Return a physical Gym env and its separately owned task adapter."""
    from ..envs.classic_control.classic import CoupledLinearEnv, PointMassEnv, PendulumEnv, CartPoleEnv
    factories = {"coupled_linear": CoupledLinearEnv, "point_mass": PointMassEnv,
                 "pendulum": PendulumEnv, "cartpole": CartPoleEnv}
    if name not in factories:
        raise ValueError(f"Unsupported state task {name}")
    physical = StatePhysicalEnv(factories[name](**kwargs))
    return physical, StateControlAdapter(physical, name)


class ObservationReplayEnv(gym.Env):
    """Sensor-only physical replay under predeclared exogenous inputs.

    The original generator is used only at reset to preserve exactly the existing
    seeded physics/sensor protocol.  It is discarded after copying the replay;
    there is no estimator in this environment.  step accepts the known forcing,
    never a correction, and refuses to pretend a different input was executed.

    Each endpoint measurement receives one estimator update.  After the last
    measurement, a final zero-duration step marks the replay truncated.  This
    preserves the old observer's endpoint sampling without inventing new motion.
    """
    metadata = {"render_modes": []}

    def __init__(self, task):
        source = ObservationEnv(task)
        self.task = task
        self.model = source.model
        spec = self.model.spec
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (spec.input_dim,), dtype=np.float64)
        self.observation_space = gym.spaces.Dict({
            key: space for key, space in source.observation_space.spaces.items() if key != "estimate"})
        self._done = True

    def reset(self, *, seed=None, options=None):
        source = ObservationEnv(self.task)
        source.reset(seed=seed, options=options)
        self.model = source.model
        self._times = source._times.copy()
        self._inputs = source._inputs.copy()
        self._truth = source._truth.copy()
        self._measurements = source._measurements.copy()
        self._masks = source._masks.copy()
        self._meta = deepcopy(source._meta)
        self._index, self._done = 0, False
        return self._observation(), self._info(0.)

    def _observation(self):
        k = self._index
        return {"measurement": self._measurements[k].copy(),
                "measurement_mask": self._masks[k].astype(np.int8),
                "confidence": self._masks[k].astype(np.float64),
                "known_input": self._inputs[k].copy(), "time": np.array([self._times[k]]),
                "dt": np.array([self.model.spec.dt])}

    def _info(self, elapsed):
        return {"task": self.task, "time": float(self._times[self._index]), "dt": float(elapsed),
                "backend": "fixed_forcing_sensor_replay", "reward_convention": "neutral_physical_replay",
                "estimation_cost": "supervision_only"}

    def supervision(self):
        """Privileged labels; never merge this mapping into public observation."""
        return {"truth": self._truth[self._index].copy(), "episode": deepcopy(self._meta)}

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset a completed observation replay")
        forcing = vector(_numpy(action), self.model.spec.input_dim)
        if not np.allclose(forcing, self._inputs[self._index], rtol=0., atol=1e-12):
            raise ValueError("Replay only accepts its declared known input; correction is not a physical action")
        final = self._index == len(self._times)-1
        failed = not self.model.domain_ok(self._truth[self._index])
        elapsed = 0. if failed or final else self.model.spec.dt
        if not final and not failed:
            self._index += 1
        self._done = bool(final or failed)
        return self._observation(), 0., bool(failed), bool(final and not failed), self._info(elapsed)


class ObserverTaskAdapter:
    """Pure public-data correction and predictor, with externally carried memory.

    matrix receives the estimated state, never the unknown estimation error.
    Model functions may be replaced independently of the true replay plant.
    NumPy legacy prediction/retraction do not promise end-to-end trajectory AD;
    the gain solve retains gradients to a tensor metric provider.
    """
    capabilities = Capabilities(action_gradient=True, prediction_gradient=False,
                                physical_step_gradient=False, action_kind="estimator_update")
    public_keys = frozenset({"measurement", "measurement_mask", "confidence", "known_input", "time", "dt"})

    def __init__(self, model, *, predict: Callable | None = None,
                 measure: Callable | None = None, measurement_jacobian: Callable | None = None):
        self.model = model
        self.predictor = model.predict if predict is None else predict
        self.measure = model.measure if measure is None else measure
        self.measurement_jacobian = model.measurement_jacobian if measurement_jacobian is None else measurement_jacobian

    def initial_memory(self, *, scenario="nominal", initial_estimate=None):
        estimate = self.model.initial_estimate() if initial_estimate is None else vector(initial_estimate, self.model.spec.state_dim)
        if scenario == "initial_offset" and initial_estimate is None:
            offsets = {"motor_speed": [1.5, -1.5], "cartpole_output": [.15, .1, .12, -.1],
                "cooperative_pose": [.6, -.4, .5, .4, -.2, .3, .1, -.1, .1, .1, -.1, .1]}
            estimate = self.model.retract(estimate, np.asarray(offsets[self.model.spec.name]))
        return {"estimate": vector(estimate, self.model.spec.state_dim), "time": 0.}

    def make_context(self, observation, reference=None, *, features=None, model=None,
                     memory=None, binding=None):
        if reference is not None or features is not None:
            raise ValueError("This observer uses its measurement model and public prior, not a target feature")
        if set(observation) != self.public_keys:
            raise ValueError("Only sensor observation keys are accepted; truth and externally supplied estimate are forbidden")
        if memory is None:
            raise ValueError("Observer estimate must be supplied as explicit task memory")
        active = self.model if model is None else model
        spec = active.spec
        estimate = _finite_vector(memory["estimate"], spec.state_dim, "estimate")
        time = float(np.asarray(observation["time"]).reshape(-1)[0])
        dt = float(np.asarray(observation["dt"]).reshape(-1)[0])
        if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0 or abs(time-float(memory["time"])) > 1e-8:
            raise ValueError("Observation and estimator memory have incompatible timestamps")
        measurement = _finite_vector(observation["measurement"], spec.measurement_dim, "measurement").to(estimate)
        mask = torch.as_tensor(observation["measurement_mask"], dtype=torch.bool, device=estimate.device)
        confidence = _finite_vector(observation["confidence"], spec.measurement_dim, "confidence").to(estimate)
        if mask.shape != (spec.measurement_dim,) or torch.any(confidence < 0) or torch.any(confidence > 1):
            raise ValueError("Invalid sensor mask/confidence")
        forcing = _finite_vector(observation["known_input"], spec.input_dim, "known input").to(estimate)
        functions = ((self.measure, self.measurement_jacobian, self.predictor) if model is None
                     else (active.measure, active.measurement_jacobian, active.predict))
        measured = _finite_vector(functions[0](_numpy(estimate)), spec.measurement_dim, "predicted measurement").to(estimate)
        jacobian = _tensor(functions[1](_numpy(estimate)), like=estimate)
        if jacobian.shape != (spec.measurement_dim, spec.error_dim) or not torch.isfinite(jacobian).all():
            raise ValueError("Measurement Jacobian has wrong tangent coordinates")
        scale = estimate.new_tensor(spec.error_scale)
        residual = (measurement-measured)[mask]
        h = jacobian[mask]*scale[None, :]
        weights = confidence[mask]/estimate.new_tensor(spec.measurement_scale)[mask].square()
        # error=None prevents accidental substitution of a sensor residual for the
        # unavailable state-estimation error in a V query.
        return TaskContext(error=None, metric_context=estimate,
            model_context={"residual": residual, "H": h, "weights": weights,
                "scale": scale, "known_input": forcing, "dt": dt, "time": time,
                "model": active, "predict": functions[2], "is_final": time >= spec.horizon-1e-10},
            memory={"estimate": estimate, "time": time}, binding=binding)

    def construct_action(self, context, certificate):
        if context.error is not None:
            raise ValueError("Unknown true estimation error cannot enter online observer control")
        estimate = context.metric_context
        spec = context.model_context["model"].spec
        if hasattr(certificate, "online_matrix"):
            M = certificate.online_matrix(estimate, binding=context.binding)
        elif hasattr(certificate, "matrix"):
            # Compatibility with the legacy estimate-only metric provider.
            before = getattr(certificate, "version", None)
            M = certificate.matrix(_numpy(estimate))
            if getattr(certificate, "version", None) != before:
                raise ValueError("Metric changed during online query")
        else:
            raise TypeError("Observer certificate requires online_matrix(estimate), not evaluate(true_error)")
        M = _checked_matrix(M, estimate, spec.error_dim)
        data = context.model_context
        gain = torch.linalg.solve(M, data["H"].transpose(-1, -2)*data["weights"])
        correction = data["dt"]*data["scale"]*(gain @ data["residual"])
        posterior = data["model"].retract(_numpy(estimate), _numpy(correction))
        failed = not data["model"].domain_ok(posterior)
        if failed or data["is_final"]:
            following, time = posterior, data["time"]
        else:
            following = data["predict"](posterior.copy(), _numpy(data["known_input"]), data["dt"])
            following = vector(following, spec.state_dim)
            time = data["time"]+data["dt"]
        return ActionPlan(correction, memory={"estimate": following.copy(), "time": time}, diagnostics={
            "posterior": posterior.copy(), "M": M, "gain": gain, "prior_failure": bool(failed),
            "visible_channels": len(data["residual"]), "action_kind": "estimator_update",
            "physical_action": data["known_input"].detach().clone(), "certificate_claim": "candidate_only"})

    def evaluate_supervision(self, labels, context, plan):
        """Offline cost/error only. Never used by make_context/construct_action."""
        data = context.model_context
        model = data["model"]
        truth = vector(labels["truth"], model.spec.state_dim)
        posterior = vector(plan.diagnostics["posterior"], model.spec.state_dim)
        error = model.error(truth, posterior)
        failure = bool(plan.diagnostics["prior_failure"] or not model.domain_ok(truth))
        if model.spec.name == "cooperative_pose" and np.linalg.norm(error[3:6]) >= np.pi-.05:
            failure = True
        dt = 0. if data["is_final"] else data["dt"]
        cost = float(np.sum((error/np.asarray(model.spec.error_scale))**2))*dt
        return {"error": error.copy(), "truth": truth.copy(), "estimate": posterior.copy(),
                "physical_cost": cost, "reward": -cost, "failure": failure,
                "complete": bool(data["is_final"] and not failure), "dt": dt}

