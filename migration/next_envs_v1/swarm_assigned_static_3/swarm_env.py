"""Isolated three-agent joint-QP M controller; no global-attraction claim."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from stabilization_gym.common.types import MetricSpec
from stabilization_gym.interfaces.m_learning import PortableMFacade, MetricScope

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "DESIGN_PROTOCOL.json").read_text(encoding="utf-8"))
PAIRS = ((0, 1), (0, 2), (1, 2))
DT, KAPPA, ALPHA, LIMIT, SAFE = .05, .5, 1., 1.9, .3
VMAX = .4 / np.sqrt(2.)


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x


def digest(x):
    return hashlib.sha256(json.dumps(plain(x), sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("DESIGN_PROTOCOL.json", "swarm_env.py")}


def vector(x, n, name):
    if torch.is_tensor(x):
        if x.device.type != "cpu" or x.dtype not in (torch.float32, torch.float64):
            raise ValueError(name + " requires CPU float32/float64")
        x = x.detach().numpy()
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(name + " requires a finite vector of length " + str(n))
    return a.copy()


def geometry(p, *, goal=False):
    p = vector(p, 6, "positions").reshape(3, 2)
    if goal:
        return bool(np.max(np.abs(p)) < LIMIT and
                    min(np.linalg.norm(p[i]-p[j]) for i, j in PAIRS) >= .5)
    return bool(np.max(np.abs(p)) <= LIMIT and
                min(np.linalg.norm(p[i]-p[j]) for i, j in PAIRS) >= SAFE-1e-14)


def constraints(p):
    """All constraints A u <= b; no independent post-QP clipping."""
    p = vector(p, 6, "state")
    upper = np.minimum(VMAX, (LIMIT-p)/DT)
    lower = np.maximum(-VMAX, (-LIMIT-p)/DT)
    A = [*np.eye(6), *-np.eye(6)]
    b = [*upper, *-lower]
    for i, j in PAIRS:
        r = p[2*i:2*i+2]-p[2*j:2*j+2]
        row = np.zeros(6); row[2*i:2*i+2] = -2*r; row[2*j:2*j+2] = 2*r
        A.append(row); b.append(ALPHA*(r@r-SAFE**2))
    return np.asarray(A), np.asarray(b)


def held_geometry(p, u):
    p, u = np.asarray(p).reshape(3, 2), np.asarray(u).reshape(3, 2)
    records = []
    for i, j in PAIRS:
        r, du = p[i]-p[j], u[i]-u[j]
        tau = 0. if du@du == 0 else float(np.clip(-(r@du)/(du@du), 0., DT))
        distance = float(np.linalg.norm(r+tau*du))
        records.append({"pair": [i, j], "closest_time": tau,
                        "minimum_distance": distance, "signed_safe_margin": distance-SAFE})
    return {"pairs": records,
            "minimum_distance": min(x["minimum_distance"] for x in records),
            "max_box_coordinate": float(max(np.abs(p).max(), np.abs(p+DT*u).max())),
            "signed_safe_margin": min(x["minimum_distance"] for x in records)-SAFE,
            "raw_safe_distance_violation": min(x["minimum_distance"] for x in records) < SAFE,
            "within_safe_distance_tolerance": min(x["minimum_distance"] for x in records) >= SAFE-1e-12,
            "signed_box_margin": LIMIT-float(max(np.abs(p).max(), np.abs(p+DT*u).max())),
            "max_speed_norm": float(np.linalg.norm(u, axis=1).max())}


def verify_qp(A, b, c, u, dual):
    """Residual verification does not reuse the active-set search operations."""
    A, b, c, u, dual = map(np.asarray, (A, b, c, u, dual))
    if (A.shape != (15, 6) or b.shape != (15,) or c.shape != (6,) or
            u.shape != (6,) or dual.shape != (15,) or
            not all(np.isfinite(v).all() for v in (A, b, c, u, dual))):
        return {"accepted": False, "reason": "invalid residual inputs"}
    slack = b-A@u
    out = {"primal_violation": float(max(0., -slack.min())),
           "stationarity_inf": float(np.abs(u+c+A.T@dual).max()),
           "dual_min": float(dual.min()),
           "complementarity_inf": float(np.abs(dual*slack).max()),
           "zero_comparator_epsilon": float(max(0., c@u+u@u)),
           "objective": float(.5*(u@u)+c@u)}
    out["accepted"] = bool(out["primal_violation"] <= 1e-10 and
        out["stationarity_inf"] <= 1e-9 and out["dual_min"] >= -1e-10 and
        out["complementarity_inf"] <= 1e-9)
    return out


def solve_qp(A, b, c, *, max_iterations=128):
    """Feasible primal active-set for identity Hessian, deterministic ties."""
    A, b, c = np.asarray(A), np.asarray(b), np.asarray(c)
    x = np.zeros(6); W = []
    if np.min(b) < -1e-10:
        return {"ok": False, "reason": "zero_not_feasible", "iterations": 0}
    for i in range(len(b)):
        if b[i] <= 1e-11 and np.linalg.matrix_rank(A[W+[i]], tol=1e-11) > len(W): W.append(i)
    for iteration in range(max_iterations):
        grad = x+c
        if W:
            AW = A[W]
            lam = np.linalg.solve(AW@AW.T, -AW@grad)
            direction = -grad-AW.T@lam
        else:
            lam = np.zeros(0); direction = -grad
        if np.linalg.norm(direction, ord=np.inf) <= 1e-12:
            if not W or lam.min() >= -1e-10:
                dual = np.zeros(15)
                if W: dual[W] = lam
                verified = verify_qp(A, b, c, x, dual)
                return {"ok": verified["accepted"], "reason": "KKT" if verified["accepted"] else "KKT_rejected",
                        "action": x.copy(), "dual": dual, "iterations": iteration+1,
                        "active_set": W.copy(), "verification": verified}
            W.pop(int(np.argmin(lam)))
            continue
        step, blocking = 1., None
        Ad = A@direction
        for i in range(len(b)):
            if i not in W and Ad[i] > 1e-13:
                ratio = max(0., float((b[i]-A[i]@x)/Ad[i]))
                if ratio < step-1e-13 or (abs(ratio-step) <= 1e-13 and blocking is not None and i < blocking):
                    step, blocking = ratio, i
        x = x+step*direction
        if blocking is not None and step < 1.-1e-13:
            if np.linalg.matrix_rank(A[W+[blocking]], tol=1e-11) <= len(W):
                return {"ok": False, "reason": "dependent_blocking_constraint", "iterations": iteration+1}
            W.append(blocking)
    return {"ok": False, "reason": "iteration_budget_exhausted", "iterations": max_iterations}


class SwarmAssignedStatic3Env(gym.Env):
    """Full physical control state is p; q/assignment/M are fixed context."""
    metadata = {"render_modes": []}
    controller_form = "joint_strictly_convex_QP_unit_effort_plus_M_certificate_gradient"
    control_call_is_pure = True
    control_call_side_effects = ()

    def __init__(self):
        self.state_dim = self.error_dim = self.metric_context_dim = 6
        self.dt, self.max_steps = DT, 600
        self.metric_spec = MetricSpec(6, .05, 6.)
        self.action_space = gym.spaces.Box(-VMAX, VMAX, (6,), dtype=np.float64)
        self.observation_space = gym.spaces.Dict({
            "p": gym.spaces.Box(-LIMIT, LIMIT, (6,), dtype=np.float64),
            "q": gym.spaces.Box(-LIMIT, LIMIT, (6,), dtype=np.float64),
            "assignment": gym.spaces.MultiDiscrete([3, 3, 3])})
        self.controller_capabilities = {"action_gradient": False, "model_gradient": True,
            "differentiable_candidate_scores": False, "physical_step_autograd": False,
            "episode_fixed_metric": True, "state_dependent_certificate": False,
            "cost_convention": "positive_integral", "certificate_guarantee": "candidate_only",
            "full_horizon_evaluation": True}
        self._sources = source_hashes()
        self._done = True
        self.state = np.zeros(6)
        self._pending = None

    def _assert_fixed(self):
        if not np.array_equal(self.q, self._q0) or not np.array_equal(self.assignment, self._assignment0):
            raise ValueError("Goal/assignment changed within episode")
        if not np.array_equal(self.state, self._expected_state):
            raise ValueError("Physical state changed outside step/reset")

    @property
    def goal(self): return self.q.reshape(3, 2)[self.assignment].reshape(6).copy()

    def _observation(self):
        return {"p": self.state.copy(), "q": self.q.copy(), "assignment": self.assignment.copy()}

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)
        if set(options)-{"p", "q", "assignment"}: raise ValueError("Only p/q/assignment reset options")
        default = PROTOCOL["engineering_fixtures"][0]
        p, q = vector(options.get("p", default["p"]), 6, "p"), vector(options.get("q", default["q"]), 6, "q")
        a = np.asarray(options.get("assignment", [0, 1, 2]))
        if a.shape != (3,) or a.dtype.kind not in "iu" or sorted(a.tolist()) != [0, 1, 2]:
            raise ValueError("Assignment must be an integer permutation")
        if not geometry(p) or not geometry(q, goal=True): raise ValueError("Unsafe initial state or invalid target geometry")
        super().reset(seed=seed)
        self.state = p.copy(); self._expected_state = p.copy(); self._initial = p.copy()
        self.q = q.copy(); self._q0 = q.copy(); self.assignment = a.astype(np.int64); self._assignment0 = self.assignment.copy()
        self._steps, self._queries = 0, 0
        self._pending = self._episode_M = None
        self._done, self._failed = False, False
        self._trace, self._plans = [], []
        self._origin = {"p": p.copy(), "q": q.copy(), "assignment": self.assignment.copy(), "seed": seed}
        self._hold_intervals = 0; self._first_hold = None
        self._last_info = self._info()
        return self._observation(), deepcopy(self._last_info)

    def _info(self):
        pos = np.linalg.norm((self.state-self.goal).reshape(3, 2), axis=1)
        return {"steps": self._steps, "time": self._steps*DT,
                "in_goal": bool(np.all(pos <= .03)), "max_assigned_error": float(pos.max()),
                "success": bool(self._steps == 600 and self._hold_intervals >= 40 and not self._failed),
                "hold_seconds": self._hold_intervals*DT,
                "first_hold_confirmation": self._first_hold,
                "sustained_arrival_time": None if self._hold_intervals < 40 else (self._steps-self._hold_intervals)*DT,
                "failure": self._failed, "cost_convention": "positive_integral"}

    def build_error(self, observation, goal=None, representation=None, history=None):
        self._assert_fixed()
        if representation is not None or history is not None: raise ValueError("Physical-coordinate state only")
        if goal is not None and not np.array_equal(vector(goal, 6, "goal"), self.goal): raise ValueError("Goal override forbidden")
        if isinstance(observation, dict):
            if set(observation) != {"p", "q", "assignment"} or not np.array_equal(observation["q"], self.q) or not np.array_equal(observation["assignment"], self.assignment):
                raise ValueError("Observation context differs")
            observation = observation["p"]
        p = observation if torch.is_tensor(observation) else torch.as_tensor(observation, dtype=torch.float64)
        vector(p, 6, "observation")
        if not np.allclose(p.detach().numpy(), self.state, rtol=0., atol=2e-7 if p.dtype == torch.float32 else 0.):
            raise ValueError("Stale physical observation")
        e = p-p.new_tensor(self.goal)
        return e, {"step": self._steps, "physical_state": self.state.copy(),
                   "assigned_goal": self.goal, "assignment": self.assignment.copy()}

    def metric_context(self, observation=None, goal=None):
        self._assert_fixed()
        if goal is not None and not np.array_equal(vector(goal, 6, "goal"), self.goal): raise ValueError("Goal override forbidden")
        return torch.tensor(self.goal, dtype=torch.float64)

    def _metric(self, M):
        if not torch.is_tensor(M) or M.device.type != "cpu" or M.dtype not in (torch.float32, torch.float64):
            raise ValueError("M requires CPU float32/float64 tensor")
        m = M.detach().double().numpy().copy()
        # Admission survives exact float32-to-float64 snapshot promotion.
        tol = 2e-6
        if m.shape != (6, 6) or not np.isfinite(m).all() or not np.array_equal(m, m.T): raise ValueError("M must be finite and exactly symmetric; no silent symmetrization")
        eigen = np.linalg.eigvalsh(m)
        if abs(np.trace(m)-6) > 6*tol or eigen[0] < .05-tol or eigen[-1] > 5.75+tol: raise ValueError("M violates trace/eigen contract")
        if self._episode_M is not None and not np.array_equal(m, self._episode_M): raise ValueError("Episode M changed")
        return m

    def plan(self, M, e, *, context):
        self._assert_fixed()
        if self._done: raise RuntimeError("reset required")
        m = self._metric(M); error = vector(e, 6, "error")
        if context["step"] != self._steps or not np.array_equal(context["physical_state"], self.state) or not np.array_equal(context["assigned_goal"], self.goal) or not np.array_equal(context["assignment"], self.assignment):
            raise ValueError("Stale control context")
        tol = 5e-7 if torch.is_tensor(e) and e.dtype == torch.float32 else 2e-12
        if not np.allclose(error, self.state-self.goal, atol=tol, rtol=0): raise ValueError("Error not physical assigned error")
        # QP always uses exact stored float64 coordinates. float32 errors are checked inputs.
        error = self.state-self.goal
        A, b = constraints(self.state); c = KAPPA*m@error
        try:
            result = solve_qp(A, b, c)
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            result = {"ok": False, "reason": "numeric_solver_exception", "exception_type": type(exc).__name__, "exception_text": str(exc), "iterations": None}
        u = np.asarray(result["action"]) if result["ok"] else np.zeros(6)
        held = held_geometry(self.state, u)
        safe = held["minimum_distance"] >= SAFE-1e-12 and held["max_box_coordinate"] <= LIMIT+1e-12 and held["max_speed_norm"] <= .4+1e-12
        if not safe:
            result = {**result, "ok": False, "reason": "held_geometry_rejected"}
            u = np.zeros(6); held = held_geometry(self.state, u)
        plan = {"sources": self._sources, "step": self._steps, "state": self.state.copy(),
                "goal": self.goal, "assignment": self.assignment.copy(), "M": m,
                "A": A, "b": b, "c": c, "action": u.copy(), "solver": result,
                "metric_eigenvalues": np.linalg.eigvalsh(m), "metric_trace": float(np.trace(m)),
                "solver_failure": not result["ok"], "held_geometry": held}
        plan["plan_sha256"] = digest(plan)
        return deepcopy(plan)

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        if controller_state is not None or certificate is not None: raise ValueError("Fixed M only")
        return torch.tensor(self.plan(M, e, context=context)["action"], dtype=torch.float64)

    def submit_plan(self, plan):
        self._assert_fixed()
        if self._done: raise RuntimeError("Finished episode cannot accept a plan")
        if self._pending is not None: raise RuntimeError("Pending action already exists")
        p = deepcopy(plan); signature = p.pop("plan_sha256")
        if digest(p) != signature or p["sources"] != self._sources or p["step"] != self._steps or not np.array_equal(p["state"], self.state) or not np.array_equal(p["goal"], self.goal) or not np.array_equal(p["assignment"], self.assignment):
            raise ValueError("Plan provenance mismatch")
        m = self._metric(torch.as_tensor(p["M"], dtype=torch.float64))
        A, b = constraints(self.state); c = KAPPA*m@(self.state-self.goal)
        if not np.array_equal(A, p["A"]) or not np.array_equal(b, p["b"]) or not np.array_equal(c, p["c"]): raise ValueError("Submitted QP coefficients differ")
        if p["solver_failure"]:
            if np.any(p["action"]) or p["solver"]["ok"]: raise ValueError("Failure plan must record zero and failed solver")
        elif not p["solver"]["ok"] or not verify_qp(A,b,c,np.asarray(p["action"]),np.asarray(p["solver"]["dual"]))["accepted"]:
            raise ValueError("Submitted QP residuals do not verify")
        if digest(held_geometry(self.state, p["action"])) != digest(p["held_geometry"]): raise ValueError("Submitted held geometry differs")
        self._pending = deepcopy(plan); self._queries += 1
        self._plans.append({"plan_sha256": signature, "step": self._steps, "executed": False})

    def torch_step(self, state, action):
        if not torch.is_tensor(state) or not torch.is_tensor(action) or state.shape[-1:] != (6,) or action.shape != state.shape or state.dtype != action.dtype or state.device != action.device:
            raise ValueError("Tensor model needs same-shape/dtype/device [...,6] inputs")
        if not state.is_floating_point() or not torch.isfinite(state).all() or not torch.isfinite(action).all(): raise ValueError("Finite floating model inputs required")
        return state+DT*action

    def step(self, action):
        self._assert_fixed()
        if self._done or self._pending is None: raise RuntimeError("Reset and audited facade control required")
        u = vector(action, 6, "physical action"); plan = self._pending
        if not np.array_equal(u, plan["action"]): raise ValueError("Action differs from audited QP output; no clipping")
        m = self._metric(torch.as_tensor(plan["M"], dtype=torch.float64))
        A, b = constraints(self.state); e = self.state-self.goal; c = KAPPA*m@e
        if not np.array_equal(A, plan["A"]) or not np.array_equal(b, plan["b"]) or not np.array_equal(c, plan["c"]): raise ValueError("Plan coefficients changed")
        fail = bool(plan["solver_failure"])
        if not fail and not verify_qp(A, b, c, u, plan["solver"]["dual"])["accepted"]: raise ValueError("QP no longer verifies")
        if fail and np.any(u): raise ValueError("Solver failure fallback must be zero")
        held = held_geometry(self.state, u)
        violation = held["minimum_distance"] < SAFE-1e-12 or held["max_box_coordinate"] > LIMIT+1e-12 or held["max_speed_norm"] > .4+1e-12
        before = self.state.copy(); after = before+DT*u
        v0 = float(e@m@e); en = after-self.goal; v1 = float(en@m@en)
        self._episode_M = m.copy(); self.state = after; self._expected_state = after.copy(); self._steps += 1
        self._failed = bool(self._failed or fail or violation)
        both_positions = max(np.linalg.norm(e.reshape(3, 2), axis=1).max(), np.linalg.norm(en.reshape(3, 2), axis=1).max()) <= .03
        interval_good = bool(both_positions and np.linalg.norm(u.reshape(3, 2), axis=1).max() <= .02 and not self._failed)
        self._hold_intervals = self._hold_intervals+1 if interval_good else 0
        if self._hold_intervals >= 40 and self._first_hold is None: self._first_hold = self._steps*DT
        cost = DT*float(e@e)
        tail = max(float(en@en), 25.)*(600-self._steps)*DT if self._failed else 0.
        epsilon = max(0., float(c@u+u@u))
        row = {"step": self._steps, "state_before": before, "state_after": after.copy(), "action": u.copy(),
               "plan": deepcopy(plan), "held_geometry": held, "V_before": v0, "V_after": v1,
               "delta_V": v1-v0, "same_M_bound": (-2*DT/KAPPA+DT**2*np.linalg.eigvalsh(m)[-1])*float(u@u)+2*DT*epsilon/KAPPA,
               "zero_comparator_epsilon": epsilon, "cost": cost, "failure_tail": tail,
               "failure": self._failed, "physical_contact": held["minimum_distance"] < .2,
               "safe_distance_violation": held["minimum_distance"] < .3-1e-12,
               "raw_safe_distance_violation": held["minimum_distance"] < .3,
               "minimum_safe_margin_m": held["minimum_distance"]-.3,
               "safety_tolerance_accepted": not violation,
               "hold_intervals": self._hold_intervals}
        self._trace.append(row); self._pending = None; self._plans[-1]["executed"] = True
        self._done = bool(self._failed or self._steps >= 600)
        info = self._info(); info.update(applied_action=u.copy(), cost=cost, failure_tail=tail, solver_failure=fail,
                                        physical_contact=row["physical_contact"], safe_distance_violation=row["safe_distance_violation"],
                                        raw_safe_distance_violation=row["raw_safe_distance_violation"], minimum_safe_margin_m=row["minimum_safe_margin_m"],
                                        safety_tolerance_accepted=row["safety_tolerance_accepted"])
        self._last_info = deepcopy(info)
        return self._observation(), cost+tail, self._failed, self._steps >= 600 and not self._failed, info

    def state_dict(self):
        self._assert_fixed()
        s = {"sources": self._sources, "origin": self._origin, "state": self.state, "q": self.q,
             "assignment": self.assignment, "steps": self._steps, "queries": self._queries,
             "episode_M": self._episode_M, "pending": self._pending, "trace": self._trace,
             "plans": self._plans, "done": self._done, "failed": self._failed,
             "hold_intervals": self._hold_intervals, "first_hold": self._first_hold,
             "last_info": self._last_info, "rng": self.np_random.bit_generator.state}
        s = plain(s); s["snapshot_sha256"] = digest(s)
        return deepcopy(s)

    def load_state_dict(self, state):
        s = deepcopy(state); checksum = s.pop("snapshot_sha256", None)
        if checksum != digest(s) or s["sources"] != source_hashes(): raise ValueError("Snapshot source/protocol/checksum mismatch")
        if type(s["steps"]) is not int or not 0 <= s["steps"] <= 600 or type(s["queries"]) is not int or not s["steps"] <= s["queries"] <= s["steps"]+1:
            raise ValueError("Snapshot integer counter bounds invalid")
        origin = s["origin"]; temp = type(self)()
        temp.reset(seed=origin["seed"], options={k: origin[k] for k in ("p", "q", "assignment")})
        if not np.array_equal(s["q"], temp.q) or not np.array_equal(s["assignment"], temp.assignment): raise ValueError("Snapshot target changed")
        # Validate saved records without calling physical step or solver.
        previous = temp.state.copy(); hold = 0; first_hold = None; failed = False
        metric = None if s["episode_M"] is None else np.asarray(s["episode_M"], dtype=np.float64)
        if metric is not None: temp._metric(torch.as_tensor(metric))
        for k, row in enumerate(s["trace"], 1):
            before, after, u = (np.asarray(row[n]) for n in ("state_before", "state_after", "action"))
            if row["step"] != k or not np.array_equal(before, previous) or not np.array_equal(after, before+DT*u) or metric is None or failed: raise ValueError("Snapshot physical chain invalid")
            p = deepcopy(row["plan"]); token = p.pop("plan_sha256")
            if digest(p) != token or p["sources"] != self._sources or p["step"] != k-1 or not np.array_equal(p["M"], metric) or not np.array_equal(p["action"], u) or not np.array_equal(p["state"], before) or not np.array_equal(p["goal"], temp.goal) or not np.array_equal(p["assignment"], temp.assignment): raise ValueError("Snapshot plan invalid")
            A, b = constraints(before); e, en = before-temp.goal, after-temp.goal
            if not np.array_equal(A, p["A"]) or not np.array_equal(b, p["b"]) or not np.array_equal(KAPPA*metric@e, p["c"]): raise ValueError("Saved QP coefficients differ")
            if not p["solver_failure"] and not verify_qp(A, b, KAPPA*metric@e, u, p["solver"]["dual"])["accepted"]: raise ValueError("Saved KKT failed")
            if p["solver_failure"] and np.any(u): raise ValueError("Nonzero failure fallback")
            geom = held_geometry(before, u)
            if digest(geom) != digest(row["held_geometry"]) or digest(geom) != digest(p["held_geometry"]): raise ValueError("Saved held geometry differs")
            failed = bool(p["solver_failure"] or geom["minimum_distance"] < SAFE-1e-12 or geom["max_box_coordinate"] > LIMIT+1e-12 or geom["max_speed_norm"] > .4+1e-12)
            good = max(np.linalg.norm(e.reshape(3, 2), axis=1).max(), np.linalg.norm(en.reshape(3, 2), axis=1).max()) <= .03 and np.linalg.norm(u.reshape(3, 2), axis=1).max() <= .02 and not failed
            hold = hold+1 if good else 0
            if hold >= 40 and first_hold is None: first_hold = k*DT
            expected = {"cost": DT*float(e@e), "V_before": float(e@metric@e), "V_after": float(en@metric@en),
                        "delta_V": float(en@metric@en)-float(e@metric@e),
                        "zero_comparator_epsilon": max(0., float((KAPPA*metric@e)@u+u@u)),
                        "minimum_safe_margin_m": geom["minimum_distance"]-.3,
                        "failure_tail": max(float(en@en), 25.)*(600-k)*DT if failed else 0.}
            expected["same_M_bound"] = (-2*DT/KAPPA+DT**2*np.linalg.eigvalsh(metric)[-1])*float(u@u)+2*DT*expected["zero_comparator_epsilon"]/KAPPA
            if any(not np.isclose(row[key], val, rtol=1e-12, atol=1e-12) for key, val in expected.items()) or row["failure"] != failed or row["hold_intervals"] != hold: raise ValueError("Saved metrics invalid")
            if row["raw_safe_distance_violation"] != (geom["minimum_distance"] < .3) or row["safe_distance_violation"] != (geom["minimum_distance"] < .3-1e-12) or row["physical_contact"] != (geom["minimum_distance"] < .2) or row["safety_tolerance_accepted"] != (geom["minimum_distance"] >= SAFE-1e-12 and geom["max_box_coordinate"] <= LIMIT+1e-12 and geom["max_speed_norm"] <= .4+1e-12): raise ValueError("Saved safety flags invalid")
            previous = after
        if s["steps"] != len(s["trace"]) or not np.array_equal(previous, s["state"]) or s["hold_intervals"] != hold or s["first_hold"] != first_hold or s["failed"] != failed or s["done"] != (failed or s["steps"] >= 600): raise ValueError("Snapshot final state invalid")
        if s["queries"] != len(s["plans"]) or sum(p["executed"] for p in s["plans"]) != s["steps"] or len(s["plans"])-s["steps"] != int(s["pending"] is not None): raise ValueError("Snapshot budget invalid")
        for k, row in enumerate(s["trace"]):
            if s["plans"][k] != {"plan_sha256": row["plan"]["plan_sha256"], "step": k, "executed": True}: raise ValueError("Executed plan log does not match trace")
        if s["pending"] is not None and s["plans"][-1] != {"plan_sha256": s["pending"]["plan_sha256"], "step": s["steps"], "executed": False}: raise ValueError("Pending log does not match plan")
        temp.state = previous.copy(); temp._expected_state = previous.copy(); temp._steps = s["steps"]
        temp._episode_M = metric; temp._trace = deepcopy(s["trace"]); temp._plans = deepcopy(s["plans"])
        temp._queries = s["queries"]; temp._done = s["done"]; temp._failed = failed
        temp._hold_intervals = hold; temp._first_hold = first_hold
        expected_info = temp._info()
        if s["trace"]:
            last = s["trace"][-1]
            expected_info.update(applied_action=np.asarray(last["action"]), cost=last["cost"], failure_tail=last["failure_tail"],
                solver_failure=last["plan"]["solver_failure"], physical_contact=last["physical_contact"],
                safe_distance_violation=last["safe_distance_violation"], raw_safe_distance_violation=last["raw_safe_distance_violation"],
                minimum_safe_margin_m=last["minimum_safe_margin_m"], safety_tolerance_accepted=last["safety_tolerance_accepted"])
        if digest(expected_info) != digest(s["last_info"]): raise ValueError("Snapshot final info mismatch")
        temp._last_info = deepcopy(expected_info)
        temp.np_random.bit_generator.state = deepcopy(s["rng"])
        if s["pending"] is not None:
            # submit validates full pending provenance and immutable M; restore accounting afterwards.
            temp.submit_plan(s["pending"]); temp._queries = s["queries"]; temp._plans = deepcopy(s["plans"])
        self.__dict__.update(temp.__dict__)


class SwarmMFacade(PortableMFacade):
    """Callable engineering adapter; formal algorithm support deliberately empty."""
    def __init__(self, env):
        super().__init__(env, name="isolated_swarm_assigned_static_3_v1")
        self.description = replace(self.description, support_level="engineering_only_control_and_step",
            metric_scope=MetricScope.GOAL_CONDITIONED, conditioning=("assigned_static_goal",),
            control_call_is_pure=False, control_call_side_effects=("pending_audited_plan", "QP_query_count"),
            supported_algorithms=(), missing_capabilities=("formal PPO integration and optimizer acceptance", "QP action gradients", "moving goals and reassignment", "global attraction"),
            gradients=replace(self.description.gradients, note="Analytic torch model differentiable; hard active-set QP action and physical step not advertised as autograd."))
        self.source_signature_sha256 = digest({"description": self.description.to_dict(), "sources": env._sources})

    def calculate_action_from_M(self, M, e, *, context, controller_state=None, certificate=None):
        if controller_state is not None or certificate is not None: raise ValueError("Fixed episode M only")
        plan = self.env.plan(M, e, context=context)
        self.env.submit_plan(plan)
        return torch.tensor(plan["action"], dtype=torch.float64)
