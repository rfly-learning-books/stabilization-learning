"""Four-stage A/B contact optimization; frozen observer, genuine feedback only.

This is a development controller, not a hybrid stability certificate.
All infeasible stages terminate explicitly. Hypothetical contact starts are
used only to rank A/B, never written to replay as measured transitions.
"""
from dataclasses import dataclass, replace
from heapq import heappop, heappush
import numpy as np
import torch
from shapely.geometry import Point, LineString

from .state import DT, corrected_state, model_context, wrap


@dataclass
class FourStageConfig:
    horizon: int = 40
    translate_horizon: int = 0
    fixed_rotation_deadline: bool = False
    candidates: int = 128
    contact_samples: int = 32
    action_blocks: int = 5
    search_iterations: int = 3
    contact_gap_tolerance: float = .01
    max_speed: float = .15
    pusher_radius: float = .018
    clearance: float = .003
    grid_spacing: float = .005
    workspace: float = .232
    navigation_gain: float = 3.
    navigation_tolerance: float = .004
    pose_drift_tolerance: float = .002
    pose_drift_angle: float = .02
    position_scale: float = .10
    angle_tolerance: float = .017453292519943295
    position_tolerance: float = .001
    speed_tolerance: float = .005
    omega_tolerance: float = .02
    input_tolerance: float = .005
    hold_steps: int = 100
    switch_hold: int = 3
    angular_decay: float = .01
    position_decay: float = .01
    angular_margin: float = 0.
    correction_mode: str = "first"
    max_stage_steps: int = 400

    def validate(self):
        if min(self.horizon, self.contact_samples, self.action_blocks,
               self.search_iterations, self.hold_steps, self.switch_hold,
               self.max_stage_steps) < 1 or self.candidates < 8:
            raise ValueError("Positive four-stage budgets required")
        if self.translate_horizon < 0:
            raise ValueError("translate_horizon must be nonnegative")
        if not 0 < self.max_speed <= .30:
            raise ValueError("Physical command limit must be within .30 m/s per axis")
        if not 0 < self.angular_decay < 1 or not 0 < self.position_decay < 1:
            raise ValueError("Decay factors must be in (0,1)")
        if self.correction_mode not in {"first", "full", "none"}:
            raise ValueError("Unknown observer correction mode")
        for key in ("grid_spacing", "workspace", "navigation_gain",
                    "position_scale", "angle_tolerance", "position_tolerance",
                    "speed_tolerance", "omega_tolerance", "input_tolerance",
                    "pusher_radius", "navigation_tolerance"):
            if getattr(self, key) <= 0:
                raise ValueError(key)
        if self.clearance < 0 or self.angular_margin < 0:
            raise ValueError("Margins must be nonnegative")
        return self


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def angle_error(state, goal, period):
    order = 2*np.pi/period
    return wrap(order*(np.asarray(state)[..., 3]-goal[2]))/order


def contact_sites(geometry, count, radius, clearance):
    """Return surface q, outward n, and pusher-center positions in body frame."""
    parts = [geometry.polygon] if geometry.polygon.geom_type == "Polygon" else geometry.polygon.geoms
    result = []
    length = sum(p.exterior.length for p in parts)
    for part in parts:
        coords = np.asarray(part.exterior.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            delta = b-a
            nseg = max(1, int(np.ceil(count*np.linalg.norm(delta)/length)))
            n = np.array([delta[1], -delta[0]])/np.linalg.norm(delta)
            for f in (np.arange(nseg)+.5)/nseg:
                q = a+f*delta
                if geometry.polygon.contains(Point(q+.0001*n)):
                    n = -n
                touch, pre = q+radius*n, q+(radius+clearance)*n
                if geometry.polygon.distance(Point(touch)) < radius-1e-7:
                    continue
                if geometry.polygon.distance(Point(pre)) < radius+clearance-1e-7:
                    continue
                result.append(dict(q=q, normal=n, touch=touch, pre=pre))
    return result


def free_grid_path(start, target, obstacle, settings):
    """A* pusher-center path. Every edge, including endpoints, is checked."""
    start, target = np.asarray(start), np.asarray(target)
    limit, spacing = settings.workspace, settings.grid_spacing
    def free(a, b):
        return (np.max(np.abs(a)) <= limit and np.max(np.abs(b)) <= limit
                and not obstacle.intersects(LineString([a, b])))
    if not free(start, start) or not free(target, target):
        return None
    if free(start, target):
        return np.array([start, target])
    axis = np.linspace(-limit, limit, int(np.ceil(2*limit/spacing))+1)
    def xy(index):
        return np.array([axis[index[0]], axis[index[1]]])
    def nearest(point):
        return tuple(np.argmin(abs(axis-x)) for x in point)
    # Include exact endpoints without trusting snapped grid cells.
    queue, distance, parent = [], {}, {}
    anchor = nearest(start)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            key = (anchor[0]+dx, anchor[1]+dy)
            if min(key)<0 or max(key)>=len(axis) or not free(start, xy(key)):
                continue
            cost = np.linalg.norm(xy(key)-start)
            distance[key], parent[key] = cost, None
            heappush(queue, (cost+np.linalg.norm(xy(key)-target), cost, key))
    last = None
    while queue:
        _, cost, key = heappop(queue)
        if cost > distance[key]+1e-12:
            continue
        point = xy(key)
        if np.linalg.norm(point-target) < 3*spacing and free(point, target):
            last = key
            break
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nxt = (key[0]+dx, key[1]+dy)
            if min(nxt)<0 or max(nxt)>=len(axis) or not free(point, xy(nxt)):
                continue
            nc = cost+np.linalg.norm(xy(nxt)-point)
            if nc < distance.get(nxt, float("inf")):
                distance[nxt], parent[nxt] = nc, key
                heappush(queue, (nc+np.linalg.norm(xy(nxt)-target), nc, nxt))
    if last is None:
        return None
    path = [target]
    while last is not None:
        path.append(xy(last))
        last = parent[last]
    path.append(start)
    path = path[::-1]
    # Collision-checked shortcutting.
    compact, i = [path[0]], 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1 and not free(path[i], path[j]):
            j -= 1
        compact.append(path[j]); i = j
    return np.asarray(compact)


def horizon_metrics(states, initial, actions, goal, period, stage, cfg):
    """Hard per-step angular envelope, terminal angle for A, position for B."""
    e = angle_error(states, goal, period)
    e0 = float(angle_error(initial, goal, period))
    ep = np.sum(((states[..., :2]-goal[:2])/cfg.position_scale)**2, axis=-1)
    ep0 = np.sum(((initial[:2]-goal[:2])/cfg.position_scale)**2)
    steps = np.arange(1, states.shape[1]+1)
    valid = (np.isfinite(states).all(axis=(1,2))
             & (abs(states[..., :2]) <= .35).all(axis=(1,2))
             & (abs(states[..., 8:10]) <= cfg.workspace+1e-6).all(axis=(1,2))
             & (states[..., 2]>=0).all(axis=1) & (states[..., 2]<=.1).all(axis=1))
    if stage == "rotate":
        envelope = (1-cfg.angular_decay)**steps*e0**2 + cfg.angle_tolerance**2
        violation = np.maximum(
            np.max(e**2 + cfg.angular_margin-envelope, axis=1),
            e[:, -1]**2 + cfg.angular_margin-cfg.angle_tolerance**2)
    else:
        envelope = (1-cfg.position_decay)**steps*ep0 + (cfg.position_tolerance/cfg.position_scale)**2
        violation = np.maximum(
            np.max(e**2+cfg.angular_margin-cfg.angle_tolerance**2, axis=1),
            np.max(ep-envelope, axis=1))
    # Terminal rest is necessary before moving from A to B or holding.
    rest = (np.sum(states[:, -1, 4:7]**2, axis=-1)-cfg.speed_tolerance**2)
    spin = states[:, -1, 7]**2-cfg.omega_tolerance**2
    violation = np.maximum(violation, np.maximum(rest, spin))
    score = ep.mean(axis=1) + ep[:, -1] + .01*np.mean(actions**2, axis=(1,2))
    return score, violation, valid & (violation <= 0)


class FourStageController:
    def __init__(self, geometry, model, predictor, goal, cfg, rng):
        self.geometry, self.model, self.predictor = geometry, model, predictor
        self.goal, self.cfg, self.rng = np.asarray(goal), cfg.validate(), rng
        self.device = next(model.parameters()).device
        with torch.no_grad():
            self.embedding = model.encode_geometry(
                torch.as_tensor(geometry.grid[None], device=self.device),
                torch.as_tensor(geometry.dimensions[None], device=self.device))
        self.memory = model.initial()
        self.phase, self.site, self.path = "select_A", None, None
        self.phase_steps, self.switch_count, self.hold_count = 0, 0, 0
        self.nav_pose = None
        self.failure = None
        self.pending = None
        self.last_plan = None
        self.task_energy_integral = 0.

    @torch.no_grad()
    def predict(self, states, actions, correction=True):
        nominal, _ = self.predictor.predict(states, actions)
        contexts = np.stack([model_context(s, a[None], n[None], self.geometry.local(s))[0]
                             for s, a, n in zip(states, actions, nominal)])
        residual, phi = self.model.predict(
            torch.as_tensor(contexts, device=self.device),
            self.embedding.expand(len(states), -1), self.memory.expand(len(states)))
        predicted = (corrected_state(nominal, residual.cpu().numpy())
                     if correction and self.cfg.correction_mode != "none" else nominal)
        return predicted.astype(np.float32), nominal, contexts, residual, phi

    def rollout(self, initial, sequences):
        states = np.asarray(initial, np.float32).copy()
        result = []
        for k in range(sequences.shape[1]):
            correction = self.cfg.correction_mode == "full" or (
                k == 0 and self.cfg.correction_mode == "first")
            states = (self.predict(states, sequences[:, k], True)[0] if correction
                      else self.predictor.predict(states, sequences[:, k])[0])
            result.append(states)
        return np.stack(result, axis=1)

    def optimize(self, initial, stage, site=None):
        c = replace(self.cfg, horizon=self.cfg.translate_horizon) if stage == "translate" and self.cfg.translate_horizon else self.cfg
        if stage == "rotate" and site is not None and c.fixed_rotation_deadline:
            remaining = c.horizon-self.phase_steps
            if remaining <= 0:
                return None
            c = replace(c, horizon=remaining)
        sites = contact_sites(self.geometry, c.contact_samples, c.pusher_radius, c.clearance)
        count, horizon = c.candidates, c.horizon
        ids = np.arange(count) % len(sites) if sites else np.zeros(count, int)
        if not sites:
            return None
        starts = np.repeat(np.asarray(initial, np.float32)[None], count, axis=0)
        if site is None:
            for i, sid in enumerate(ids):
                starts[i, 8:10] = initial[:2]+rotation(initial[3]) @ sites[sid]["touch"]
                starts[i, 10:12] = 0.
        directions = np.stack([-rotation(initial[3]) @
                              (sites[sid] if site is None else site)["normal"] for sid in ids])
        speeds = np.resize(np.array([.01, .03, .06, .10, .15]), count)
        sequences = np.repeat((directions*speeds[:, None])[:, None], horizon, axis=1)
        sequences[:, -max(1, horizon//5):] = 0.
        best = None
        iteration_audits = []
        for iteration in range(c.search_iterations):
            if iteration:
                # Preserve each site's own starting geometry during optimization.
                noise = self.rng.normal(0, c.max_speed*.4/(iteration+1),
                                        (count, c.action_blocks, 2))
                bins = np.minimum(np.arange(horizon)*c.action_blocks//horizon, c.action_blocks-1)
                sequences = np.clip(sequences+noise[:, bins], -c.max_speed, c.max_speed)
                sequences[:, -max(1, horizon//5):] = 0.
            trajectory = self.rollout(starts, sequences.astype(np.float32))
            score, violation, feasible = horizon_metrics(
                trajectory, initial, sequences, self.goal, self.geometry.period, stage, c)
            gaps=self.geometry.contact_gaps(trajectory)
            moving=np.linalg.norm(sequences,axis=-1)>.001
            contact_fraction=np.sum((gaps<=c.contact_gap_tolerance)&moving,axis=1)/np.maximum(1,moving.sum(axis=1))
            feasible &= contact_fraction>=.5
            order = np.argsort(np.where(feasible, score, np.inf))
            for index in order:
                if not feasible[index]:
                    break
                candidate_site = sites[ids[index]] if site is None else site
                path = None
                if site is None:
                    path = self.route(initial, candidate_site)
                    if path is None:
                        continue
                if best is None or score[index] < best["score"]:
                    best = dict(site=candidate_site, path=path,
                                sequence=sequences[index].copy(),
                                prediction=trajectory[index].copy(),
                                score=float(score[index]), violation=float(violation[index]),
                                hypothetical_start=site is None)
                break
            iteration_audits.append(dict(iteration=iteration, feasible=int(feasible.sum()),
                minimum_violation=float(np.nanmin(violation))))
            self.last_plan = dict(stage=stage, horizon=horizon, iterations=iteration_audits.copy(),
                                  feasible_total=sum(x["feasible"] for x in iteration_audits),
                                  feasible=int(feasible.sum()),
                                  minimum_violation=float(np.nanmin(violation)),
                                  hypothetical_start=site is None)
        return best

    def obstacle(self, state, radius):
        from shapely.affinity import rotate, translate
        poly = translate(rotate(self.geometry.polygon, state[3], origin=(0,0),
                                use_radians=True), *state[:2])
        return poly.buffer(radius)

    def route(self, state, site):
        target = state[:2]+rotation(state[3])@site["pre"]
        obstacle = self.obstacle(state, self.cfg.pusher_radius+.0001)
        start=state[8:10]
        if obstacle.intersects(Point(start)):
            # Explicit separating segment before free-space navigation. Reject
            # substantial geometric penetration; it needs model calibration.
            if self.geometry.contact_gap(state)<-.001:
                return None
            from shapely.ops import nearest_points
            boundary=np.asarray(nearest_points(obstacle.boundary,Point(start))[0].coords[0])
            direction=boundary-start
            if np.linalg.norm(direction)<1e-9:
                return None
            exit_point=boundary+.002*direction/np.linalg.norm(direction)
            route=free_grid_path(exit_point,target,obstacle,self.cfg)
            return np.vstack([start,route]) if route is not None else None
        return free_grid_path(start, target, obstacle, self.cfg)

    def separating_or_free(self, state, endpoint):
        obstacle=self.obstacle(state,self.cfg.pusher_radius)
        start=state[8:10]
        if not obstacle.intersects(LineString([start,endpoint])):
            return True
        if not obstacle.intersects(Point(start)) or self.geometry.contact_gap(state)<-.001:
            return False
        # Signed distance to the buffered obstacle must increase monotonically.
        distances=[]
        for f in np.linspace(0.,1.,9):
            point=Point(start+f*(endpoint-start))
            distances.append((-1 if obstacle.contains(point) else 1)
                             *point.distance(obstacle.boundary))
        return distances[-1]>distances[0]+1e-8 and np.min(np.diff(distances))>=-1e-9

    def fail(self, why):
        self.failure = why
        return None, dict(phase=self.phase, failure=why, planning=self.last_plan)

    def command(self, state):
        if self.pending is not None:
            raise RuntimeError("Consume real feedback before issuing another command")
        c = self.cfg
        if self.failure:
            return self.fail(self.failure)
        if (self.phase == "rotate" and c.fixed_rotation_deadline
                and self.phase_steps >= c.horizon):
            return self.fail("fixed_rotation_deadline_exhausted")
        if self.phase_steps >= c.max_stage_steps:
            return self.fail("stage_timeout")
        err = abs(float(angle_error(state, self.goal, self.geometry.period)))
        position = np.linalg.norm(state[:2]-self.goal[:2])
        rest = np.linalg.norm(state[4:7])<=c.speed_tolerance and abs(state[7])<=c.omega_tolerance
        if self.phase in ("select_A", "select_B"):
            stage = "rotate" if self.phase == "select_A" else "translate"
            # A can be skipped only when real orientation and motion meet its guard.
            if stage == "rotate" and err <= c.angle_tolerance and rest:
                self.phase = "select_B"; stage = "translate"
            plan = self.optimize(state, stage)
            if plan is None:
                return self.fail("no_feasible_"+stage+"_contact")
            self.site, self.path = plan["site"], plan["path"]
            self.nav_pose = state[[0,1,3]].copy()
            self.phase = "to_A" if stage == "rotate" else "to_B"
            self.phase_steps = 0
        log = dict(phase=self.phase, planning=self.last_plan,
                   actual_phase_steps=self.phase_steps, task_energy_integral=float(self.task_energy_integral),
                   rotation_deadline_steps=c.horizon if c.fixed_rotation_deadline else None,
                   surface_body=self.site["q"].tolist(), normal_body=self.site["normal"].tolist())
        if self.phase in ("to_A", "to_B"):
            if (np.linalg.norm(state[:2]-self.nav_pose[:2]) > c.pose_drift_tolerance
                    or abs(wrap(state[3]-self.nav_pose[2])) > c.pose_drift_angle):
                return self.fail("object_moved_during_navigation")
            while len(self.path)>1 and np.linalg.norm(state[8:10]-self.path[0])<c.navigation_tolerance:
                # A near waypoint may not be skipped across an obstacle corner.
                # Check the entire new segment from the measured position.
                if not self.separating_or_free(state, self.path[1]):
                    break
                self.path = self.path[1:]
            target = self.path[0]
            log["navigation_target"] = target.tolist()
            log["navigation_path_remaining"] = self.path.tolist()
            log["navigation_contact_gap"] = self.geometry.contact_gap(state)
            log["contact_exit_segment"] = bool(self.obstacle(state,c.pusher_radius).intersects(Point(state[8:10])))
            delta = target-state[8:10]
            action = c.navigation_gain*delta
            action *= min(1., c.max_speed/max(np.linalg.norm(action), 1e-12))
            endpoint = state[8:10]+DT*action
            if not self.separating_or_free(state,endpoint):
                return self.fail("navigation_collision")
            log["navigation_V"] = float(delta@delta)
            log["navigation_predicted_deltaV"] = float(
                np.sum((endpoint-target)**2)-delta@delta)
            if len(self.path)==1 and np.linalg.norm(delta)<c.navigation_tolerance:
                self.phase = "approach_A" if self.phase == "to_A" else "approach_B"
                self.phase_steps = 0
                action = np.zeros(2)
        elif self.phase in ("approach_A", "approach_B"):
            # The search starts at touch, whereas free navigation ends at pre.
            # Execute and log this physical gap-closing segment explicitly.
            target = state[:2] + rotation(state[3]) @ self.site["touch"]
            delta = target - state[8:10]
            log["approach_target"] = target.tolist()
            log["approach_gap"] = self.geometry.contact_gap(state)
            if (np.linalg.norm(state[:2]-self.nav_pose[:2]) > c.pose_drift_tolerance
                    or abs(wrap(state[3]-self.nav_pose[2])) > c.pose_drift_angle):
                return self.fail("object_moved_during_approach")
            action = 3.*delta
            action *= min(1., .03/max(np.linalg.norm(action),1e-12))
            # Permit only the intended near-contact endpoint; reject penetration.
            local_start = rotation(state[3]).T @ (state[8:10]-state[:2])
            local_end = rotation(state[3]).T @ (state[8:10]+DT*action-state[:2])
            sweep_gap = self.geometry.polygon.distance(LineString([local_start,local_end]))-c.pusher_radius
            log["approach_swept_gap"] = float(sweep_gap)
            if sweep_gap < -.0001:
                return self.fail("approach_collision")
            if np.linalg.norm(delta) < .0004:
                self.phase = "rotate" if self.phase == "approach_A" else "translate"
                self.phase_steps = 0
                action = np.zeros(2)
        elif self.phase in ("rotate", "translate"):
            reached = rest and err<=c.angle_tolerance
            if self.phase == "translate":
                reached &= position<=c.position_tolerance
            self.switch_count = self.switch_count+1 if reached else 0
            if self.switch_count >= c.switch_hold:
                self.phase = "select_B" if self.phase == "rotate" else "hold"
                self.phase_steps, self.switch_count = 0, 0
                action = np.zeros(2)
            else:
                plan = self.optimize(state, self.phase, self.site)
                if plan is None:
                    return self.fail("no_feasible_"+self.phase+"_sequence")
                action = plan["sequence"][0]
                log.update(planned_score=plan["score"], planned_violation=plan["violation"])
        else:
            action = np.zeros(2)
        action = np.asarray(action, np.float32)
        pred, nom, ctx, residual, phi = self.predict(state[None], action[None])
        if log["phase"] in ("to_A","to_B") and np.linalg.norm(action) > 0:
            projected=pred[0,8:10]
            if not self.separating_or_free(state,projected):
                return self.fail("predicted_navigation_collision")
            before=float(np.sum((state[8:10]-target)**2))
            after=float(np.sum((projected-target)**2))
            log["navigation_model_deltaV"]=after-before
            if after-before > -.1*c.navigation_gain*DT*before:
                return self.fail("navigation_clf_infeasible")
        self.pending = (state.copy(), action.copy(), pred[0], nom[0], ctx[0], residual, phi)
        self.phase_steps += 1
        return action, log

    @torch.no_grad()
    def feedback(self, real_next):
        if self.pending is None:
            raise RuntimeError("No executed prediction to compare")
        state, action, predicted, nominal, context, residual, phi = self.pending
        target = torch.as_tensor((real_next-nominal)[None], device=self.device)/self.model.residual_scale
        self.memory, _ = self.model.feedback(
            self.memory, phi, residual-target, target,
            torch.as_tensor(action[None], device=self.device))
        c = self.cfg
        self.task_energy_integral += DT*(np.sum(((real_next[:2]-self.goal[:2])/c.position_scale)**2)
            + float(angle_error(real_next,self.goal,self.geometry.period))**2)
        strict = (np.linalg.norm(real_next[:2]-self.goal[:2])<=c.position_tolerance
                  and abs(angle_error(real_next,self.goal,self.geometry.period))<=c.angle_tolerance
                  and np.linalg.norm(real_next[4:7])<=c.speed_tolerance
                  and abs(real_next[7])<=c.omega_tolerance
                  and np.linalg.norm(real_next[10:12])<=c.speed_tolerance
                  and np.linalg.norm(action)<=c.input_tolerance)
        self.hold_count = self.hold_count+1 if strict else 0
        if self.phase == "hold" and not strict:
            self.failure = "terminal_hold_lost"
        self.pending = None
        return dict(nominal=nominal, contexts=context, predictions=predicted,
                    strict_hold=self.hold_count)
