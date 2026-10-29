"""Finite-window evidence for approaching and holding the goal, not a zero theorem."""
from __future__ import annotations

import numpy as np

from .control import metrics


def convergence_report(states, actions, goal, period, config):
    if not len(actions):
        return dict(protocol="dl_push_hold_v2", transitions=0, strict_final_hold=False,
                    strict_ever_hold=False, maximum_strict_hold_steps=0, tails={})
    rows = [metrics(s, goal, period) for s in states[1:]]
    fields = {name: np.array([r[name] for r in rows]) for name in rows[0]}
    fields["input_mps"] = np.linalg.norm(actions, axis=-1)
    quiet = ((fields["object_speed_mps"] <= config.velocity_tolerance)
             & (fields["pusher_speed_mps"] <= config.velocity_tolerance)
             & (fields["object_omega_radps"] <= config.angular_velocity_tolerance)
             & (fields["input_mps"] <= config.input_tolerance))
    strict = (quiet & (fields["position_m"] <= config.position_tolerance)
              & (fields["angle_rad"] <= config.angle_tolerance))
    current = maximum = 0
    for value in strict:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    tails = {}
    for length in (20, 100, 200):
        if len(actions) >= length:
            tails[str(length)] = {
                key: dict(rms=float(np.sqrt(np.mean(values[-length:] ** 2))),
                          maximum=float(values[-length:].max()),
                          mean=float(values[-length:].mean()))
                for key, values in fields.items()
            }
    levels = {}
    for label, p, a in [("20mm_10deg", .02, np.deg2rad(10)),
                        ("5mm_3deg", .005, np.deg2rad(3)),
                        ("1mm_1deg", config.position_tolerance, config.angle_tolerance)]:
        gate = (fields["position_m"] <= p) & (fields["angle_rad"] <= a)
        levels[label] = dict(ever_pose=bool(gate.any()), final_pose=bool(gate[-1]))
    return dict(protocol="dl_push_hold_v2", transitions=len(actions),
                strict_final_hold=current >= config.hold_steps,
                strict_ever_hold=maximum >= config.hold_steps,
                maximum_strict_hold_steps=maximum, levels=levels, tails=tails)
