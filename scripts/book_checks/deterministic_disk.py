"""Chapter 8's ideal one-sided disk example; analytic check, no M learning.

This is a new implementation of the printed equations, not recovered book code.
Run only through run_book_checks.py so outputs stay in an isolated run directory.
"""
import json
import numpy as np


def main():
    kappa, dt, radius, action_max, gain = 1.0, 0.1, 0.05, 0.05, 1.0
    initial = np.linspace(-radius, 0.0, 101)
    error = initial.copy()
    energy_initial = 0.5 * error**2
    factor = (1.0 - kappa * gain * dt)**2
    max_energy_residual = max_domain_violation = max_action_violation = 0.0
    max_derivative_residual = max_intersample_violation = 0.0
    for _ in range(200):
        action = -gain * error
        following = error + dt * kappa * action
        residual = 0.5 * following**2 - factor * 0.5 * error**2
        max_energy_residual = max(max_energy_residual, float(np.max(np.abs(residual))))
        max_domain_violation = max(max_domain_violation, float(np.max(following)),
                                   float(np.max(-radius - following)))
        max_action_violation = max(max_action_violation, float(np.max(-action)),
                                   float(np.max(action - action_max)))
        derivative = error * kappa * action
        max_derivative_residual = max(max_derivative_residual,
                                     float(np.max(np.abs(derivative + kappa*gain*error**2))))
        # A held action has an affine path; its extrema occur at these endpoints.
        endpoints = np.stack((error, following))
        max_intersample_violation = max(max_intersample_violation,
                                       float(np.max(endpoints)),
                                       float(np.max(-radius - endpoints)))
        error = following
    final_residual = float(np.max(np.abs(0.5*error**2-factor**200*energy_initial)))
    result = {
        "scope": "ideal one-dimensional fixed-contact model, one-sided domain; no learning",
        "parameters": {"kappa": kappa, "dt": dt, "radius": radius,
                       "action_max": action_max, "K": gain},
        "initial_points": len(initial), "steps": 200,
        "V_ratio": factor, "gain_limit": min(action_max/radius, 1/(kappa*dt)),
        "max_energy_equality_residual": max_energy_residual,
        "max_derivative_equality_residual": max_derivative_residual,
        "max_domain_violation": max_domain_violation,
        "max_intersample_domain_violation": max_intersample_violation,
        "max_action_violation": max_action_violation,
        "final_energy_equality_residual": final_residual,
        "terminal_max_abs_error": float(np.max(np.abs(error))),
        "goal_action": float(-gain * 0.0),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
