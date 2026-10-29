"""Independent result checks, imported only by the explicit numerical run mode."""
import json
import math
from pathlib import Path


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(group_id, workspace, stdout_paths):
    import numpy as np

    workspace = Path(workspace)
    checks = []

    def add(name, passed, actual=None, expected=None):
        checks.append({"name": name, "passed": bool(passed),
                       "actual": actual, "expected": expected})

    def close(name, actual, expected, atol=1e-10, rtol=1e-8):
        actual = float(actual)
        add(name, math.isfinite(actual) and math.isclose(actual, expected,
            abs_tol=atol, rel_tol=rtol), actual, expected)

    def finite_tree(value):
        if isinstance(value, dict):
            return all(finite_tree(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_tree(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)

    if group_id == "frequency_energy":
        from scipy.linalg import eigvalsh
        q = _json(workspace / "example_Q_report.json")
        c = _json(workspace / "example_controller_report.json")
        matrix = np.loadtxt(workspace / "example_Qw.csv", delimiter=",", skiprows=1)
        coefficients = np.loadtxt(workspace / "example_controller_coefficients.csv",
                                  delimiter=",", skiprows=1)
        feedback = np.loadtxt(workspace / "example_closed_loop_F.csv", delimiter=",")
        add("Q_shape_36", matrix.shape == (36, 36), list(matrix.shape), [36, 36])
        add("Q_symmetric", np.allclose(matrix, matrix.T, atol=1e-10, rtol=0))
        eig_min = float(np.linalg.eigvalsh(matrix)[0])
        add("Q_positive_definite_float_check", eig_min > 0, eig_min, "> 0")
        close("Q_report_min_eigenvalue", q["Q_eigen_min_approx"], eig_min)
        add("quadrature_order8_vs12", q["Q_order8_vs12_max_abs"] < 1e-8,
            q["Q_order8_vs12_max_abs"], "< 1e-8")
        a = np.array([[0., 1.], [-1., -.02]])
        b = np.array([0., 1.])
        beta = math.sqrt(1-.01**2)
        exp_a = math.exp(-.01)*(math.cos(beta)*np.eye(2)
                    + math.sin(beta)*(a+.01*np.eye(2))/beta)
        f0 = np.zeros((36, 36)); f0[:2, :2] = exp_a
        f0[3:36, 2:35] = np.eye(33)
        g = np.zeros(36); g[:2] = np.linalg.solve(a, (exp_a-np.eye(2)) @ b); g[2] = 1
        h = float(g @ matrix @ g)
        independent_b = g @ matrix @ f0
        independent_k = independent_b / (120+h)
        independent_f = f0 - np.outer(g, independent_k)
        add("coefficient_dimensions", coefficients.shape == (36, 2))
        add("b_from_original_energy", np.allclose(coefficients[:, 0], independent_b,
                                                  atol=1e-10, rtol=1e-9))
        add("K_from_energy_and_rho120", np.allclose(coefficients[:, 1], independent_k,
                                                   atol=1e-11, rtol=1e-9))
        add("closed_loop_recomputed", np.allclose(feedback, independent_f,
                                                  atol=1e-11, rtol=1e-9))
        close("H_recomputed", c["H"], h)
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(independent_f))))
        add("sampled_closed_loop_schur", spectral_radius < 1, spectral_radius, "< 1")
        power = np.linalg.matrix_power(independent_f, 256)
        difference = .5*matrix-power.T@matrix@power
        d_min = float(np.linalg.eigvalsh(difference)[0])
        ratio = float(eigvalsh(power.T@matrix@power, matrix)[-1])
        add("256_step_half_energy_matrix_positive", d_min > 0, d_min, "> 0")
        add("256_step_energy_ratio_below_half", ratio < .5, ratio, "< 0.5")
        close("reported_256_step_ratio", c["energy_ratio_256"], ratio)
        add("printed_example_V_decreases", c["V_next"] < c["V_initial"])
        result = {"Q": q, "controller": c}
    elif group_id == "deterministic_disk":
        result = _json(stdout_paths[-1])
        close("V_ratio", result["V_ratio"], .81)
        close("gain_limit", result["gain_limit"], 1.)
        add("prescribed_grid_and_horizon", result["initial_points"] == 101
            and result["steps"] == 200)
        for key in ["max_energy_equality_residual", "max_derivative_equality_residual",
                    "max_domain_violation", "max_intersample_domain_violation",
                    "max_action_violation", "final_energy_equality_residual"]:
            add(key, result[key] <= 1e-12, result[key], "<= 1e-12")
        close("zero_action_at_goal", result["goal_action"], 0.)
        close("terminal_error_analytic", result["terminal_max_abs_error"],
              .05*.9**200, atol=1e-18)
    elif group_id == "stochastic_disk":
        result = _json(workspace / "disk_data/results.json")
        train = np.loadtxt(workspace / "disk_data/training_transitions.csv", delimiter=",", skiprows=1)
        valid = np.loadtxt(workspace / "disk_data/validation_transitions.csv", delimiter=",", skiprows=1)
        add("training_shape", train.shape == (8000, 4), list(train.shape), [8000, 4])
        add("validation_shape", valid.shape == (4000, 4), list(valid.shape), [4000, 4])
        add("distinct_fixed_seeds", result["training_seed"] == 202610211
            and result["validation_seed"] == 202610212)
        xi, action, following, labels = train.T
        vxi, va, vfollowing, ratios = valid.T
        dt, radius, limit = .1, .05, .05
        inferred_gains = (following-xi)/(dt*action)
        validation_gains = (vfollowing-vxi)/(dt*va)
        for name, before, act, after, gains in [
                ("train", xi, action, following, inferred_gains),
                ("valid", vxi, va, vfollowing, validation_gains)]:
            add(name+"_finite", np.isfinite(np.column_stack((before, act, after))).all())
            add(name+"_input_and_domain", np.all((act >= 0) & (act <= limit))
                and np.all((before >= -radius) & (before <= 0))
                and np.all((after >= -radius) & (after <= 0)))
            add(name+"_gain_support", np.all((gains >= .8-1e-12) & (gains <= 1.2+1e-12)))
        add("D_labels_from_physical_successors", np.allclose(labels,
            .5*(following**2-xi**2), atol=1e-16, rtol=1e-12))
        add("validation_ratios_from_successors", np.allclose(ratios,
            (vfollowing/vxi)**2, atol=1e-14, rtol=1e-12))
        features = np.column_stack((xi*action, action**2))
        add("D_features_identifiable", np.linalg.matrix_rank(features) == 2)
        close("selected_K", result["selected_K"], 1.)
        add("validation_action_is_selected_policy", np.allclose(va, -result["selected_K"]*vxi))
        close("validation_mean_independent", result["q_validation"], float(ratios.mean()))
        analytic = 1-2*dt*1.+dt**2*(1.+.4**2/12)
        close("analytic_mean_square_ratio", result["q_analytic"], analytic)
        z_min = (1-dt*1.2*result["selected_K"])**2
        margin = (1-z_min)*math.sqrt(math.log(100)/(2*4000))
        close("Hoeffding_margin_recomputed", result["hoeffding_margin"], margin)
        close("q_upper_recomputed", result["q_upper"], min(1., float(ratios.mean())+margin))
        add("independent_validation_contraction", result["q_upper"] < 1., result["q_upper"], "< 1")
    elif group_id == "minimal_certificate":
        result = _json(stdout_paths[-1])
        i, g, d = result["integrator"], result["grid"], result["discount_counterexample"]
        close("metric_trace2", i["metric_trace"], 2.)
        add("metric_spectrum", min(i["metric_eigenvalues"]) >= .1-1e-12
            and max(i["metric_eigenvalues"]) <= 1.9+1e-12)
        add("integrator_contraction_bound", 0 < i["contraction_bound"] < 1)
        add("integrator_step_residual", i["max_contraction_residual"] <= 1e-12)
        add("integrator_final_bound", i["final_V_after_200_steps"]
            <= i["contraction_bound"]**200*i["initial_V"]+1e-12)
        add("grid_is_25_states", g["states"] == 25, g["states"], 25)
        close("grid_Bellman_residual", g["max_Bellman_residual"], 0., atol=1e-12)
        close("grid_goal_absorbing_drift", g["target_drift"], 0., atol=1e-12)
        close("grid_max_hitting_steps", g["max_expected_hitting_steps"], 4/.75)
        close("discount_value_at_one", d["value_at_one"], 2.)
        close("discount_positive_drift", d["positive_drift"], 6.)
        close("discount_Bellman_residual", d["Bellman_residual"], 0., atol=1e-12)
    elif group_id == "observer_guidance":
        result = _json(stdout_paths[-1])
        expected_gains = [.2, .8, 1.5, 2.1]
        add("four_observer_gains", [x["gain"] for x in result["observer"]] == expected_gains)
        for row in result["observer"]:
            gain = row["gain"]; q = abs(1-gain)
            close(f"observer_q_{gain}", row["q"], q)
            close(f"observer_energy_factor_{gain}", row["energy_factor"], q*q)
            if q < 1:
                close(f"observer_noise_bound_{gain}", row["ultimate_bound"], gain*.01/(1-q))
            else:
                add(f"unstable_gain_{gain}_has_no_contraction_bound", row["ultimate_bound"] is None)
        expected = [(.1, 1.125), (.25, .28125), (.6, 4.)]
        add("three_guidance_steps", len(result["quadratic_guidance"]) == len(expected))
        for row, (step, value) in zip(result["quadratic_guidance"], expected):
            close(f"guidance_step_{step}", row["step_size"], step)
            close(f"guidance_energy_{step}", row["energy_after_one_step"], value)
        add("nine_Gaussian_KL_checks", result["gaussian_kl_checks"] == 9)
    else:
        raise ValueError(f"No implemented audit for {group_id}")
    add("all_reported_numbers_finite", finite_tree(result))
    return {"group": group_id, "passed": all(c["passed"] for c in checks),
            "check_count": len(checks), "checks": checks, "result": result,
            "interpretation": "Numerical checks of the stated low-dimensional models; not full-task training or a global proof."}
