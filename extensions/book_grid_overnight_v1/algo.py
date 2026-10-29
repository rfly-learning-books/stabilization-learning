"""Fixed certificate construction and reward policy iteration, separate from env."""
from fractions import Fraction
import math
import numpy as np


def hitting_certificate(env):
    domain, allowed, distance, sizes = env.invariant_domain()
    policy = np.eye(5)[np.full(95, 4)]
    for i in domain:
        if env.goal[i]:
            continue
        candidates = [a for a in allowed[i] if any(distance[j] < distance[i] for j in np.flatnonzero(env.kernel[i, a]))]
        if not candidates:
            raise ValueError("No proper baseline action")
        action = min(candidates, key=lambda a: (sum(probability*distance[env.index[s]]
            for s,probability in env.exact_branches(env.states[i],a).items()), a))
        policy[i] = np.eye(5)[action]
    transient = [i for i in domain if not env.goal[i]]
    kernel = env.policy_kernel(policy)
    h = np.linalg.solve(np.eye(len(transient))-kernel[np.ix_(transient, transient)], np.ones(len(transient)))
    value = np.zeros(95); value[transient] = np.round(h, 8)
    kappa = Fraction(1, math.ceil(2*value.max()))
    return dict(domain=domain, allowed=allowed, distance=distance, sizes=sizes, baseline=policy,
                value=value, kappa=kappa, h=h, transient=transient)


def extreme_distributions(expectation, threshold):
    """Vertices of simplex intersected with one expected-drift halfspace."""
    candidates = []
    for a, v in enumerate(expectation):
        if v <= threshold+1e-12:
            candidates.append(np.eye(len(expectation))[a])
    for a in range(len(expectation)):
        for b in range(a+1, len(expectation)):
            x, y = expectation[a], expectation[b]
            if (x-threshold)*(y-threshold) < 0:
                w = (threshold-y)/(x-y)
                p = np.zeros(len(expectation)); p[a] = w; p[b] = 1-w
                candidates.append(p)
    return candidates


def policy_iteration(env, certificate=None, kappa=None, gamma=.95, tolerance=1e-10, maximum=100, evaluation_starts=None):
    constrained = certificate is not None
    domain = certificate['domain'] if constrained else list(range(95))
    vertices = {}
    policy = np.eye(5)[np.full(95, 4)]
    for i in domain:
        if env.terminal[i]:
            vertices[i] = [np.eye(5)[4]]
            continue
        if constrained:
            actions = certificate['allowed'][i]
            expectation = env.kernel[i, actions] @ certificate['value']
            threshold = (1-float(kappa))*certificate['value'][i]
            v = extreme_distributions(expectation, threshold)
            if not v:
                return dict(feasible=False, infeasible_state=env.states[i], kappa=float(kappa))
            vertices[i] = []
            for weights in v:
                p = np.zeros(5); p[actions] = weights; vertices[i].append(p)
            policy[i] = np.eye(5)[actions[int(np.argmin(expectation))]]
        else:
            vertices[i] = list(np.eye(5)); policy[i] = vertices[i][0]
    history = []
    starts = certificate['transient'] if constrained else np.flatnonzero(~env.terminal)
    if evaluation_starts is not None:
        starts = evaluation_starts
    for iteration in range(maximum):
        kernel = env.policy_kernel(policy); reward = (env.reward*policy).sum(1)
        value = np.linalg.solve(np.eye(95)-gamma*kernel, reward)
        q = env.reward+gamma*np.einsum('iaj,j->ia', env.kernel, value)
        improved = policy.copy(); residual = 0.
        for i in domain:
            objectives = np.array([p @ q[i] for p in vertices[i]])
            best = int(np.argmax(objectives))
            improved[i] = vertices[i][best]
            residual = max(residual, float(objectives[best]-value[i]))
        history.append(dict(iteration=iteration, mean_value=float(value[starts].mean()),
                            bellman_improvement_residual=residual, changed_states=int(np.any(abs(improved-policy)>1e-12, axis=1).sum())))
        if residual <= tolerance:
            return dict(feasible=True, policy=policy, reward_value=value, history=history,
                        bellman_residual=residual, evaluations=iteration+1)
        policy = improved
    raise RuntimeError("Policy iteration did not meet the frozen stopping criterion")


def exact_audit(env, certificate, policy, kappa):
    domain = set(certificate['domain']); value = [Fraction(f'{v:.8f}') for v in certificate['value']]
    residuals = []; outside = []; rational_policies = []
    for i in sorted(domain):
        fractions = [Fraction(str(float(p))).limit_denominator(10**12) for p in policy[i]]
        total = sum(fractions)
        fractions = [p/total for p in fractions]
        expected = Fraction(0); escaped = Fraction(0)
        for a, weight in enumerate(fractions):
            if not weight:
                continue
            for state, probability in env.exact_branches(env.states[i], a).items():
                j = env.index[state]
                escaped += weight*probability*(j not in domain)
                expected += weight*probability*value[j]
        residual = expected-(1-kappa)*value[i]
        outside.append(escaped); residuals.append(residual)
        rational_policies.append(dict(state=list(env.states[i]), action_probabilities=list(map(str, fractions)),
            contraction_residual=str(residual), outside_probability=str(escaped)))
    ratios = []
    for i in certificate['transient']:
        s = env.states[i]
        distance2 = min((s[0]-g[0])**2+(s[1]-g[1])**2 for g in env.goals)
        ratios.append(value[i]/distance2)
    return dict(passed=max(outside)==0 and max(residuals)<=0,
                maximum_outside_probability=str(max(outside)), maximum_contraction_residual=str(max(residuals)),
                minimum_comparison_ratio=str(min(ratios)), maximum_comparison_ratio=str(max(ratios)),
                rationalized_policy_maximum_difference=float(max(abs(float(Fraction(p))-policy[env.index[tuple(r['state'])],a])
                    for r in rational_policies for a,p in enumerate(r['action_probabilities']))),
                states=rational_policies)
