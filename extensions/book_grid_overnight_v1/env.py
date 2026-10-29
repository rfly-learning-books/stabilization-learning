"""The 95-state grid specified in Reviewed chapter9, without learned objects."""
from collections import deque
from fractions import Fraction
import numpy as np


class GridEnv:
    moves = ((0, 1), (0, -1), (-1, 0), (1, 0), (0, 0))
    obstacles = {(3, 3), (3, 4), (3, 5), (7, 7), (7, 8)}
    goals = {(x, y) for x in (8, 9) for y in (8, 9)}
    traps = {(x, y) for x in (4, 5, 6) for y in (4, 5, 6)} | {(x, y) for x in (7, 8, 9) for y in (1, 2, 3)}

    def __init__(self, main_probability=.8):
        if not 0 < main_probability < 1:
            raise ValueError("Nonzero forward and lateral branches required")
        self.main = Fraction(str(main_probability))
        self.states = sorted({(x, y) for x in range(10) for y in range(10)} - self.obstacles)
        self.index = {s: i for i, s in enumerate(self.states)}
        self.goal = np.array([s in self.goals for s in self.states])
        self.trap = np.array([s in self.traps for s in self.states])
        self.terminal = self.goal | self.trap
        self.kernel = np.zeros((95, 5, 95))
        self.rewards = np.zeros_like(self.kernel)
        for i, s in enumerate(self.states):
            for a in range(5):
                for target, prob in self.exact_branches(s, a).items():
                    j = self.index[target]
                    self.kernel[i, a, j] = float(prob)
                    self.rewards[i, a, j] = 0 if self.terminal[i] else 10 if self.goal[j] else -10 if self.trap[j] else -.01
        self.reward = (self.kernel * self.rewards).sum(2)
        if not np.allclose(self.kernel.sum(2), 1, rtol=0, atol=1e-15):
            raise AssertionError("Invalid transition kernel")

    def exact_branches(self, state, action):
        if state in self.goals | self.traps or action == 4:
            return {state: Fraction(1)}
        dx, dy = self.moves[action]
        side = (1-self.main)/2
        branches = ((dx, dy, self.main), (-dy, dx, side), (dy, -dx, side))
        result = {}
        for x, y, probability in branches:
            target = state[0]+x, state[1]+y
            if target not in self.index:
                target = state
            result[target] = result.get(target, Fraction(0)) + probability
        return result

    def invariant_domain(self):
        domain = set(np.flatnonzero(~self.trap))
        sizes = []
        while True:
            allowed = {i: [a for a in range(5) if set(np.flatnonzero(self.kernel[i, a])) <= domain] for i in domain}
            pred = {i: set() for i in domain}
            for i in domain:
                for a in allowed[i]:
                    for j in np.flatnonzero(self.kernel[i, a]):
                        pred[j].add(i)
            distance = {int(i): 0 for i in np.flatnonzero(self.goal)}
            queue = deque(distance)
            while queue:
                j = queue.popleft()
                for i in sorted(pred[j]):
                    if i not in distance:
                        distance[i] = distance[j]+1
                        queue.append(i)
            reachable = set(distance)
            sizes.append(len(reachable))
            if reachable == domain:
                return sorted(domain), allowed, distance, sizes
            domain = reachable

    def policy_kernel(self, policy):
        if policy.shape != (95, 5) or (policy < -1e-13).any() or not np.allclose(policy.sum(1), 1):
            raise ValueError("Invalid action probabilities")
        return np.einsum('ia,iaj->ij', policy, self.kernel)

    def propagate(self, policy, starts, horizon, gamma=.95):
        kernel = self.policy_kernel(policy)
        distribution = np.zeros(95); distribution[starts] = 1/len(starts)
        reward = (policy*self.reward).sum(1)
        history = []; total = 0.
        for t in range(horizon):
            total += gamma**t * float(distribution @ reward)
            distribution = distribution @ kernel
            history.append([float(distribution[self.goal].sum()), float(distribution[self.trap].sum()),
                            float(distribution[~self.terminal].sum()), total])
        return np.asarray(history)

    def sample(self, policy, initial_states, horizon, seed, gamma=.95):
        """Independent episodes; terminal episodes never receive further rewards."""
        rng = np.random.default_rng(seed)
        state = np.array(initial_states, int).copy(); n = len(state)
        states = np.full((n, horizon+1), 255, np.uint8); states[:, 0] = state
        actions = np.full((n, horizon), 255, np.uint8)
        total = np.zeros(n); steps = np.zeros(n, int)
        pcdf = np.cumsum(self.kernel, 2); acdf = np.cumsum(policy, 1)
        for t in range(horizon):
            ids = np.flatnonzero(~self.terminal[state])
            if not len(ids):
                break
            s = state[ids]
            a = (rng.random(len(ids))[:, None] > acdf[s]).sum(1).clip(max=4)
            following = (rng.random(len(ids))[:, None] > pcdf[s, a]).sum(1).clip(max=94)
            total[ids] += gamma**t*self.rewards[s, a, following]
            state[ids] = following; steps[ids] += 1
            actions[ids, t] = a; states[ids, t+1] = following
        return dict(states=states, actions=actions, steps=steps, discounted_return=total,
                    goal=self.goal[state], trap=self.trap[state], timeout=~self.terminal[state])
