"""McCall job search as a standard MDP with full states (U, w) and (E, w)."""

import numpy as np
from mdp import MDP


class McCallModel(MDP):
    def __init__(self, wages=(1, 2, 3, 4, 5),
                 probabilities=(0.1, 0.2, 0.4, 0.2, 0.1), benefit=1.0, beta=0.9):
        self.wages = np.array(wages, dtype=float)
        self.q = np.array(probabilities, dtype=float)
        self.benefit = benefit
        self.n = len(self.wages)
        assert self.n > 0 and self.q.shape == self.wages.shape
        assert np.all(self.q >= 0) and np.isclose(self.q.sum(), 1)
        states = [("U", float(w)) for w in self.wages] + [("E", float(w)) for w in self.wages]
        actions = np.tile(["accept", "reject", "work"], (2 * self.n, 1))
        rewards = np.zeros((2 * self.n, 3))
        P = np.zeros((2 * self.n, 3, 2 * self.n))
        feasible = np.zeros((2 * self.n, 3), dtype=bool)
        for i, wage in enumerate(self.wages):
            feasible[i, :2] = True
            rewards[i, 0] = wage
            P[i, 0, self.n + i] = 1  # Accept: employed at today's offered wage.
            rewards[i, 1] = benefit
            P[i, 1, :self.n] = self.q  # Reject: draw another offer.
            feasible[self.n + i, 2] = True
            rewards[self.n + i, 2] = wage
            P[self.n + i, 2, self.n + i] = 1  # Employed: keep earning this wage.
        super().__init__(states, actions, rewards, P, beta, feasible)

    def step(self, state, action, rng):
        j = list(self.actions[state]).index(action)
        assert self.feasible[state, j]
        next_state = rng.choice(len(self.states), p=self.P[state, j])
        return next_state, self.rewards[state, j]
