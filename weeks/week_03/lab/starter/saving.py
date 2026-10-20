"""Continuous saving model, converted to a finite MDP for numerical solution."""

import numpy as np
from mdp import MDP


class ConsumptionSavingModel(MDP):
    def __init__(self, M=5.0, B=3.0, alpha=0.5, beta=0.9,
                 n_states=101, n_actions=81, n_income=41):
        assert M >= 0 and B > 0 and 0 < alpha < 1
        assert n_states >= 2 and n_actions >= 2 and n_income >= 1
        self.M, self.B, self.alpha = M, B, alpha
        self.state_bounds = (0, M + B)
        # The economic state is continuous. These are its numerical grid points.
        states = np.linspace(0, M + B, n_states)
        fractions = np.linspace(0, 1, n_actions)
        actions = np.minimum(states, M)[:, None] * fractions[None, :]
        rewards = (states[:, None] - actions) ** alpha
        self.incomes = (np.arange(n_income) + 0.5) * B / n_income
        P = np.zeros((n_states, n_actions, n_states))

        # Midpoint income draws and linear interpolation define transition weights.
        # An off-grid next stock z splits its weight between its two neighbors.
        # Thus P[i, j] @ V equals the average interpolated continuation value.
        for i in range(n_states):
            for j, saving in enumerate(actions[i]):
                next_stock = saving + self.incomes
                right = np.searchsorted(states, next_stock).clip(1, n_states - 1)
                left = right - 1
                weight_right = (next_stock - states[left]) / (states[right] - states[left])
                np.add.at(P[i, j], left, (1 - weight_right) / n_income)
                np.add.at(P[i, j], right, weight_right / n_income)
        super().__init__(states, actions, rewards, P, beta)

    def reward(self, state, action):
        assert 0 <= state <= self.M + self.B
        assert 0 <= action <= min(state, self.M)
        return (state - action) ** self.alpha

    def step(self, state, action, rng):
        # Simulation uses the original continuous income distribution.
        reward = self.reward(state, action)
        return action + rng.uniform(0, self.B), reward
