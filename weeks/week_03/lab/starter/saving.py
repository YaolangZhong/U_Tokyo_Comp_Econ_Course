"""Continuous consumption-saving, approximated by grids and midpoint integration."""

import numpy as np


class ConsumptionSavingModel:
    def __init__(self, M=5.0, B=3.0, alpha=0.5, beta=0.9,
                 n_states=101, n_actions=81, n_income=41):
        assert M >= 0 and B > 0 and 0 < alpha < 1 and 0 < beta < 1
        assert n_states >= 2 and n_actions >= 2 and n_income >= 1
        self.M, self.B, self.alpha, self.beta = M, B, alpha, beta
        # These points represent a continuous value function; they are not its state space.
        self.states = np.linspace(0, M + B, n_states)
        self.n_actions = n_actions
        self.incomes = (np.arange(n_income) + 0.5) * B / n_income

    def reward(self, state, action):
        assert 0 <= state <= self.M + self.B
        assert 0 <= action <= min(state, self.M)
        return (state - action) ** self.alpha

    def initial_values(self):
        return np.zeros(len(self.states))

    def q_values(self, state, v):
        actions = np.linspace(0, min(state, self.M), self.n_actions)
        # Each row is one saving choice; columns are possible income draws.
        next_stock = actions[:, None] + self.incomes[None, :]
        # Linear interpolation supplies values between the stored stock points.
        continuation = np.interp(next_stock, self.states, v).mean(axis=1)
        scores = (state - actions) ** self.alpha + self.beta * continuation
        return actions, scores

    def bellman(self, v):
        new_v, policy = np.zeros_like(v), np.zeros_like(v)
        for i, stock in enumerate(self.states):
            actions, scores = self.q_values(stock, v)
            best = np.argmax(scores)
            new_v[i], policy[i] = scores[best], actions[best]
        return new_v, policy

    def step(self, state, action, rng):
        reward = self.reward(state, action)
        next_state = action + rng.uniform(0, self.B)
        return next_state, reward
