"""Common finite-MDP input shared by the three model files."""

import numpy as np


class MDP:
    """One row per state; one column per possible action.

    states[i]             label of state i
    actions[i, j]         actual action represented by column j at state i
    feasible[i, j]        whether that action is allowed
    rewards[i, j]         deterministic reward r(s_i, a_j)
    P[i, j, next_i]       transition probability to state next_i
    beta                  discount factor

    A numerical approximation to a continuous model can use this same format.
    """

    def __init__(self, states, actions, rewards, P, beta, feasible=None):
        self.states = states
        self.actions = np.asarray(actions)
        self.rewards = np.asarray(rewards, dtype=float)
        self.P = np.asarray(P, dtype=float)
        self.beta = beta
        self.feasible = (np.ones_like(self.rewards, dtype=bool) if feasible is None
                         else np.asarray(feasible, dtype=bool))
        n = len(states)
        # Check only the essential dimensions and probability conditions.
        assert self.rewards.ndim == 2 and self.rewards.shape[0] == n
        assert self.actions.shape == self.feasible.shape == self.rewards.shape
        assert self.P.shape == (*self.rewards.shape, n)
        assert 0 < beta < 1 and np.all(self.feasible.any(axis=1))
        assert np.all(np.isfinite(self.rewards[self.feasible]))
        probabilities = self.P[self.feasible]
        assert np.all(probabilities >= 0)
        assert np.allclose(probabilities.sum(axis=1), 1)
