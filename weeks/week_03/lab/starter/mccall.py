"""McCall job search with the full states (U, w) and (E, w)."""

import numpy as np


class McCallModel:
    def __init__(self, wages=(1, 2, 3, 4, 5),
                 probabilities=(0.1, 0.2, 0.4, 0.2, 0.1), benefit=1.0, beta=0.9):
        self.wages = np.array(wages, dtype=float)
        self.q = np.array(probabilities, dtype=float)
        self.benefit, self.beta = benefit, beta
        self.n = len(self.wages)
        assert self.n > 0 and len(self.q) == self.n
        assert np.all(self.q >= 0) and np.isclose(self.q.sum(), 1)
        assert 0 < beta < 1
        self.states = [("U", float(w)) for w in self.wages] + [("E", float(w)) for w in self.wages]

    def actions(self, state):
        return ["accept", "reject"] if state < self.n else ["work"]

    def reward(self, state, action):
        assert action in self.actions(state)
        status, wage = self.states[state]
        return self.benefit if action == "reject" else wage

    def transition_probabilities(self, state, action):
        p = np.zeros(len(self.states))
        if action == "reject":
            p[:self.n] = self.q  # New offer, still unemployed.
        elif action == "accept":
            p[state + self.n] = 1  # Employment at the current wage.
        else:
            p[state] = 1
        return p

    def initial_values(self):
        v = np.zeros(len(self.states))
        v[self.n:] = self.wages / (1 - self.beta)  # Known employed values.
        return v

    def bellman(self, v):
        new_v = np.zeros(len(self.states))
        policy = []
        for s in range(len(self.states)):
            actions = self.actions(s)
            scores = []
            for a in actions:
                p = self.transition_probabilities(s, a)
                scores.append(self.reward(s, a) + self.beta * (p @ v))
            best = np.argmax(scores)
            new_v[s] = scores[best]
            policy.append(actions[best])
        return new_v, np.array(policy)

    def step(self, state, action, rng):
        r = self.reward(state, action)
        next_state = rng.choice(len(self.states), p=self.transition_probabilities(state, action))
        return next_state, r
