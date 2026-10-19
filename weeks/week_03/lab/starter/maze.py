"""The 47-cell lecture maze; array index 0 corresponds to cell 1."""

import numpy as np


class Maze:
    def __init__(self, beta=0.9):
        assert 0 < beta < 1
        self.beta = beta
        self.walls = {(2, 2), (2, 3), (2, 4), (7, 3), (7, 4), (7, 5), (5, 1)}
        self.cells = [(x, y) for y in range(5, -1, -1) for x in range(9)
                      if (x, y) not in self.walls]
        self.states = np.arange(len(self.cells))
        self.start, self.goal = 15, 7  # Lecture cell numbers 16 and 8.
        self.actions = ["U", "D", "L", "R"]
        self.directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        lookup = {cell: i for i, cell in enumerate(self.cells)}
        self.next_states = np.zeros((len(self.states), 4), dtype=int)
        for s, (x, y) in enumerate(self.cells):
            for a, (dx, dy) in enumerate(self.directions):
                # Hitting a wall or the boundary leaves the agent in place.
                self.next_states[s, a] = lookup.get((x + dx, y + dy), s)
        self.next_states[self.goal, :] = self.goal  # Absorbing goal.
        self.rewards = -np.ones(len(self.states))
        self.rewards[self.goal] = 0  # Entering the goal still costs one.

    def initial_values(self):
        return np.zeros(len(self.states))

    def q_values(self, v):
        # Rows are states; columns are actions. Rewards broadcast across columns.
        return self.rewards[:, None] + self.beta * v[self.next_states]

    def bellman(self, v):
        q = self.q_values(v)
        return q.max(axis=1), q.argmax(axis=1)

    def step(self, state, action, rng):
        return self.next_states[state, action], self.rewards[state]

    def path(self, policy):
        """Trace a deterministic policy from the start to the goal."""
        state, path = self.start, [self.start]
        for _ in self.states:
            if state == self.goal:
                return path
            state = self.next_states[state, policy[state]]
            if state in path:
                raise ValueError("This policy cycles before reaching the goal.")
            path.append(state)
        raise ValueError("This policy does not reach the goal.")
