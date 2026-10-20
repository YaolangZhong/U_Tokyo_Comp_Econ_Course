"""The 47-cell lecture maze as a standard finite MDP."""

import numpy as np
from mdp import MDP


class Maze(MDP):
    def __init__(self, beta=0.9):
        self.walls = {(2, 2), (2, 3), (2, 4), (7, 3), (7, 4), (7, 5), (5, 1)}
        self.cells = [(x, y) for y in range(5, -1, -1) for x in range(9)
                      if (x, y) not in self.walls]
        n = len(self.cells)
        self.start, self.goal = 15, 7  # Lecture cells 16 and 8; Python starts at 0.
        self.action_names = ["U", "D", "L", "R"]
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        lookup = {cell: i for i, cell in enumerate(self.cells)}
        self.next_states = np.zeros((n, 4), dtype=int)
        P = np.zeros((n, 4, n))
        for i, (x, y) in enumerate(self.cells):
            for j, (dx, dy) in enumerate(directions):
                # Hitting a wall leaves the agent in place; the goal is absorbing.
                next_i = self.goal if i == self.goal else lookup.get((x + dx, y + dy), i)
                self.next_states[i, j] = next_i
                P[i, j, next_i] = 1
        actions = np.tile(np.arange(4), (n, 1))
        rewards = -np.ones((n, 4))
        rewards[self.goal, :] = 0  # Entering the goal still costs one.
        super().__init__(np.arange(n), actions, rewards, P, beta)

    def step(self, state, action, rng):
        return self.next_states[state, action], self.rewards[state, action]

    def path(self, policy):
        """Trace the policy from start to goal; values in policy are action codes."""
        state, path = self.start, [self.start]
        for _ in self.states:
            if state == self.goal:
                return path
            state = self.next_states[state, policy[state]]
            if state in path:
                raise ValueError("This policy cycles before reaching the goal.")
            path.append(state)
        raise ValueError("This policy does not reach the goal.")
