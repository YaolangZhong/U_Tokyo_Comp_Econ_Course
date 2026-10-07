"""Companion to markov_processes.ipynb; code cells in notebook order.
Complete the maze-policy and cake-eating exercises. Reference solutions are
separate functions: call maze_solution() or cake_solution() to check them.
"""

import numpy as np
import matplotlib.pyplot as plt

states = ["Unemployed", "Employed"]   # A Python list
current_state = 0
alpha = 0.20
p_unemployed = np.array([1 - alpha, alpha])

print("Current state:", states[current_state])
print("Next-state probabilities [U, E]:", p_unemployed)
print("Sum:", p_unemployed.sum())

delta = 0.05
P = np.array([[1 - alpha, alpha],
              [delta, 1 - delta]])
mu0 = np.array([0.60, 0.40])
mu1 = mu0 @ P

print("P:", P, sep="\n")
print("U to E probability:", P[0, 1])
print("Next distribution:", mu1)       # [0.50, 0.50]
assert np.all(P >= 0)
assert np.allclose(P.sum(axis=1), 1)   # Each row sums to one

class MarkovProcess:
    def __init__(self, states, P):
        self.states = states
        self.P = np.array(P)
        assert self.P.shape == (len(states), len(states))
        assert np.all(self.P >= 0)
        assert np.allclose(self.P.sum(axis=1), 1)

    def simulate(self, initial_state, periods, seed=7):
        rng = np.random.default_rng(seed)
        state = initial_state
        path = [state]
        for period in range(periods):
            state = rng.choice(len(self.states), p=self.P[state])
            path.append(state)
        return np.array(path)

employment = MarkovProcess(states, P)
path = employment.simulate(initial_state=0, periods=24, seed=7)
print("Simulated states:", path)

plt.figure(figsize=(7, 2.5))
plt.step(range(len(path)), path, where="post")
plt.yticks([0, 1], states)
plt.xlabel("Month")
plt.title("One simulated employment path")
plt.ylim(-0.2, 1.2)
plt.tight_layout()
plt.show()

class MazeMDP:
    def __init__(self):
        self.goal = np.array([0, 2])
        self.wall = np.array([1, 1])
        self.beta = 0.95
        self.moves = {
            "up": np.array([-1, 0]),
            "down": np.array([1, 0]),
            "left": np.array([0, -1]),
            "right": np.array([0, 1])
        }

    def step(self, state, action):
        if np.array_equal(state, self.goal):
            return state.copy(), 0
        next_state = state + self.moves[action]
        inside = np.all(next_state >= 0) and np.all(next_state < 3)
        if not inside or np.array_equal(next_state, self.wall):
            next_state = state.copy()
        return next_state, -1

    def simulate(self, policy, initial_state, periods):
        state = np.array(initial_state)
        path = [state.copy()]
        rewards = []
        for period in range(periods):
            action = policy(state)
            state, reward = self.step(state, action)
            path.append(state.copy())
            rewards.append(reward)
        return np.array(path), np.array(rewards)

def up_then_right(state):
    if state[0] > 0:
        return "up"
    else:
        return "right"

maze = MazeMDP()
maze_path, rewards = maze.simulate(up_then_right, [2, 0], periods=6)
print("Locations:", maze_path, sep="\n")
print("Rewards:", rewards)

def right_then_up(state):
    pass

# path, rewards = maze.simulate(right_then_up, [2, 0], periods=6)
# print(path)
# print(rewards)
# assert np.array_equal(path[4:], [[0, 2], [0, 2], [0, 2]])
# assert np.array_equal(rewards, [-1, -1, -1, -1, 0, 0])

#@title Solution 1: Maze policy (double-click to show code)
#@markdown The code is hidden. Double-click to reveal it; run the cell to define `maze_solution()`, then call that function in a new cell to check the solution.
def maze_solution():
    # Keep this reference policy separate from your exercise code.
    def right_then_up(state):
        if state[1] < 2:
            return "right"
        else:
            return "up"

    maze = MazeMDP()
    path, rewards = maze.simulate(right_then_up, [2, 0], periods=6)
    print("Locations:", path, sep="\n")
    print("Rewards:", rewards)
    assert np.array_equal(path, [[2, 0], [2, 1], [2, 2], [1, 2],
                                 [0, 2], [0, 2], [0, 2]])
    assert np.array_equal(rewards, [-1, -1, -1, -1, 0, 0])
    next_state, reward = maze.step(np.array([1, 0]), "right")
    assert np.array_equal(next_state, [1, 0]) and reward == -1
    return path, rewards

class CakeMDP:
    def __init__(self):
        self.beta = 0.95

    def step(self, state, action):
        # Check feasibility, then return next stock and reward.
        pass

    def simulate(self, policy, initial_state, periods):
        state = initial_state
        path, rewards = [state], []
        for period in range(periods):
            action = policy(state)
            state, reward = self.step(state, action)
            path.append(state)
            rewards.append(reward)
        return np.array(path), np.array(rewards)


def consume_one(state):
    # Consume one unit if state > 0; otherwise consume zero.
    pass

# cake = CakeMDP()
# cake_path, cake_rewards = cake.simulate(consume_one, 4, periods=5)
# print("Cake remaining:", cake_path)
# print("Rewards:", cake_rewards)
# assert np.array_equal(cake_path, [4, 3, 2, 1, 0, 0])
# assert np.allclose(cake_rewards, [1, 1, 1, 1, 0])
# stock, reward = cake.step(4, 2)
# assert stock == 2 and np.isclose(reward, np.sqrt(2))

#@title Solution 2: Cake eating (double-click to show code)
#@markdown The code is hidden. Double-click to reveal it; run the cell to define `cake_solution()`, then call that function in a new cell to check the solution.
def cake_solution():
    # These local definitions leave your CakeMDP and policies unchanged.
    class CakeMDP:
        def __init__(self):
            self.beta = 0.95

        def step(self, state, action):
            assert isinstance(state, (int, np.integer)) and 0 <= state <= 4
            assert isinstance(action, (int, np.integer)) and 0 <= action <= state
            return state - action, np.sqrt(action)

        def simulate(self, policy, initial_state, periods):
            state = initial_state
            path = [state]
            rewards = []
            for period in range(periods):
                action = policy(state)
                state, reward = self.step(state, action)
                path.append(state)
                rewards.append(reward)
            return np.array(path), np.array(rewards)

    def consume_one(state):
        return min(1, state)

    cake = CakeMDP()
    path_one, rewards_one = cake.simulate(consume_one, 4, periods=5)
    print("Consume one: stocks", path_one, "rewards", rewards_one)
    assert np.array_equal(path_one, [4, 3, 2, 1, 0, 0])
    assert np.allclose(rewards_one, [1, 1, 1, 1, 0])
    next_stock, reward = cake.step(4, 2)
    assert next_stock == 2 and np.isclose(reward, np.sqrt(2))
    next_stock, reward = cake.step(0, 0)
    assert next_stock == 0 and reward == 0
    return cake, path_one, rewards_one
