"""Companion to markov_processes.ipynb; supplied code cells in notebook order.
The final CakeMDP exercise is intentionally a scaffold for students.
"""

import sys
print("Python version:", sys.version.split()[0])

# The ordering must be the same in every probability vector and matrix.
states = ["Unemployed", "Employed"]
current_state = 0
print("State names:", states)
print("Current state:", states[current_state])

import numpy as np

alpha = 0.20
p_unemployed = np.array([1 - alpha, alpha])
print("Next-state probabilities [U, E]:", p_unemployed)
print("Probability of finding a job:", p_unemployed[1])
print("Sum:", p_unemployed.sum())

print("All probabilities are nonnegative:", np.all(p_unemployed >= 0))
assert np.all(p_unemployed >= 0)
assert np.allclose(p_unemployed.sum(), 1)

delta = 0.05
P = np.array([
    [1 - alpha, alpha],
    [delta, 1 - delta]
])
print("Transition matrix:")
print(P)
print("Unemployed to employed:", P[0, 1])
print("Employed to unemployed:", P[1, 0])
print("Row sums:", P.sum(axis=1))
assert np.all(P >= 0)
assert np.allclose(P.sum(axis=1), 1)

mu0 = np.array([0.60, 0.40])
mu1 = mu0 @ P
print("Initial probabilities:", mu0)
print("Next-month probabilities:", mu1)

# Two ways to end up unemployed next month.
unemployment_next = mu0[0] * P[0, 0] + mu0[1] * P[1, 0]
print("Unemployment, calculated separately:", unemployment_next)
assert np.allclose(mu1[0], unemployment_next)
assert np.allclose(mu1.sum(), 1)

rng = np.random.default_rng(7)
current_state = 0
next_state = rng.choice(2, p=P[current_state])
print("Today:", states[current_state])
print("One simulated next month:", states[next_state])

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
print("State indices from month 0 to month 24:")
print(path)
print("Initial state:", employment.states[path[0]])
print("Final state:", employment.states[path[-1]])

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3))
plt.step(range(len(path)), path, where="post")
plt.yticks([0, 1], states)
plt.xlabel("Month")
plt.title("One simulated employment path")
plt.ylim(-0.2, 1.2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

alpha_trial = 0.40
P_trial = np.array([
    [1 - alpha_trial, alpha_trial],
    [delta, 1 - delta]
])
employment_trial = MarkovProcess(states, P_trial)
print("Baseline next-month distribution:", mu0 @ employment.P)
print("Higher job-finding probability:", mu0 @ employment_trial.P)
print("A new simulated path:", employment_trial.simulate(0, 24, seed=7))

state = np.array([2, 0])
move_up = np.array([-1, 0])
print("Current location:", state)
print("After moving up:", state + move_up)

def up_then_right(state):
    if state[0] > 0:
        return "up"
    else:
        return "right"

print("Action at the start:", up_then_right(np.array([2, 0])))
print("Action at the top-left corner:", up_then_right(np.array([0, 0])))

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

maze = MazeMDP()
maze_path, rewards = maze.simulate(up_then_right, initial_state=[2, 0], periods=6)
print("Locations, including the start:")
print(maze_path)
print("Rewards for the six moves:", rewards)

plt.figure(figsize=(4, 4))
plt.scatter(maze.wall[1], maze.wall[0], marker="s", s=1800, color="gray", label="Wall")
plt.plot(maze_path[:, 1], maze_path[:, 0], "o-", label="Policy path")
plt.text(0, 2, " S", fontsize=14)
plt.text(2, 0, " G", fontsize=14)
plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])
plt.xlim(-0.5, 2.5)
plt.ylim(2.5, -0.5)
plt.xlabel("Column")
plt.ylabel("Row")
plt.title("Maze path: up first, then right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

def always_right(state):
    return "right"

stuck_path, stuck_rewards = maze.simulate(always_right, [2, 0], periods=6)
print("Always-right locations:")
print(stuck_path)
print("Rewards:", stuck_rewards)

# Define your right_then_up(state) policy here.
# Then call maze.simulate(right_then_up, [2, 0], periods=6).
# Print the returned path and rewards.

class CakeMDP:
    def __init__(self):
        self.beta = 0.95

    def step(self, state, action):
        # Check: action is a whole number between 0 and state.
        # Return: next state, reward.
        pass

    def simulate(self, policy, initial_state, periods):
        # Start a path and reward list.
        # Each period: choose an action, step, and record the results.
        # Return two NumPy arrays, as in MazeMDP.simulate.
        pass


def consume_one(state):
    # Consume one unit if cake remains; otherwise consume zero.
    pass

# Run these checks after completing Task A and the policy.
# cake = CakeMDP()
# cake_path, cake_rewards = cake.simulate(consume_one, initial_state=4, periods=5)
# print("Cake remaining:", cake_path)
# print("Rewards:", cake_rewards)
# assert np.array_equal(cake_path, [4, 3, 2, 1, 0, 0])
# assert np.allclose(cake_rewards, [1, 1, 1, 1, 0])
# next_stock, reward = cake.step(4, 2)
# assert next_stock == 2 and np.isclose(reward, np.sqrt(2))
# next_stock, reward = cake.step(0, 0)
# assert next_stock == 0 and reward == 0

# Define consume_two(state), then simulate it with the same CakeMDP object.
# Print and compare the stock and reward paths.
