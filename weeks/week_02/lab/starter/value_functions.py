"""Companion to value_functions.ipynb; code cells in notebook order.
Evaluate specified policies in McCall job search and consumption-saving models.
Call mccall_solution() or savings_solution() separately to run exercise solutions.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

print("Python:", sys.version.split()[0])
np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.figsize": (8, 3.5), "axes.spines.top": False,
                     "axes.spines.right": False})

def simulate_policy(model, policy, initial_state, periods, seed=7):
    rng = np.random.default_rng(seed)
    state = int(initial_state)
    if state not in range(len(model.states)) or periods < 0:
        raise ValueError("Use a valid initial state and a nonnegative horizon.")
    path, rewards = [state], []
    for _ in range(periods):
        action = policy(state)
        state, reward = model.step(state, action, rng)
        path.append(state)
        rewards.append(reward)
    return np.array(path), np.array(rewards, dtype=float)


def discounted_return(rewards, beta):
    weights = beta ** np.arange(len(rewards))
    return float(weights @ rewards)

class McCallModel:
    def __init__(self, wages=(1, 2, 3, 4, 5),
                 probabilities=(0.1, 0.2, 0.4, 0.2, 0.1), benefit=1.0, beta=0.9):
        self.wages = np.asarray(wages, dtype=float)
        self.q = np.asarray(probabilities, dtype=float)
        self.benefit, self.beta = float(benefit), float(beta)
        if (self.wages.ndim != 1 or len(self.wages) == 0
                or self.q.shape != self.wages.shape
                or not np.all(np.isfinite(self.wages)) or np.any(self.wages <= 0)
                or not np.all(np.diff(self.wages) > 0)
                or not np.all(np.isfinite(self.q)) or np.any(self.q < 0)
                or not np.isclose(self.q.sum(), 1)
                or not np.isfinite(self.benefit) or self.benefit < 0
                or not 0 < self.beta < 1):
            raise ValueError("Check wages, offer probabilities, benefit, and beta.")
        self.n = len(self.wages)
        self.states = [(status, float(wage)) for status in ("U", "E") for wage in self.wages]

    def actions(self, state):
        if not isinstance(state, (int, np.integer)) or not 0 <= state < 2 * self.n:
            raise ValueError("Invalid state index.")
        return ("accept", "reject") if state < self.n else ("work",)

    def reward(self, state, action):
        if action not in self.actions(state):
            raise ValueError("This action is unavailable at the current state.")
        return self.benefit if action == "reject" else self.wages[state % self.n]

    def transition_probabilities(self, state, action):
        self.reward(state, action)  # Also check feasibility.
        p = np.zeros(len(self.states))
        if action == "reject":
            p[:self.n] = self.q
        elif action == "accept":
            p[self.n + state] = 1.0
        else:
            p[state] = 1.0
        return p

    def step(self, state, action, rng):
        reward = self.reward(state, action)
        p = self.transition_probabilities(state, action)
        next_state = int(rng.choice(len(self.states), p=p))
        return next_state, reward

    def simulate(self, policy, initial_state, periods, seed=7):
        return simulate_policy(self, policy, initial_state, periods, seed)

mccall = McCallModel()
print("State indices and labels:")
for index, label in enumerate(mccall.states):
    print(index, label)
print("Reject an offer: probabilities over all states")
print(mccall.transition_probabilities(1, "reject"))
print("Accept wage 2: probabilities over all states")
print(mccall.transition_probabilities(1, "accept"))

def make_threshold_policy(model, threshold):
    def policy(state):
        if state >= model.n:
            return "work"
        return "accept" if model.wages[state] >= threshold else "reject"
    return policy


search_policy = make_threshold_policy(mccall, threshold=3)
search_start = 1  # (U, wage 2)
search_path, search_rewards = mccall.simulate(search_policy, search_start, periods=12)
print("Rollout:", [mccall.states[s] for s in search_path])
print("Reward stream:", search_rewards)
print("12-step return:", discounted_return(search_rewards, mccall.beta))

def policy_arrays(model, policy):
    n = len(model.states)
    P = np.zeros((n, n))
    r = np.zeros(n)
    for state in range(n):
        action = policy(state)
        P[state] = model.transition_probabilities(state, action)
        r[state] = model.reward(state, action)
    if np.any(P < 0) or not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Every transition row must be a probability vector.")
    return P, r


def evaluate_policy(model, policy):
    P, r = policy_arrays(model, policy)
    V = np.linalg.solve(np.eye(len(r)) - model.beta * P, r)
    return V, P, r


V_search, P_search, r_search = evaluate_policy(mccall, search_policy)
print("Offer | action | value while unemployed")
for i, wage in enumerate(mccall.wages):
    print(f"{wage:5.1f} | {search_policy(i):6s} | {V_search[i]:.4f}")
print("Value from (U, wage 2):", V_search[search_start])
print("Employed-state values:", V_search[mccall.n:])
assert np.allclose(V_search[mccall.n:], mccall.wages / (1 - mccall.beta))

fig, ax = plt.subplots()
ax.plot(mccall.wages, V_search[:mccall.n], "o-", label="Supplied threshold = 3")
ax.plot(mccall.wages, mccall.wages / (1 - mccall.beta), "s--", label="Accept every offer")
ax.set(xlabel="Current wage offer", ylabel="Expected discounted income")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

def estimate_value(model, policy, initial_state, repetitions=600, periods=100, seed=21):
    rng = np.random.default_rng(seed)
    returns = np.empty(repetitions)
    for i in range(repetitions):
        rollout_seed = int(rng.integers(0, 2**32))
        _, rewards = model.simulate(policy, initial_state, periods, seed=rollout_seed)
        returns[i] = discounted_return(rewards, model.beta)
    _, r = policy_arrays(model, policy)
    return {"mean": returns.mean(),
            "standard_error": returns.std(ddof=1) / np.sqrt(repetitions),
            "tail_bound": model.beta**periods * np.max(np.abs(r)) / (1 - model.beta)}


search_estimate = estimate_value(mccall, search_policy, search_start)
print("Exact infinite-horizon value:", V_search[search_start])
print("Simulated mean:", search_estimate["mean"])
print("Monte Carlo standard error:", search_estimate["standard_error"])
print("Maximum omitted tail:", search_estimate["tail_bound"])

class ConsumptionSavingModel:
    def __init__(self, M=5, B=3, alpha=0.5, beta=0.9):
        if (not isinstance(M, (int, np.integer)) or M < 0
                or not isinstance(B, (int, np.integer)) or B < 0
                or not 0 < alpha < 1 or not 0 < beta < 1):
            raise ValueError("Use nonnegative integer M and B, and alpha/beta in (0, 1).")
        self.M, self.B = int(M), int(B)
        self.alpha, self.beta = float(alpha), float(beta)
        self.states = np.arange(self.M + self.B + 1)
        self.incomes = np.arange(self.B + 1)
        self.q = np.full(self.B + 1, 1 / (self.B + 1))

    def actions(self, state):
        if not isinstance(state, (int, np.integer)) or state not in range(len(self.states)):
            raise ValueError("Invalid stock.")
        return np.arange(min(state, self.M) + 1)

    def reward(self, state, action):
        if not isinstance(action, (int, np.integer)) or action not in self.actions(state):
            raise ValueError("Saving must be a feasible whole number of units.")
        return float((state - action)**self.alpha)

    def transition_probabilities(self, state, action):
        self.reward(state, action)  # Also check the saving constraint.
        p = np.zeros(len(self.states))
        p[action:action + self.B + 1] = self.q
        return p

    def step(self, state, action, rng):
        reward = self.reward(state, action)
        p = self.transition_probabilities(state, action)
        next_state = int(rng.choice(len(self.states), p=p))
        return next_state, reward

    def simulate(self, policy, initial_state, periods, seed=7):
        return simulate_policy(self, policy, initial_state, periods, seed)

savings = ConsumptionSavingModel()
print("Stocks:", savings.states)
print("Possible income:", savings.incomes)
print("Saving choices at stock 2:", savings.actions(2))
print("Next-state probabilities after saving 1:", savings.transition_probabilities(2, 1))


def save_half(state):
    return min(state // 2, savings.M)  # // rounds down to whole units.


saving_start = 5
saving_path, saving_rewards = savings.simulate(save_half, saving_start, periods=24, seed=7)
saving_actions = np.array([save_half(s) for s in saving_path[:-1]])
consumption = saving_path[:-1] - saving_actions
income = saving_path[1:] - saving_actions
print("First five stocks:", saving_path[:5])
print("First five consumption choices:", consumption[:5])
print("First five incomes:", income[:5])
print("24-step return:", discounted_return(saving_rewards, savings.beta))
assert np.allclose(saving_rewards, consumption**savings.alpha)
assert np.all((0 <= income) & (income <= savings.B))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].step(range(len(saving_path)), saving_path, where="post", label="Stock")
axes[0].set(xlabel="Step", ylabel="Units of goods", title="One rollout: save half")
axes[1].step(range(len(consumption)), consumption, where="mid", label="Consumption")
axes[1].step(range(len(income)), income, where="mid", label="Next-period income", alpha=0.7)
axes[1].set(xlabel="Step", ylabel="Units of goods")
for ax in axes:
    ax.legend()
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

V_saving, P_saving, r_saving = evaluate_policy(savings, save_half)
V_consume, _, _ = evaluate_policy(savings, lambda state: 0)
print("Value from stock 5 under save-half:", V_saving[saving_start])
print("Stock | saving | consumption | value")
for state in savings.states:
    action = save_half(state)
    print(f"{state:5d} | {action:6d} | {state - action:11d} | {V_saving[state]:.4f}")

fig, ax = plt.subplots()
ax.plot(savings.states, V_saving, "o-", label="Save half (rounded down)")
ax.plot(savings.states, V_consume, "s--", label="Consume everything")
ax.set(xlabel="Initial stock", ylabel="Expected discounted utility")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

saving_estimate = estimate_value(savings, save_half, saving_start, seed=22)
print("Exact infinite-horizon value:", V_saving[saving_start])
print("Simulated mean:", saving_estimate["mean"])
print("Monte Carlo standard error:", saving_estimate["standard_error"])
print("Maximum omitted tail:", saving_estimate["tail_bound"])

for label, model, V, P, r in [
    ("McCall", mccall, V_search, P_search, r_search),
    ("Consumption-saving", savings, V_saving, P_saving, r_saving),
]:
    residual = np.max(np.abs(V - r - model.beta * (P @ V)))
    print(label, "maximum policy Bellman residual:", residual)
    assert residual < 1e-10

# Independent check: consume everything, so next stock equals the new income.
mean_future_utility = savings.q @ (savings.incomes**savings.alpha)
consume_benchmark = (savings.states**savings.alpha
                     + savings.beta * mean_future_utility / (1 - savings.beta))
assert np.allclose(V_consume, consume_benchmark)
print("Analytical benchmarks passed.")

def compare_thresholds(model, thresholds):
    # For each threshold: construct a policy, evaluate it, and keep the U-state values.
    pass

# Run after completing the function:
# results = compare_thresholds(mccall, [2, 4])
# for threshold, values in results.items():
#     print(threshold, "value from offer 2:", values[1])
#     plt.plot(mccall.wages, values, "o-", label=f"Threshold {threshold}")
# plt.xlabel("Wage offer")
# plt.ylabel("Expected discounted income")
# plt.legend()
# plt.show()

#@title Solution 1: Compare McCall policies (Show code)
#@markdown Select **Show code** to reveal the answer; call `mccall_solution()` to run it.
def mccall_solution():
    model = McCallModel()
    results = {}
    for threshold in [2, 4]:
        policy = make_threshold_policy(model, threshold)
        V, _, _ = evaluate_policy(model, policy)
        results[threshold] = V[:model.n]
        print(threshold, "value from offer 2:", V[1])
        plt.plot(model.wages, V[:model.n], "o-", label=f"Threshold {threshold}")
    plt.xlabel("Wage offer")
    plt.ylabel("Expected discounted income")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results

def evaluate_saving_target(model, target):
    # Define the saving policy, evaluate it, and return V.
    pass

# Run after completing the function:
# for target in [1, 3]:
#     values = evaluate_saving_target(savings, target)
#     print(target, "value from stock 5:", values[5])
#     plt.plot(savings.states, values, "o-", label=f"Saving target {target}")
# plt.xlabel("Initial stock")
# plt.ylabel("Expected discounted utility")
# plt.legend()
# plt.show()

#@title Solution 2: Compare saving targets (Show code)
#@markdown Select **Show code** to reveal the answer; call `savings_solution()` to run it.
def savings_solution():
    model = ConsumptionSavingModel()
    results = {}
    for target in [1, 3]:
        def policy(state):
            return min(state, target, model.M)
        V, _, _ = evaluate_policy(model, policy)
        results[target] = V
        print(target, "value from stock 5:", V[5])
        plt.plot(model.states, V, "o-", label=f"Saving target {target}")
    plt.xlabel("Initial stock")
    plt.ylabel("Expected discounted utility")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results
