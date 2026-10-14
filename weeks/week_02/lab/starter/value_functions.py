"""Companion to value_functions.ipynb; code cells in notebook order.
Evaluate specified policies in McCall job search and continuous consumption-saving.
Call mccall_solution() or savings_solution() separately to run exercise solutions.
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)  # Shorter printed numbers.
plt.rcParams["figure.figsize"] = (8, 3.5)

def simulate_policy(model, policy, initial_state, periods, seed=7):
    rng = np.random.default_rng(seed)  # Same seed gives the same random draws.
    state = initial_state
    path = [state]
    rewards = []

    for t in range(periods):
        action = policy(state)
        state, reward = model.step(state, action, rng)
        path.append(state)       # append adds one item to a list.
        rewards.append(reward)

    return np.array(path), np.array(rewards)


def discounted_return(rewards, beta):
    # For H rewards: weights = [1, beta, beta**2, ..., beta**(H-1)].
    weights = beta ** np.arange(len(rewards))
    return np.sum(weights * rewards)

class McCallModel:
    def __init__(self, wages=(1, 2, 3, 4, 5),
                 probabilities=(0.1, 0.2, 0.4, 0.2, 0.1), benefit=1.0, beta=0.9):
        self.wages = np.array(wages)
        self.q = np.array(probabilities)
        self.benefit = benefit
        self.beta = beta
        self.n = len(wages)  # Number of possible wage offers.

        # assert stops the cell if a check fails.
        # Each wage needs one probability; probabilities must sum to one.
        assert len(self.q) == self.n, "One probability per wage."
        assert np.all(self.q >= 0) and np.isclose(self.q.sum(), 1), "Check probabilities."
        assert 0 < beta < 1, "Use a discount factor between 0 and 1."

        # Join two lists: unemployed states first, then employed states.
        self.states = [("U", w) for w in wages] + [("E", w) for w in wages]

    def actions(self, state):
        if state < self.n:
            return ["accept", "reject"]
        return ["work"]

    def reward(self, state, action):
        assert action in self.actions(state), "Choose an available action."
        status, wage = self.states[state]  # Unpack the full state.
        if action == "reject":
            return self.benefit
        return wage

    def transition_probabilities(self, state, action):
        p = np.zeros(len(self.states))
        if action == "reject":
            p[:self.n] = self.q       # A new offer: first n states only.
        elif action == "accept":
            p[state + self.n] = 1    # Move to employment at the same wage.
        else:
            p[state] = 1             # Already employed: stay here.
        return p

    def step(self, state, action, rng):
        reward = self.reward(state, action)
        p = self.transition_probabilities(state, action)
        next_state = rng.choice(len(self.states), p=p)  # Sample using probabilities p.
        return next_state, reward

    def simulate(self, policy, initial_state, periods, seed=7):
        return simulate_policy(self, policy, initial_state, periods, seed)

mccall = McCallModel()
for i, state in enumerate(mccall.states):  # enumerate gives the index and its item.
    print(i, state)

# Index 1 means (U, 2).
print("Reject:", mccall.transition_probabilities(1, "reject"))
print("Accept:", mccall.transition_probabilities(1, "accept"))

def make_threshold_policy(model, threshold):
    def policy(state):
        status, wage = model.states[state]
        if status == "E":
            return "work"
        if wage >= threshold:
            return "accept"
        return "reject"
    return policy  # Return the function itself; call it later with a state.


search_policy = make_threshold_policy(mccall, threshold=3)
search_start = 1  # (U, 2)
search_path, search_rewards = mccall.simulate(search_policy, search_start, periods=12)
print("States:", [mccall.states[i] for i in search_path])
print("Rewards:", search_rewards)
print("Return:", discounted_return(search_rewards, mccall.beta))

def policy_arrays(model, policy):
    n = len(model.states)
    P = np.zeros((n, n))  # One row for each current state.
    mu = np.zeros(n)
    for state in range(n):
        action = policy(state)
        mu[state] = model.reward(state, action)
        P[state] = model.transition_probabilities(state, action)
    return P, mu


def evaluate_policy(model, policy):
    P, mu = policy_arrays(model, policy)
    I = np.eye(len(mu))   # Identity matrix: ones on the diagonal.
    V = np.linalg.solve(I - model.beta * P, mu)
    return V, P, mu       # solve(A, b) finds the vector x satisfying A @ x = b.


V_search, P_search, mu_search = evaluate_policy(mccall, search_policy)
for i in range(mccall.n):
    print(mccall.states[i], "value:", round(V_search[i], 4))

# A known value: permanent employment pays w every period.
assert np.allclose(V_search[mccall.n:], mccall.wages / (1 - mccall.beta))

# [:mccall.n] selects the unemployed-state values.
fig, ax = plt.subplots()
ax.plot(mccall.wages, V_search[:mccall.n], "o-", label="Supplied threshold = 3")
ax.plot(mccall.wages, mccall.wages / (1 - mccall.beta), "s--", label="Accept every offer")
ax.set(xlabel="Current wage offer", ylabel="Expected discounted income")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

def simulate_returns(model, policy, initial_state, repetitions=600, periods=100):
    returns = []
    for seed in range(repetitions):  # Use a different seed for each rollout.
        path, rewards = model.simulate(policy, initial_state, periods, seed=seed)
        returns.append(discounted_return(rewards, model.beta))
    return np.array(returns)


search_returns = simulate_returns(mccall, search_policy, search_start)
print("Exact value:", V_search[search_start])
print("Average simulated return:", search_returns.mean())

class ConsumptionSavingModel:
    def __init__(self, M=5.0, B=3.0, alpha=0.5, beta=0.9):
        assert M >= 0 and B > 0, "Use M >= 0 and B > 0."
        assert 0 < alpha < 1 and 0 < beta < 1, "Keep alpha and beta between 0 and 1."
        self.M = M
        self.B = B
        self.alpha = alpha
        self.beta = beta

    def reward(self, state, action):
        assert 0 <= state <= self.M + self.B, "Stock must lie in [0, M+B]."
        assert 0 <= action <= min(state, self.M), "Saving must be feasible."
        consumption = state - action
        return consumption ** self.alpha

    def step(self, state, action, rng):
        reward = self.reward(state, action)
        income = rng.uniform(0, self.B)  # Draw a real number from Uniform[0, B].
        next_state = action + income
        return next_state, reward

    def simulate(self, policy, initial_state, periods, seed=7):
        return simulate_policy(self, policy, initial_state, periods, seed)

savings = ConsumptionSavingModel()


def save_half(state):
    return min(state / 2, savings.M)  # / keeps fractional units; min applies the cap.


saving_start = 5.5
saving_path, saving_rewards = savings.simulate(save_half, saving_start, periods=24)

# Drop the last state: a path has one more state than rewards.
saving_actions = np.array([save_half(s) for s in saving_path[:-1]])
consumption = saving_path[:-1] - saving_actions
income = saving_path[1:] - saving_actions  # Next stock minus saving is new income.
print("First five stocks:", saving_path[:5])
print("First five consumption choices:", consumption[:5])
print("Return:", discounted_return(saving_rewards, savings.beta))

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

def estimate_saving_values(model, policy, initial_stocks):
    values = []
    for stock in initial_stocks:
        returns = simulate_returns(model, policy, stock)
        values.append(returns.mean())  # Average rollouts from this initial stock.
    return np.array(values)


def consume_all(state):
    return 0.0  # Save nothing.


initial_stocks = np.array([0.0, 1.5, 3.0, 5.5, 8.0])
V_saving = estimate_saving_values(savings, save_half, initial_stocks)
V_consume = estimate_saving_values(savings, consume_all, initial_stocks)
for stock, value in zip(initial_stocks, V_saving):  # Pair each stock with its estimate.
    print(f"Initial stock {stock:.1f}: estimated value {value:.4f}")

fig, ax = plt.subplots()
ax.plot(initial_stocks, V_saving, "o-", label="Save half")
ax.plot(initial_stocks, V_consume, "s--", label="Consume everything")
ax.set(xlabel="Initial stock", ylabel="Estimated expected discounted utility")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Check against a known value: consume everything each period.
# E[Y**alpha] = B**alpha / (alpha + 1) for uniform income.
mean_income_utility = savings.B ** savings.alpha / (savings.alpha + 1)
V_consume_exact = initial_stocks ** savings.alpha + (
    savings.beta / (1 - savings.beta) * mean_income_utility
)
print("Consume-all estimates:", V_consume)
print("Consume-all exact values:", V_consume_exact)

def compare_thresholds(model, thresholds):
    results = {}  # A dictionary stores pairs: threshold -> value array.
    # Loop over thresholds; use make_threshold_policy and evaluate_policy.
    # Store V[:model.n] in results[threshold], then return results.
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

def evaluate_saving_target(model, target, initial_stocks):
    # Define the saving policy, estimate its values, and return the array.
    pass

# Run after completing the function:
# for target in [1.5, 3.5]:
#     values = evaluate_saving_target(savings, target, initial_stocks)
#     print(target, "estimated value from stock 5.5:", values[3])
#     plt.plot(initial_stocks, values, "o-", label=f"Saving target {target}")
# plt.xlabel("Initial stock")
# plt.ylabel("Estimated expected discounted utility")
# plt.legend()
# plt.show()

#@title Solution 2: Compare saving targets (Show code)
#@markdown Select **Show code** to reveal the answer; call `savings_solution()` to run it.
def savings_solution():
    model = ConsumptionSavingModel()
    stocks = np.array([0.0, 1.5, 3.0, 5.5, 8.0])
    results = {}
    for target in [1.5, 3.5]:
        def policy(state):
            return min(state, target, model.M)
        values = estimate_saving_values(model, policy, stocks)
        results[target] = values
        print(target, "estimated value from stock 5.5:", values[3])
        plt.plot(stocks, values, "o-", label=f"Saving target {target}")
    plt.xlabel("Initial stock")
    plt.ylabel("Estimated expected discounted utility")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results
