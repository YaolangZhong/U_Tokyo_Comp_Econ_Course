"""Value function iteration shared by all three models."""

import numpy as np


def value_iteration(model, tolerance=1e-6, max_iter=2000):
    """Return v_k, its greedy policy, and a record of the Bellman updates.

    model.bellman(v) returns (T v, policy greedy with respect to v).
    For the continuous model, T is a numerical approximation.
    """
    assert 0 < model.beta < 1 and tolerance > 0
    v = model.initial_values()
    history = [v.copy()]
    residuals = []

    for k in range(max_iter):
        new_v, policy = model.bellman(v)  # All states use the same old v.
        residual = np.max(np.abs(new_v - v))
        residuals.append(residual)

        if residual <= (1 - model.beta) * tolerance:
            # Return the OLD v: this residual and policy were computed from it.
            return {"values": v, "policy": policy, "history": np.array(history),
                    "residuals": np.array(residuals), "updates": k,
                    "error_bound": residual / (1 - model.beta)}
        v = new_v
        history.append(v.copy())

    raise RuntimeError("VFI did not converge; increase max_iter.")


def simulate_policy(model, policy, initial_state, periods=30, seed=7):
    """A policy is a function: state -> action, as in Lab 2."""
    rng = np.random.default_rng(seed)
    state = initial_state
    states, rewards = [state], []
    for t in range(periods):
        action = policy(state)
        state, reward = model.step(state, action, rng)
        states.append(state)
        rewards.append(reward)
    return np.array(states), np.array(rewards)
