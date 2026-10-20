"""One VFI implementation for any model with the MDP interface in mdp.py."""

from time import perf_counter
import numpy as np


def value_iteration(mdp, initial_V=None, tolerance=1e-6, max_iter=2000):
    """Return a value vector, its greedy policy, and convergence diagnostics.

    initial_V: optional vector, one entry per state; defaults to zeros.
    tolerance: target sup-norm value error for the supplied finite MDP.
    policy[i]: actual action at states[i], not just its column index.
    For a discretized continuous model, the bound excludes approximation error.
    """
    assert tolerance > 0 and max_iter >= 1
    started = perf_counter()
    n = len(mdp.states)

    # Step 1: initialize V_0. Copy a supplied vector so it is never modified.
    V = np.zeros(n) if initial_V is None else np.array(initial_V, dtype=float, copy=True)
    assert V.shape == (n,) and np.all(np.isfinite(V))
    history, residuals = [V.copy()], []

    for k in range(max_iter):
        # Step 2: Q_k for every state and action, always using the same old V.
        # P @ V sums over the final axis: the possible next states.
        Q = mdp.rewards + mdp.beta * (mdp.P @ V)
        Q = np.where(mdp.feasible, Q, -np.inf)

        # Step 3: a separate array for V_{k+1}.
        new_V = Q.max(axis=1)

        # Steps 4-5: Bellman residual and the value-error stopping criterion.
        residual = np.max(np.abs(new_V - V))
        residuals.append(residual)
        if residual <= (1 - mdp.beta) * tolerance:
            # Step 6: argmax chooses the first column when actions tie.
            policy_indices = Q.argmax(axis=1)
            policy = mdp.actions[np.arange(n), policy_indices]
            # Step 7: return V_k, the vector used to compute Q_k and this policy.
            return {
                "values": V, "policy": policy, "policy_indices": policy_indices,
                "history": np.array(history), "residuals": np.array(residuals),
                "updates": k, "iterations": k + 1,
                "error_bound": residual / (1 - mdp.beta),
                "seconds": perf_counter() - started,
            }

        V = new_V
        history.append(V.copy())

    raise RuntimeError("VFI did not converge; increase max_iter.")


def simulate_policy(model, policy, initial_state, periods=30, seed=7):
    """A policy function maps a state to an action, as in Lab 2."""
    rng = np.random.default_rng(seed)
    state = initial_state
    states, rewards = [state], []
    for _ in range(periods):
        action = policy(state)
        state, reward = model.step(state, action, rng)
        states.append(state)
        rewards.append(reward)
    return np.array(states), np.array(rewards)
