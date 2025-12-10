"""
Algorithms for solving consumption-saving models.

Contains:
- VFI: Value Function Iteration (Dynamic Programming)
- TDEvaluator: TD(λ) Policy Evaluation - evaluates a fixed policy
"""

import numpy as np
from typing import Tuple, Optional, List
from model import ConsumptionSavingModel


class VFI:
    """
    Value Function Iteration (Dynamic Programming).
    
    Computes exact solution using the Bellman equation:
        V(s) = max_a { r(s,a) + β * Σ_s' P(s'|s,a) * V(s') }
    """
    
    def __init__(self, model: ConsumptionSavingModel, tol: float = 1e-6, max_iter: int = 10000):
        self.model = model
        self.tol = tol
        self.max_iter = max_iter
    
    def solve(self, V0: Optional[np.ndarray] = None, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Solve for optimal value function and policy."""
        m = self.model
        n_states, n_actions = m.n_states, m.n_actions
        
        V = np.zeros(n_states) if V0 is None else V0.copy()
        errors = []
        
        # Precompute Q-values matrix for vectorized update
        # Q[s, a] = r[s,a] + β * Σ_s' P[s,a,s'] * V[s']
        for it in range(self.max_iter):
            # Vectorized Bellman update
            Q = m.rewards + m.beta * np.einsum('saj,j->sa', m.transitions, V)
            Q[~m.feasible] = -np.inf
            
            V_new = np.max(Q, axis=1)
            error = np.max(np.abs(V_new - V))
            errors.append(error)
            
            if verbose and (it + 1) % 100 == 0:
                print(f"VFI Iteration {it+1}: error = {error:.2e}")
            
            if error < self.tol:
                if verbose:
                    print(f"VFI converged after {it+1} iterations")
                policy = np.argmax(Q, axis=1)
                return V_new, policy, errors
            
            V = V_new
        
        if verbose:
            print("VFI: max iterations reached")
        policy = np.argmax(Q, axis=1)
        return V, policy, errors


class TDEvaluator:
    """
    TD(λ) Policy Evaluation - Evaluates a fixed policy via simulation.
    Finds V(s) given policy pi.
    
    Optimizations:
    - Vectorized operations where possible
    - Trace decay
    """
    
    def __init__(
        self,
        model: ConsumptionSavingModel,
        lambd: float = 0.0,
        alpha: float = 0.1,
        n_episodes: int = 5000,
        episode_length: int = 100,
    ):
        self.model = model
        self.lambd = lambd
        self.alpha_0 = alpha
        self.n_episodes = n_episodes
        self.episode_length = episode_length
    
    def evaluate(
        self,
        policy: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = True,
        track_every: int = 100,
    ) -> Tuple[np.ndarray, dict]:
        """Evaluate the given policy using TD(λ)."""
        if rng is None:
            rng = np.random.default_rng(42)
        
        m = self.model
        n_states = m.n_states
        
        # Initialize V (Value Function)
        V = np.zeros(n_states)
        
        visit_counts = np.zeros(n_states, dtype=int)
        V_history = []
        
        alpha = self.alpha_0
        
        for ep in range(self.n_episodes):
            # We could decay alpha per episode, but let's stick to per-state decay or constant
            # For simplicity/robustness in eval, let's use the per-visit decay strategy inside episodes
            
            if self.lambd == 1.0:
                self._mc_episode(V, visit_counts, policy, alpha, rng)
            elif self.lambd == 0.0:
                self._td0_episode(V, visit_counts, policy, alpha, rng)
            else:
                self._td_lambda_episode(V, visit_counts, policy, alpha, rng)
            
            if (ep + 1) % track_every == 0:
                V_history.append(V.copy())
                if verbose and (ep + 1) % 1000 == 0:
                    print(f"TD(λ={self.lambd}) Episode {ep+1}")
        
        return V, {'visit_counts': visit_counts, 'V_history': V_history}
    
    def _td0_episode(self, V: np.ndarray, visit_counts: np.ndarray, 
                     policy: np.ndarray, alpha: float, rng: np.random.Generator):
        """TD(0) prediction."""
        m = self.model
        s = rng.integers(0, m.n_states)
        
        for _ in range(self.episode_length):
            a = policy[s]
            
            s_next, r = m.sample_next_state(s, a, rng)
            
            visit_counts[s] += 1
            lr = alpha / (1 + 0.01 * visit_counts[s])
            
            target = r + m.beta * V[s_next]
            V[s] += lr * (target - V[s])
            
            s = s_next
    
    def _mc_episode(self, V: np.ndarray, visit_counts: np.ndarray,
                    policy: np.ndarray, alpha: float, rng: np.random.Generator):
        """Monte Carlo prediction."""
        m = self.model
        s = rng.integers(0, m.n_states)
        
        trajectory = []
        for _ in range(self.episode_length):
            a = policy[s]
            s_next, r = m.sample_next_state(s, a, rng)
            trajectory.append((s, r))
            s = s_next
        
        # Compute returns backward
        G = 0.0
        visited = set()
        for s, r in reversed(trajectory):
            G = r + m.beta * G
            if s not in visited:
                visited.add(s)
                visit_counts[s] += 1
                lr = alpha / (1 + 0.01 * visit_counts[s])
                V[s] += lr * (G - V[s])
    
    def _td_lambda_episode(self, V: np.ndarray, visit_counts: np.ndarray,
                           policy: np.ndarray, alpha: float, rng: np.random.Generator):
        """TD(λ) prediction with replacing traces."""
        m = self.model
        n_states = m.n_states
        
        s = rng.integers(0, n_states)
        e = np.zeros(n_states)
        
        for _ in range(self.episode_length):
            a = policy[s]
            s_next, r = m.sample_next_state(s, a, rng)
            
            delta = r + m.beta * V[s_next] - V[s]
            
            e[s] = 1.0  # Replacing traces
            
            # Vectorized update with per-state learning rate (approx)
            # or just use uniform lr for the trace update?
            # Using per-state LR is more consistent with the other methods.
            
            # Note: We need to update all states where e > 0
            # Ideally: V += alpha_vec * delta * e
            # where alpha_vec[x] = alpha / (1 + 0.01 * visit_counts[x])
            
            # But constructing alpha_vec every step is expensive.
            # Let's approximate by using alpha for current state or just global alpha?
            # The original code did: lr = alpha / (1 + ... visit_counts[s,a])
            # and Q += lr * delta * e
            # This applies `lr` of the CURRENT state-action to the whole trace.
            # Let's align with that logic:
            
            visit_counts[s] += 1
            lr = alpha / (1 + 0.01 * visit_counts[s])
            
            V += lr * delta * e
            
            e *= m.beta * self.lambd
            s = s_next


def compare_vfi_td(
    model: ConsumptionSavingModel,
    lambdas: List[float] = [0.0, 0.5, 0.9, 1.0],
    n_runs: int = 5,
    n_episodes: int = 5000,
    alpha: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Compare VFI (Optimal Policy) with TD(λ) Policy Evaluation."""
    
    if verbose:
        print("=== Solving with VFI (Optimal Policy) ===")
    vfi = VFI(model)
    V_vfi, policy_vfi, errors_vfi = vfi.solve(verbose=verbose)
    
    results = {
        'V_vfi': V_vfi,
        'policy_vfi': policy_vfi,
        'errors_vfi': errors_vfi,
        'td_results': {},
    }
    
    for lambd in lambdas:
        if verbose:
            print(f"\n=== Evaluating with TD(λ={lambd}) ===")
        
        V_runs = []
        
        for run in range(n_runs):
            evaluator = TDEvaluator(
                model,
                lambd=lambd,
                alpha=alpha,
                n_episodes=n_episodes,
                episode_length=100,
            )
            V_td, _ = evaluator.evaluate(policy_vfi, rng=np.random.default_rng(run * 100), verbose=False)
            V_runs.append(V_td)
        
        V_runs = np.array(V_runs)
        V_mean = V_runs.mean(axis=0)
        V_std = V_runs.std(axis=0)
        
        mse = np.mean((V_mean - V_vfi) ** 2)
        max_error = np.max(np.abs(V_mean - V_vfi))
        
        results['td_results'][lambd] = {
            'V_mean': V_mean,
            'V_std': V_std,
            'V_runs': V_runs,
            'mse': mse,
            'max_error': max_error,
            'avg_variance': np.mean(V_std ** 2),
        }
        
        if verbose:
            print(f"  MSE vs VFI: {mse:.6f}")
            print(f"  Max error: {max_error:.4f}")
            print(f"  Avg Variance: {results['td_results'][lambd]['avg_variance']:.6f}")
    
    return results
