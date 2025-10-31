import numpy as np
from typing import Tuple, Optional
from mdp import MDP

class VFI:
    """
    Value Function Iteration (time-homogeneous, infinite horizon).
    Synchronous (Jacobi-style) backups.  Computes the greedy policy after the value function has converged.
    """
    def __init__(self, mdp: MDP, tol: float = 1e-6, max_iter: int = 10000):
        self.mdp = mdp
        self.tol = tol
        self.max_iter = max_iter

    def f(x):
        return x
    
    

    def bellman_value_update(self, V: np.ndarray) -> np.ndarray:
        """One Bellman optimality update on V; no policy returned here."""
        S, A = self.mdp.S, self.mdp.A
        V_new = np.empty_like(V)
        for s in range(S):
            # Q(s,a) = r(s,a) + beta * E[V(S') | s,a]
            Q_sa = [self.mdp.R[s, a] + self.mdp.beta * (self.mdp.P[s, a, :] @ V) for a in range(A)]
            V_new[s] = np.max(Q_sa)
        return V_new

    def greedy_policy(self, V: np.ndarray) -> np.ndarray:
        """Greedy policy with respect to V: pi(s) ∈ argmax_a Q_V(s,a)."""
        S, A = self.mdp.S, self.mdp.A
        pi = np.zeros(S, dtype=int)
        for s in range(S):
            Q_sa = [self.mdp.R[s, a] + self.mdp.beta * (self.mdp.P[s, a, :] @ V) for a in range(A)]
            pi[s] = int(np.argmax(Q_sa))
        return pi

    def solve(self, V0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """Returns (V*, pi*, number_of_iterations)."""
        S = self.mdp.S
        V = np.zeros(S) if V0 is None else V0.copy()

        for it in range(1, self.max_iter + 1):
            V_next = self.bellman_value_update(V)
            if np.max(np.abs(V_next - V)) <= self.tol:
                pi_star = self.greedy_policy(V_next)  # compute policy only once, after convergence
                return V_next, pi_star, it
            V = V_next

        # If not converged within max_iter, still return greedy policy of last iterate
        return V, self.greedy_policy(V), self.max_iter

class PFI:
    """
    Policy Function Iteration (Howard). Gauss–Seidel (in-place) policy evaluation.
    """
    def __init__(self, mdp: MDP, eval_tol: float = 1e-8, max_eval_iter: int = 100000, max_outer: int = 1000):
        self.mdp = mdp
        self.eval_tol = eval_tol
        self.max_eval_iter = max_eval_iter
        self.max_outer = max_outer

    def policy_evaluation_inplace(self, pi: np.ndarray, V0: Optional[np.ndarray] = None) -> np.ndarray:
        """Gauss–Seidel policy evaluation: V <- T^pi V until convergence."""
        S = self.mdp.S
        V = np.zeros(S) if V0 is None else V0.copy()

        for sweep in range(1, self.max_eval_iter + 1):
            delta = 0.0
            for s in range(S):
                a = int(pi[s])
                v_old = V[s]
                V[s] = self.mdp.R[s, a] + self.mdp.beta * (self.mdp.P[s, a, :] @ V)
                delta = max(delta, abs(V[s] - v_old))
            if delta <= self.eval_tol:
                return V
        return V

    def greedy_improvement(self, V: np.ndarray) -> np.ndarray:
        """pi'(s) ∈ argmax_a [ r(s,a) + beta E[V(S')|s,a] ] over all actions."""
        S, A = self.mdp.S, self.mdp.A
        pi_new = np.zeros(S, dtype=int)
        for s in range(S):
            Q_sa = [self.mdp.R[s, a] + self.mdp.beta * (self.mdp.P[s, a, :] @ V) for a in range(A)]
            pi_new[s] = int(np.argmax(Q_sa))
        return pi_new

    def solve(self, pi0: Optional[np.ndarray] = None, V0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (V*, pi*)."""
        if pi0 is None:
            # Initialize with greedy policy w.r.t. zero value function
            V_init = np.zeros(self.mdp.S)
            pi0 = self.greedy_improvement(V_init)

        pi = pi0.copy()
        V = np.zeros(self.mdp.S) if V0 is None else V0.copy()

        for _ in range(self.max_outer):
            # Policy evaluation
            V = self.policy_evaluation_inplace(pi, V0=V)
            # Policy improvement
            pi_new = self.greedy_improvement(V)
            if np.array_equal(pi_new, pi):
                return V, pi
            pi = pi_new

        return V, pi  # reached outer-iteration cap

class HowardPFI:
    """
    Howard's Policy Function Iteration with fixed number of policy evaluation steps.
    Performs exactly m policy evaluation iterations before policy improvement.
    """
    def __init__(self, mdp: MDP, m: int = 1, max_outer: int = 1000):
        self.mdp = mdp
        self.m = m  # number of policy evaluation iterations
        self.max_outer = max_outer

    def policy_evaluation_m_steps(self, pi: np.ndarray, V0: Optional[np.ndarray] = None) -> np.ndarray:
        """Gauss–Seidel policy evaluation: exactly m iterations of V <- T^pi V."""
        S = self.mdp.S
        V = np.zeros(S) if V0 is None else V0.copy()

        for _ in range(self.m):
            for s in range(S):
                a = int(pi[s])
                V[s] = self.mdp.R[s, a] + self.mdp.beta * (self.mdp.P[s, a, :] @ V)
        return V

    def greedy_improvement(self, V: np.ndarray) -> np.ndarray:
        """pi'(s) ∈ argmax_a [ r(s,a) + beta E[V(S')|s,a] ] over all actions."""
        S, A = self.mdp.S, self.mdp.A
        pi_new = np.zeros(S, dtype=int)
        for s in range(S):
            Q_sa = [self.mdp.R[s, a] + self.mdp.beta * (self.mdp.P[s, a, :] @ V) for a in range(A)]
            pi_new[s] = int(np.argmax(Q_sa))
        return pi_new

    def solve(self, pi0: Optional[np.ndarray] = None, V0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (V*, pi*)."""
        if pi0 is None:
            # Initialize with greedy policy w.r.t. zero value function
            V_init = np.zeros(self.mdp.S)
            pi0 = self.greedy_improvement(V_init)

        pi = pi0.copy()
        V = np.zeros(self.mdp.S) if V0 is None else V0.copy()

        for _ in range(self.max_outer):
            # Policy evaluation (exactly m steps)
            V = self.policy_evaluation_m_steps(pi, V0=V)
            # Policy improvement
            pi_new = self.greedy_improvement(V)
            if np.array_equal(pi_new, pi):
                return V, pi
            pi = pi_new

        return V, pi  # reached outer-iteration cap