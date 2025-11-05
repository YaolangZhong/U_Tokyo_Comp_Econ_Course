import numpy as np
from typing import Tuple, Optional
from model import ConsumptionSavingModel

from scipy.optimize import brentq

class TISolver:
    """Time Iteration solver for the consumption-saving model."""
    
    def __init__(self, model: ConsumptionSavingModel, max_iter: int = 1000, tol: float = 1e-6):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        
    def solve(self, damping: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the model using Time Iteration.
        
        Parameters:
        -----------
        damping : float
            Damping parameter for policy updates (default: 1.0, no damping)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (consumption policy, next-period capital)
        """
        # Initialize with borrowing cap
        c = self.model.pi_cap.copy()
        k_next = np.zeros_like(c)
        
        # Pre-allocate arrays for efficiency
        c_next_all = np.zeros(self.model.N)
        rhs_vec = np.zeros(self.model.M * self.model.N)
        
        for it in range(self.max_iter):
            c_old = c.copy()
            
            # Compute next period k grid for all current states
            k_next = self.model.R * self.model.k_grid[:, None] + self.model.w_grid[None, :] - c
            
            # Update at each grid point
            for m in range(self.model.M):
                for n in range(self.model.N):
                    # Current state
                    k = self.model.k_grid[m]
                    w = self.model.w_grid[n]
                    k_next_val = k_next[m, n]
                    
                    def residual(c_try):
                        k_next = self.model.next_capital(k, w, c_try)
                        
                        # Vectorized RHS computation
                        for n_next in range(self.model.N):
                            c_next_all[n_next] = self.model.interpolate_policy(
                                k_next, self.model.w_grid[n_next],
                                self.model.k_grid, c_old, n_next
                            )
                        
                        rhs = np.sum(self.model.P[n] * self.model.marginal_utility(c_next_all))
                        rhs *= self.model.beta * self.model.R
                        
                        return self.model.marginal_utility(c_try) - rhs
                    
                    # Find root of residual equation with constraints
                    try:
                        c_new = brentq(residual, 1e-6, self.model.pi_cap[m, n], 
                                     xtol=self.tol, rtol=self.tol)
                    except ValueError:
                        c_new = self.model.pi_cap[m, n]
                    
                    # Update policy (no damping for TI)
                    c[m, n] = c_new
                    k_next[m, n] = self.model.next_capital(k, w, c[m, n])
            
            # Check convergence
            max_diff = np.max(np.abs(c - c_old))
            if max_diff < self.tol:
                print(f"TI converged after {it+1} iterations with max diff {max_diff:.2e}")
                break
        
        return c, k_next

class EGMSolver:
    """Endogenous Grid Method solver for the consumption-saving model."""
    
    def __init__(self, model: ConsumptionSavingModel, max_iter: int = 1000, tol: float = 1e-6):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        
    def solve(self, damping: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the model using EGM.
        
        Parameters:
        -----------
        damping : float
            Damping parameter for policy updates (default: 1.0, no damping)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (consumption policy, next-period capital)
        """
        # Initialize with borrowing cap
        c = self.model.pi_cap.copy()
        k_next = np.zeros_like(c)
        
        for it in range(self.max_iter):
            c_old = c.copy()
            
            # Backward step: compute consumption on the endogenous grid
            c_tilde = np.zeros((self.model.M, self.model.N))
            k_tilde = np.zeros_like(c_tilde)
            
            for n in range(self.model.N):
                for m in range(self.model.M):
                    # Future capital is the grid point
                    k_next = self.model.k_grid[m]
                    
                    # Compute RHS of Euler equation
                    rhs = 0
                    for n_next in range(self.model.N):
                        c_next = self.model.interpolate_policy(
                            k_next, self.model.w_grid[n_next],
                            self.model.k_grid, c_old, n_next
                        )
                        rhs += self.model.P[n, n_next] * self.model.marginal_utility(c_next)
                    rhs *= self.model.beta * self.model.R
                    
                    # Compute consumption and implied current capital
                    c_tilde[m, n] = self.model.inverse_marginal_utility(rhs)
                    k_tilde[m, n] = (k_next + c_tilde[m, n] - self.model.w_grid[n]) / self.model.R
            
            # Forward step: interpolate back to original grid
            for n in range(self.model.N):
                for m in range(self.model.M):
                    k = self.model.k_grid[m]
                    
                    # Handle boundary cases
                    if k <= k_tilde[0, n]:
                        c[m, n] = self.model.pi_cap[m, n]
                    elif k >= k_tilde[-1, n]:
                        # Linear extrapolation from last two points
                        slope = ((c_tilde[-1, n] - c_tilde[-2, n]) / 
                               (k_tilde[-1, n] - k_tilde[-2, n]))
                        c[m, n] = c_tilde[-1, n] + slope * (k - k_tilde[-1, n])
                        c[m, n] = min(c[m, n], self.model.pi_cap[m, n])
                    else:
                        # Linear interpolation
                        idx = np.searchsorted(k_tilde[:, n], k)
                        lambda_ = ((k - k_tilde[idx-1, n]) / 
                                 (k_tilde[idx, n] - k_tilde[idx-1, n]))
                        c[m, n] = ((1 - lambda_) * c_tilde[idx-1, n] + 
                                 lambda_ * c_tilde[idx, n])
                        c[m, n] = min(c[m, n], self.model.pi_cap[m, n])
                    
                    # Update next period capital
                    k_next[m, n] = self.model.next_capital(k, self.model.w_grid[n], c[m, n])
            
            # Check convergence
            if np.max(np.abs(c - c_old)) < self.tol:
                break
        
        return c, k_next

class ForwardRolloutSolver:
    """Forward rollout (lagged-Coleman explicit update) solver for the consumption-saving model."""
    
    def __init__(self, model: ConsumptionSavingModel, max_iter: int = 1000, tol: float = 1e-6):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        
    def solve(self, damping: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the model using forward rollout.
        
        Parameters:
        -----------
        damping : float
            Damping parameter for policy updates (default: 1.0, no damping)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (consumption policy, next-period capital)
        """
        # Initialize with borrowing cap
        c = self.model.pi_cap.copy()
        k_next = np.zeros_like(c)
        
        for it in range(self.max_iter):
            c_old = c.copy()
            
            # Forward step using old policy
            for m in range(self.model.M):
                for n in range(self.model.N):
                    k = self.model.k_grid[m]
                    w = self.model.w_grid[n]
                    k_next[m, n] = self.model.next_capital(k, w, c_old[m, n])
            
            # Update consumption policy
            for m in range(self.model.M):
                for n in range(self.model.N):
                    # Compute RHS of Euler equation
                    rhs = 0
                    for n_next in range(self.model.N):
                        c_next = self.model.interpolate_policy(
                            k_next[m, n], self.model.w_grid[n_next],
                            self.model.k_grid, c_old, n_next
                        )
                        rhs += self.model.P[n, n_next] * self.model.marginal_utility(c_next)
                    rhs *= self.model.beta * self.model.R
                    
                    # Update consumption with constraint
                    c_new = self.model.inverse_marginal_utility(rhs)
                    c_new = min(c_new, self.model.pi_cap[m, n])
                    
                    # Apply damping
                    c[m, n] = (1 - damping) * c_old[m, n] + damping * c_new
            
            # Update next period capital with new consumption
            for m in range(self.model.M):
                for n in range(self.model.N):
                    k_next[m, n] = self.model.next_capital(
                        self.model.k_grid[m], 
                        self.model.w_grid[n], 
                        c[m, n]
                    )
            
            # Check convergence
            if np.max(np.abs(c - c_old)) < self.tol:
                break
        
        return c, k_next