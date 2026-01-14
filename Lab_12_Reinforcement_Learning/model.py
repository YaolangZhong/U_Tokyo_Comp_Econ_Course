"""
Aiyagari Model Definition

This module defines the Aiyagari incomplete markets model with:
- Idiosyncratic income shocks (AR(1) process)
- Borrowing constraints
- CRRA preferences
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, NamedTuple


class AiyagariParams(NamedTuple):
    """Parameters for the Aiyagari model."""
    beta: float = 0.96        # Discount factor
    gamma: float = 2.0        # Risk aversion (CRRA)
    r: float = 0.03           # Interest rate
    w: float = 1.0            # Wage rate
    a_min: float = 0.0        # Borrowing constraint
    a_max: float = 20.0       # Maximum assets (for normalization)
    
    # Income process: AR(1) discretized using Tauchen
    rho: float = 0.9          # Persistence
    sigma: float = 0.2        # Std dev of innovations
    n_z: int = 5              # Number of income states


# Simplified parameters for testing
class SimpleParams(NamedTuple):
    """Simplified parameters for debugging."""
    beta: float = 0.95        # Discount factor
    gamma: float = 1.0        # Log utility (simpler)
    r: float = 0.04           # Interest rate
    w: float = 1.0            # Wage rate
    a_min: float = 0.0        # No borrowing
    a_max: float = 10.0       # Smaller range
    
    # Simpler shocks
    rho: float = 0.0          # IID shocks (no persistence)
    sigma: float = 0.1        # Small shocks
    n_z: int = 2              # Just 2 states


def tauchen(rho: float, sigma: float, n: int, m: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize AR(1) process using Tauchen method.
    
    z' = rho * z + epsilon, epsilon ~ N(0, sigma^2)
    
    Parameters
    ----------
    rho : float
        Persistence parameter
    sigma : float
        Standard deviation of innovations
    n : int
        Number of grid points
    m : float
        Number of standard deviations for grid bounds
    
    Returns
    -------
    z_grid : np.ndarray
        Grid of productivity states (in levels, not logs)
    P : np.ndarray
        Transition probability matrix
    """
    # Unconditional std dev
    sigma_z = sigma / np.sqrt(1 - rho**2)
    
    # Grid in log space
    z_max = m * sigma_z
    z_grid = np.linspace(-z_max, z_max, n)
    step = z_grid[1] - z_grid[0]
    
    # Transition matrix
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == 0:
                P[i, j] = norm.cdf((z_grid[j] + step/2 - rho * z_grid[i]) / sigma)
            elif j == n - 1:
                P[i, j] = 1 - norm.cdf((z_grid[j] - step/2 - rho * z_grid[i]) / sigma)
            else:
                P[i, j] = (norm.cdf((z_grid[j] + step/2 - rho * z_grid[i]) / sigma) -
                          norm.cdf((z_grid[j] - step/2 - rho * z_grid[i]) / sigma))
    
    # Convert log productivity to levels
    z_grid = np.exp(z_grid)
    
    return z_grid, P


class AiyagariEnv:
    """
    Aiyagari model environment for reinforcement learning.
    
    State: (a, z_idx) - assets and productivity index
    Action: a' (next period assets)
    Reward: u(c) instantaneous utility
    """
    
    def __init__(self, params: AiyagariParams, z_grid: np.ndarray, P_z: np.ndarray):
        """
        Initialize the Aiyagari environment.
        
        Parameters
        ----------
        params : AiyagariParams
            Model parameters
        z_grid : np.ndarray
            Productivity grid (in levels)
        P_z : np.ndarray
            Transition probability matrix for productivity
        """
        self.params = params
        self.z_grid = z_grid
        self.P_z = P_z
        self.n_z = len(z_grid)
    
    def utility(self, c: np.ndarray) -> np.ndarray:
        """CRRA utility function."""
        gamma = self.params.gamma
        c_safe = np.maximum(c, 1e-10)
        if gamma == 1:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma) - 1) / (1 - gamma)
    
    def marginal_utility(self, c: np.ndarray) -> np.ndarray:
        """Marginal utility: c^(-gamma)."""
        c_safe = np.maximum(c, 1e-10)
        return c_safe ** (-self.params.gamma)
    
    def budget(self, a: np.ndarray, z: np.ndarray, a_next: np.ndarray) -> np.ndarray:
        """Compute consumption from budget constraint."""
        p = self.params
        return (1 + p.r) * a + p.w * z - a_next
    
    def step(self, a: np.ndarray, z_idx: np.ndarray, a_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Take a step in the environment.
        
        Parameters
        ----------
        a : np.ndarray
            Current assets (batch)
        z_idx : np.ndarray
            Current productivity index (batch)
        a_next : np.ndarray
            Next period assets (batch)
            
        Returns
        -------
        z_idx_next : np.ndarray
            Next period productivity index
        reward : np.ndarray
            Utility from consumption
        c : np.ndarray
            Consumption
        """
        batch_size = len(a)
        z = self.z_grid[z_idx]
        
        # Consumption from budget constraint
        c = (1 + self.params.r) * a + self.params.w * z - a_next
        
        # Utility (reward)
        reward = self.utility(c)
        
        # Penalize negative consumption heavily
        reward = np.where(c > 0, reward, -1e6)
        
        # Transition to next productivity state
        z_idx_next = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            z_idx_next[i] = np.random.choice(self.n_z, p=self.P_z[z_idx[i]])
        
        return z_idx_next, reward, c
    
    def sample_states(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample random states for training."""
        a = np.random.uniform(self.params.a_min, self.params.a_max, n)
        z_idx = np.random.randint(0, self.n_z, n)
        return a, z_idx


class StateNormalizer:
    """Normalize state variables to [0, 1] range for neural network inputs."""
    
    def __init__(self, a_min: float, a_max: float, z_grid: np.ndarray):
        self.a_min = a_min
        self.a_max = a_max
        self.z_min = z_grid.min()
        self.z_max = z_grid.max()
        self.z_grid = z_grid
    
    def normalize(self, a: np.ndarray, z_idx: np.ndarray) -> np.ndarray:
        """Normalize (a, z) to [0, 1]^2."""
        a_norm = (a - self.a_min) / (self.a_max - self.a_min)
        z = self.z_grid[z_idx]
        z_norm = (z - self.z_min) / (self.z_max - self.z_min + 1e-8)
        return np.stack([a_norm, z_norm], axis=-1)
    
    def normalize_action(self, a_next: np.ndarray) -> np.ndarray:
        """Normalize action to [0, 1]."""
        return (a_next - self.a_min) / (self.a_max - self.a_min)
