import numpy as np
from typing import Tuple, Callable, Union
from scipy.interpolate import interp1d

class ConsumptionSavingModel:
    """
    A class representing the consumption-saving model with borrowing constraints.
    """
    def __init__(self, 
                 beta: float,
                 R: float,
                 sigma: float,
                 b: float,
                 k_grid: np.ndarray,
                 w_grid: np.ndarray,
                 P: np.ndarray):
        """
        Initialize the consumption-saving model.
        
        Parameters:
        -----------
        beta : float
            Discount factor
        R : float
            Gross interest rate (1 + r)
        sigma : float
            Risk aversion parameter
        b : float
            Borrowing constraint
        k_grid : np.ndarray
            Grid for capital
        w_grid : np.ndarray
            Grid for wage
        P : np.ndarray
            Transition matrix for wage states
        """
        self.beta = beta
        self.R = R
        self.sigma = sigma
        self.b = b
        self.k_grid = k_grid
        self.w_grid = w_grid
        self.P = P
        
        # Derived quantities
        self.M = len(k_grid)
        self.N = len(w_grid)
        
        # Pre-compute borrowing caps
        self.pi_cap = np.zeros((self.M, self.N))
        for m in range(self.M):
            for n in range(self.N):
                self.pi_cap[m, n] = self.R * self.k_grid[m] + self.w_grid[n] - self.b
    
    def utility(self, c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute utility for given consumption level."""
        if self.sigma == 1:
            return np.log(c)
        return (c**(1 - self.sigma) - 1) / (1 - self.sigma)
    
    def marginal_utility(self, c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute marginal utility for given consumption level."""
        return c**(-self.sigma)
    
    def inverse_marginal_utility(self, mu: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute inverse of marginal utility."""
        return mu**(-1/self.sigma)
    
    def interpolate_policy(self, k: Union[float, np.ndarray], w: float, k_grid: np.ndarray, 
                          policy: np.ndarray, w_index: int) -> Union[float, np.ndarray]:
        """
        Interpolate policy function for given capital and wage state.
        
        Parameters:
        -----------
        k : float or np.ndarray
            Capital value(s) to interpolate at
        w : float
            Current wage state
        k_grid : np.ndarray
            Grid points for capital
        policy : np.ndarray
            Policy values on the grid
        w_index : int
            Index of current wage state
            
        Returns:
        --------
        float or np.ndarray
            Interpolated policy values
        """
        # Create interpolation function (use linear interpolation with fill value)
        interp = interp1d(k_grid, policy[:, w_index], 
                         bounds_error=False, 
                         fill_value=(policy[0, w_index], policy[-1, w_index]))
        
        # Handle both scalar and array inputs
        if np.isscalar(k):
            return float(interp(k))
        return interp(k)
    
    def next_capital(self, k: float, w: float, c: float) -> float:
        """Compute next period capital given current state and consumption."""
        return self.R * k + w - c
    
    def state_constraint(self, k: float, w: float) -> float:
        """Compute maximum consumption given current state (borrowing cap)."""
        return self.R * k + w - self.b