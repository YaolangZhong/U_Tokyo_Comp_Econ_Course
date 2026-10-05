import numpy as np

class ConsumptionSavingModel:
    def __init__(self, beta, R, sigma, b, k_grid, w_grid, P):
        """
        Initialize the consumption-savings model
        
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
        k_grid : array
            Grid for capital/assets
        w_grid : array
            Grid for income
        P : array (N x N)
            Transition matrix for income
        """
        self.beta = beta
        self.R = R
        self.sigma = sigma
        self.b = b
        self.k_grid = k_grid
        self.w_grid = w_grid
        self.P = P
        
        self.M = len(k_grid)
        self.N = len(w_grid)
        
        # Compute borrowing caps: pi_cap = R*k + w - b
        # This represents the maximum feasible consumption (next k = b)
        self.pi_cap = np.zeros((self.M, self.N))
        for m in range(self.M):
            for n in range(self.N):
                self.pi_cap[m,n] = R * k_grid[m] + w_grid[n] - b
    
    def u_prime(self, c):
        """Marginal utility: c^(-sigma)"""
        # Safeguard against c <= 0 is handled by solvers or here if needed
        # relying on numpy to handle arrays; for scalar <=0 it raises warning/error or returns inf
        return c ** (-self.sigma)
    
    def u_prime_inv(self, uc):
        """Inverse marginal utility: uc^(-1/sigma)"""
        return uc ** (-1/self.sigma)
