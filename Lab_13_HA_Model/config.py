"""
Krusell-Smith (1998) Model Configuration
========================================
This file contains all parameters and grid specifications for the KS model.

Model Overview:
- Continuum of agents facing idiosyncratic employment shocks
- Aggregate productivity shocks affect the entire economy
- Incomplete markets: agents can only save in capital (no insurance)
- Key insight: aggregate capital can be predicted by a simple log-linear rule
"""

import numpy as np

# =============================================================================
# 1. Structural Parameters
# =============================================================================
beta = 0.99      # Discount factor
gamma = 1.0      # Risk aversion (CRRA utility parameter, gamma=1 is log utility)
alpha = 0.36     # Capital share in production (Cobb-Douglas)
delta = 0.025    # Depreciation rate
mu = 0.15        # Unemployment benefits as share of wage

# Labor supply normalization
# l_bar normalizes so that aggregate labor = 1 when employment rate = 0.9
l_bar = 1 / 0.9

# Steady-state capital (deterministic model with L=1)
k_ss = ((1/beta - (1-delta)) / alpha) ** (1/(alpha-1))

# =============================================================================
# 2. Shock Process: Joint Markov Chain for (Z, epsilon)
# =============================================================================
# States: (bad-unemployed, bad-employed, good-unemployed, good-employed)
# Notation: bu=0, be=1, gu=2, ge=3

# Transition matrix from Den Haan, Judd, Juillard (2008)
Pi = np.array([
    [0.525,    0.35,     0.03125,  0.09375],   # from bu
    [0.038889, 0.836111, 0.002083, 0.122917],  # from be
    [0.09375,  0.03125,  0.291667, 0.583333],  # from gu
    [0.009115, 0.115885, 0.024306, 0.850694]   # from ge
])

# Unemployment rates by aggregate state
ur_bad = 0.10    # 10% unemployment in bad times
ur_good = 0.04   # 4% unemployment in good times

# Aggregate productivity levels
delta_z = 0.01
Z_grid = np.array([1 - delta_z, 1 + delta_z])  # [bad, good]
Z_ngrid = 2

# Employment states
epsilon_grid = np.array([0.0, 1.0])  # [unemployed, employed]
epsilon_ngrid = 2

# Aggregate labor supply (fraction employed)
L_grid = np.array([1 - ur_bad, 1 - ur_good])

# =============================================================================
# 3. State Space Grids
# =============================================================================
# Individual capital grid (k)
k_min, k_max = 0.0, 500.0
k_ngrid = 100
k_poly_degree = 7  # Polynomial degree for grid clustering near k_min

def create_polynomial_grid(n_points, x_min, x_max, degree):
    """Create grid with more points near x_min (useful for borrowing constraint)."""
    x = np.linspace(0, 0.5, n_points)
    y = x**degree / np.max(x**degree)
    return x_min + (x_max - x_min) * y

k_grid = create_polynomial_grid(k_ngrid, k_min, k_max, k_poly_degree)

# Aggregate capital grid (K)
K_min, K_max = 30.0, 50.0
K_ngrid = 4
K_grid = np.linspace(K_min, K_max, K_ngrid)

# =============================================================================
# 4. Price Functions
# =============================================================================
def compute_prices(K, L, Z):
    """
    Compute factor prices from firm's FOCs.
    
    Parameters:
    -----------
    K : float or array - Aggregate capital
    L : float or array - Aggregate labor (employment rate)
    Z : float or array - Aggregate productivity
    
    Returns:
    --------
    r : Interest rate (rental rate of capital)
    w : Wage rate
    tau : Tax rate for unemployment benefits
    """
    r = alpha * Z * (K / (l_bar * L))**(alpha - 1)
    w = (1 - alpha) * Z * (K / (l_bar * L))**alpha
    tau = mu * (1 - L) / (l_bar * L)  # Tax to fund unemployment benefits
    return r, w, tau

def compute_budget(k, epsilon, r, w, tau):
    """
    Compute agent's budget (cash-on-hand).
    
    Budget = after-tax labor income + unemployment benefits + capital income
    """
    labor_income = (1 - tau) * w * l_bar * epsilon
    unemployment_benefit = mu * w * (1 - epsilon)
    capital_income = (1 - delta + r) * k
    return labor_income + unemployment_benefit + capital_income

# =============================================================================
# 5. Algorithm Parameters
# =============================================================================
# Convergence criteria
tol_policy = 1e-8      # Tolerance for policy function iteration
tol_B = 1e-8           # Tolerance for PLM coefficients

# Damping factors (for stability)
damping_policy = 0.7   # Damping for policy function update
damping_B = 0.3        # Damping for PLM coefficient update

# Maximum iterations
max_iter_policy = 2000
max_iter_B = 50

# =============================================================================
# 6. Simulation Parameters
# =============================================================================
N_agents = 10_000      # Number of agents for stochastic simulation
T_sim = 1100           # Simulation length
T_burn = 100           # Burn-in periods to discard

# =============================================================================
# 7. Initialization
# =============================================================================
# Initial guess for PLM coefficients: K' = exp(B[0] + B[1]*log(K))
# B_init[z, :] = [intercept, slope] for aggregate state z
B_init = np.array([
    [0.0, 1.0],  # Bad state: log(K') = 0 + 1*log(K)
    [0.0, 1.0]   # Good state: log(K') = 0 + 1*log(K)
])

# =============================================================================
# 8. Precomputed Grids (for vectorized operations)
# =============================================================================
# Create meshgrid for all state combinations
k_idx, K_idx, Z_idx, eps_idx = np.meshgrid(
    range(k_ngrid), range(K_ngrid), range(Z_ngrid), range(epsilon_ngrid),
    indexing='ij'
)

# State values on the grid
k_mesh = k_grid[k_idx]
K_mesh = K_grid[K_idx]
Z_mesh = Z_grid[Z_idx]
eps_mesh = epsilon_grid[eps_idx]
L_mesh = L_grid[Z_idx]

# Prices on the grid
r_mesh, w_mesh, tau_mesh = compute_prices(K_mesh, L_mesh, Z_mesh)

# Budget on the grid
budget_mesh = compute_budget(k_mesh, eps_mesh, r_mesh, w_mesh, tau_mesh)

# Initial policy function guess: save 90% of current capital
policy_init = k_mesh * 0.9
