"""
Aggregate Simulation for Krusell-Smith Model
=============================================
Simulates the economy forward given:
- A policy function k'(k, K, Z, ε)
- Shock sequences for aggregate and idiosyncratic states

The simulation tracks:
- Cross-sectional distribution of individual capital
- Time series of aggregate capital K_t = (1/N) Σ_i k_it
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from config import (
    k_grid, K_grid, Z_grid, epsilon_grid,
    k_min, k_max, K_min, K_max,
    T_burn, damping_B
)


def interpolate_policy_fast(policy, k_agents, K_agg, Z_idx, eps_agents):
    """
    Fast interpolation of policy function for many agents.
    
    Parameters:
    -----------
    policy : ndarray (k_ngrid, K_ngrid, Z_ngrid, ε_ngrid)
        Policy function on grid
    k_agents : ndarray (N,) - Individual capital holdings
    K_agg : float - Aggregate capital
    Z_idx : int - Aggregate state (0=bad, 1=good)
    eps_agents : ndarray (N,) - Employment status (0=unemployed, 1=employed)
    
    Returns:
    --------
    k_prime : ndarray (N,) - Next-period capital choices
    """
    N = len(k_agents)
    
    # Build query points
    Z_val = Z_grid[Z_idx]
    eps_vals = epsilon_grid[eps_agents]
    
    points = np.column_stack([
        k_agents,
        np.full(N, K_agg),
        np.full(N, Z_val),
        eps_vals
    ])
    
    # Interpolate
    interpolator = RegularGridInterpolator(
        (k_grid, K_grid, Z_grid, epsilon_grid),
        policy,
        bounds_error=False,
        fill_value=None
    )
    
    k_prime = interpolator(points)
    return np.clip(k_prime, k_min, k_max)


def simulate_economy(policy, k_init, Z_history, eps_history):
    """
    Simulate the economy forward given policy and shock sequences.
    
    Parameters:
    -----------
    policy : ndarray (k_ngrid, K_ngrid, Z_ngrid, ε_ngrid)
        Policy function
    k_init : ndarray (N,) - Initial capital distribution
    Z_history : ndarray (T,) - Aggregate shock sequence
    eps_history : ndarray (T, N) - Idiosyncratic shock sequences
    
    Returns:
    --------
    K_history : ndarray (T,) - Aggregate capital time series
    k_final : ndarray (N,) - Final cross-sectional distribution
    """
    T = len(Z_history)
    N = len(k_init)
    
    K_history = np.zeros(T)
    k_agents = k_init.copy()
    
    for t in range(T):
        # Compute aggregate capital
        K_t = np.clip(np.mean(k_agents), K_min, K_max)
        K_history[t] = K_t
        
        # Get next-period capital for all agents
        k_agents = interpolate_policy_fast(
            policy, k_agents, K_t, Z_history[t], eps_history[t]
        )
    
    return K_history, k_agents


def update_PLM_coefficients(B_old, K_history, Z_history, verbose=True):
    """
    Update PLM coefficients using OLS regression.
    
    PLM: log(K') = B[Z, 0] + B[Z, 1] * log(K)
    
    Run separate regressions for bad (Z=0) and good (Z=1) states.
    
    Parameters:
    -----------
    B_old : ndarray (2, 2) - Current PLM coefficients
    K_history : ndarray (T,) - Aggregate capital time series
    Z_history : ndarray (T,) - Aggregate state history
    verbose : bool - Print regression results
    
    Returns:
    --------
    B_new : ndarray (2, 2) - Updated PLM coefficients
    B_diff : float - Change in coefficients
    R2 : ndarray (2,) - R-squared for each regression
    """
    # Use data after burn-in
    log_K = np.log(K_history[T_burn:-1])
    log_K_prime = np.log(K_history[T_burn+1:])
    Z = Z_history[T_burn:-1]
    
    B_new = np.zeros((2, 2))
    R2 = np.zeros(2)
    
    for z in range(2):
        # Select observations for this aggregate state
        mask = (Z == z)
        if np.sum(mask) < 10:
            # Not enough observations, keep old coefficients
            B_new[z] = B_old[z]
            continue
        
        # OLS: log(K') = b0 + b1 * log(K) using numpy
        x = log_K[mask]
        y = log_K_prime[mask]
        X = np.column_stack([np.ones_like(x), x])
        
        # OLS: beta = (X'X)^{-1} X'y
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        B_new[z, 0] = beta[0]  # Intercept
        B_new[z, 1] = beta[1]  # Slope
        
        # R-squared
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        R2[z] = 1 - ss_res / ss_tot
    
    # Compute change
    B_diff = np.max(np.abs(B_new - B_old))
    
    if verbose:
        print(f"  PLM update: ||ΔB|| = {B_diff:.2e}")
        print(f"    Bad state:  log(K') = {B_new[0,0]:.4f} + {B_new[0,1]:.4f} * log(K), R² = {R2[0]:.6f}")
        print(f"    Good state: log(K') = {B_new[1,0]:.4f} + {B_new[1,1]:.4f} * log(K), R² = {R2[1]:.6f}")
    
    # Apply damping
    B_updated = damping_B * B_new + (1 - damping_B) * B_old
    
    return B_updated, B_diff, R2


def compute_PLM_forecast(B, K_history, Z_history):
    """
    Compute PLM-forecasted K series for comparison with actual.
    
    Parameters:
    -----------
    B : ndarray (2, 2) - PLM coefficients
    K_history : ndarray (T,) - Actual aggregate capital
    Z_history : ndarray (T,) - Aggregate state history
    
    Returns:
    --------
    K_plm : ndarray (T,) - PLM-forecasted capital series
    """
    T = len(K_history)
    K_plm = np.zeros(T)
    K_plm[0] = K_history[0]
    
    for t in range(T - 1):
        z = Z_history[t]
        K_plm[t+1] = np.exp(B[z, 0] + B[z, 1] * np.log(K_plm[t]))
        K_plm[t+1] = np.clip(K_plm[t+1], K_min, K_max)
    
    return K_plm


def compute_forecast_errors(K_actual, K_plm, Z_history):
    """
    Compute forecast errors by aggregate state.
    
    Returns:
    --------
    errors : dict with MAE, RMSE, and max error by state
    """
    errors = {}
    
    for z, state_name in enumerate(['bad', 'good']):
        mask = Z_history[T_burn:-1] == z
        actual = K_actual[T_burn+1:][mask]
        forecast = K_plm[T_burn+1:][mask]
        
        if len(actual) > 0:
            errors[state_name] = {
                'MAE': np.mean(np.abs(actual - forecast)),
                'RMSE': np.sqrt(np.mean((actual - forecast)**2)),
                'max_error': np.max(np.abs(actual - forecast))
            }
    
    return errors
