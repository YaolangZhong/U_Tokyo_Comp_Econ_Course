"""
Household Problem for Krusell-Smith Model
==========================================
Solves the individual household's dynamic optimization problem using
Euler equation iteration (time iteration).

The household problem:
    max E[sum_{t=0}^∞ β^t u(c_t)]
    
    s.t. c + k' = (1-τ)wl·ε + μw(1-ε) + (1-δ+r)k
         k' >= 0
         
State variables: (k, K, Z, ε)
    k: individual capital
    K: aggregate capital (used to forecast prices via PLM)
    Z: aggregate productivity shock
    ε: individual employment status

The PLM (Perceived Law of Motion) for K:
    log(K') = B[Z, 0] + B[Z, 1] * log(K)
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline

from config import (
    beta, gamma, delta, k_min, k_max, k_grid, K_grid, K_min, K_max,
    Z_grid, Z_ngrid, epsilon_grid, epsilon_ngrid, L_grid, Pi,
    compute_prices, compute_budget, budget_mesh,
    Z_idx, eps_idx, K_mesh,
    damping_policy, tol_policy, max_iter_policy
)


def utility(c):
    """CRRA utility function."""
    if gamma == 1:
        return np.log(np.maximum(c, 1e-10))
    else:
        return (np.maximum(c, 1e-10)**(1 - gamma) - 1) / (1 - gamma)


def marginal_utility(c):
    """Marginal utility u'(c) = c^(-γ)."""
    return np.maximum(c, 1e-10)**(-gamma)


def inverse_marginal_utility(mu):
    """Inverse of marginal utility: (u')^{-1}(mu) = mu^(-1/γ)."""
    return np.maximum(mu, 1e-10)**(-1/gamma)


def compute_Kprime(K, Z_idx_grid, B):
    """
    Compute next-period aggregate capital using PLM.
    
    PLM: log(K') = B[Z, 0] + B[Z, 1] * log(K)
    
    Parameters:
    -----------
    K : ndarray - Current aggregate capital
    Z_idx_grid : ndarray - Aggregate state indices
    B : ndarray (2, 2) - PLM coefficients [intercept, slope] for each Z
    
    Returns:
    --------
    K_prime : ndarray - Next-period aggregate capital
    """
    log_K_prime = B[Z_idx_grid, 0] + B[Z_idx_grid, 1] * np.log(K)
    K_prime = np.exp(log_K_prime)
    return np.clip(K_prime, K_min, K_max)


def euler_equation_iteration(policy, B, verbose=False):
    """
    One iteration of the Euler equation method.
    
    Given current policy k'(k, K, Z, ε) and PLM coefficients B,
    compute the updated policy using the Euler equation:
    
    u'(c) = β E[(1-δ+r') u'(c')]
    
    Parameters:
    -----------
    policy : ndarray (k_ngrid, K_ngrid, Z_ngrid, ε_ngrid)
        Current policy function k'(k, K, Z, ε)
    B : ndarray (2, 2)
        PLM coefficients
    verbose : bool
        Print progress
    
    Returns:
    --------
    policy_new : ndarray - Updated policy function
    consumption : ndarray - Implied consumption
    policy_diff : float - Max absolute change in policy
    euler_error : float - Mean squared Euler equation error
    """
    # Current consumption
    consumption = budget_mesh - policy
    consumption = np.maximum(consumption, 1e-10)
    
    # Compute expected marginal utility for next period
    # E[u'(c') * (1-δ+r')]
    expected_mu = np.zeros_like(policy)
    
    # Next period aggregate capital from PLM
    K_prime = compute_Kprime(K_mesh, Z_idx, B)
    
    # Sum over all possible next-period states (Z', ε')
    for Z_next in range(Z_ngrid):
        for eps_next in range(epsilon_ngrid):
            # Transition probability P(Z', ε' | Z, ε)
            # State index: s = Z * ε_ngrid + ε
            s_current = Z_idx * epsilon_ngrid + eps_idx
            s_next = Z_next * epsilon_ngrid + eps_next
            prob = Pi[s_current, s_next]
            
            # Next period prices
            Z_prime = Z_grid[Z_next]
            L_prime = L_grid[Z_next]
            r_prime, w_prime, tau_prime = compute_prices(K_prime, L_prime, Z_prime)
            
            # Next period budget
            eps_prime = epsilon_grid[eps_next]
            budget_prime = compute_budget(policy, eps_prime, r_prime, w_prime, tau_prime)
            
            # Interpolate next-period policy k''(k', K', Z', ε')
            # For each (Z_next, eps_next), we interpolate over (k, K)
            policy_slice = policy[:, :, Z_next, eps_next]
            interpolator = RectBivariateSpline(k_grid, K_grid, policy_slice)
            policy_prime_prime = interpolator.ev(policy, K_prime)
            policy_prime_prime = np.clip(policy_prime_prime, k_min, k_max)
            
            # Next period consumption
            c_prime = budget_prime - policy_prime_prime
            c_prime = np.maximum(c_prime, 1e-10)
            
            # Accumulate expected marginal utility
            expected_mu += prob * (1 - delta + r_prime) * marginal_utility(c_prime)
    
    # Euler equation: u'(c) = β * E[u'(c') * (1-δ+r')]
    # => c = (β * E[...])^(-1/γ)
    c_euler = inverse_marginal_utility(beta * expected_mu)
    
    # New policy from budget constraint
    policy_new = budget_mesh - c_euler
    
    # Apply borrowing constraint
    policy_new = np.clip(policy_new, k_min, k_max)
    
    # Compute Euler equation error (for interior solutions)
    # Error = |u'(c) / (β * E[...]) - 1|
    mu_ratio = marginal_utility(c_euler) / (beta * expected_mu)
    interior = (policy_new > k_min + 1e-6) & (policy_new < k_max - 1e-6)
    euler_error = np.mean((mu_ratio[interior] - 1)**2) if np.any(interior) else 0.0
    
    # Policy change
    policy_diff = np.max(np.abs(policy_new - policy))
    
    # Damping for stability
    policy_new = damping_policy * policy_new + (1 - damping_policy) * policy
    
    return policy_new, c_euler, policy_diff, euler_error


def solve_household_problem(policy_init, B, verbose=True):
    """
    Solve the household's problem via Euler equation iteration.
    
    Parameters:
    -----------
    policy_init : ndarray (k_ngrid, K_ngrid, Z_ngrid, ε_ngrid)
        Initial guess for policy function
    B : ndarray (2, 2)
        PLM coefficients
    verbose : bool
        Print convergence progress
    
    Returns:
    --------
    policy : ndarray - Converged policy function
    consumption : ndarray - Implied consumption function
    euler_error : float - Final Euler equation error
    converged : bool - Whether iteration converged
    """
    policy = policy_init.copy()
    
    for iteration in range(max_iter_policy):
        policy, consumption, policy_diff, euler_error = euler_equation_iteration(policy, B)
        
        if verbose and iteration % 100 == 0:
            print(f"  Policy iteration {iteration}: diff = {policy_diff:.2e}, euler = {euler_error:.2e}")
        
        if policy_diff < tol_policy:
            if verbose:
                print(f"  Policy converged in {iteration} iterations, euler error = {euler_error:.2e}")
            return policy, consumption, euler_error, True
    
    if verbose:
        print(f"  Policy did not converge after {max_iter_policy} iterations")
    return policy, consumption, euler_error, False


def interpolate_policy(policy, k, K, Z_idx, eps_idx):
    """
    Interpolate policy function at arbitrary state points.
    
    Parameters:
    -----------
    policy : ndarray (k_ngrid, K_ngrid, Z_ngrid, ε_ngrid)
        Policy function on grid
    k : ndarray - Individual capital values
    K : ndarray - Aggregate capital values
    Z_idx : ndarray - Aggregate state indices (0 or 1)
    eps_idx : ndarray - Employment state indices (0 or 1)
    
    Returns:
    --------
    k_prime : ndarray - Policy function values at query points
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolator for full 4D grid
    interpolator = RegularGridInterpolator(
        (k_grid, K_grid, Z_grid, epsilon_grid),
        policy,
        bounds_error=False,
        fill_value=None
    )
    
    # Query points
    Z = Z_grid[Z_idx]
    eps = epsilon_grid[eps_idx]
    points = np.column_stack([k, K, Z, eps])
    
    k_prime = interpolator(points)
    return np.clip(k_prime, k_min, k_max)
