"""
Utility functions for Krusell-Smith Model
=========================================
"""

import numpy as np


def compute_gini(x):
    """
    Compute the Gini coefficient of a distribution.
    
    Parameters:
    -----------
    x : ndarray - Values (e.g., wealth holdings)
    
    Returns:
    --------
    gini : float - Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    n = len(x)
    x_sorted = np.sort(x)
    cumsum = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_percentiles(x, percentiles=[10, 25, 50, 75, 90]):
    """
    Compute percentiles of a distribution.
    
    Parameters:
    -----------
    x : ndarray - Values
    percentiles : list - Percentiles to compute
    
    Returns:
    --------
    dict : Percentile values
    """
    return {p: np.percentile(x, p) for p in percentiles}


def compute_wealth_shares(x, groups=[(0, 50), (50, 90), (90, 99), (99, 100)]):
    """
    Compute wealth shares for different population groups.
    
    Parameters:
    -----------
    x : ndarray - Wealth values
    groups : list of tuples - (lower_percentile, upper_percentile)
    
    Returns:
    --------
    shares : dict - Wealth share for each group
    """
    total = np.sum(x)
    shares = {}
    
    for low, high in groups:
        low_val = np.percentile(x, low)
        high_val = np.percentile(x, high) if high < 100 else np.inf
        mask = (x >= low_val) & (x < high_val)
        shares[f'p{low}-{high}'] = np.sum(x[mask]) / total
    
    return shares


def moving_average(x, window=10):
    """Compute moving average."""
    return np.convolve(x, np.ones(window)/window, mode='valid')


def log_linear_regression(x, y):
    """
    Simple OLS for log-linear regression: log(y) = a + b*log(x)
    
    Returns:
    --------
    a, b, R2 : Intercept, slope, R-squared
    """
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Add constant
    X = np.column_stack([np.ones_like(log_x), log_x])
    
    # OLS
    beta = np.linalg.lstsq(X, log_y, rcond=None)[0]
    
    # R-squared
    y_pred = X @ beta
    ss_res = np.sum((log_y - y_pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    R2 = 1 - ss_res / ss_tot
    
    return beta[0], beta[1], R2
