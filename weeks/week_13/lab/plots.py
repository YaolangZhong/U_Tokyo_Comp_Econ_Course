"""
Plotting utilities for Krusell-Smith Model
==========================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

from config import k_grid, K_grid, Z_ngrid, epsilon_ngrid, T_burn


def plot_ALM_vs_PLM(K_actual, K_plm, Z_history, title=None):
    """
    Compare actual aggregate capital (ALM) with PLM forecast.
    
    Parameters:
    -----------
    K_actual : ndarray (T,) - Actual aggregate capital
    K_plm : ndarray (T,) - PLM-forecasted capital
    Z_history : ndarray (T,) - Aggregate state history
    title : str - Plot title
    """
    T = len(K_actual)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series comparison
    t_plot = slice(T_burn, min(T, T_burn + 300))
    axes[0].plot(K_actual[t_plot], 'b-', label='Actual (ALM)', linewidth=1.5)
    axes[0].plot(K_plm[t_plot], 'r--', label='Forecast (PLM)', linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Aggregate Capital K')
    axes[0].set_title(title or 'Aggregate Capital: ALM vs PLM')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Forecast error
    error = K_actual - K_plm
    axes[1].plot(error[t_plot], 'k-', linewidth=0.8)
    axes[1].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Forecast Error')
    axes[1].set_xlabel('Period')
    axes[1].set_title('PLM Forecast Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_policy_function(policy, K_value=None, title=None):
    """
    Plot policy function for different states.
    
    Parameters:
    -----------
    policy : ndarray (k_ngrid, K_ngrid, Z_ngrid, ε_ngrid)
        Policy function
    K_value : float - Aggregate capital value (uses median grid point if None)
    title : str - Plot title
    """
    if K_value is None:
        K_idx = len(K_grid) // 2
    else:
        K_idx = np.argmin(np.abs(K_grid - K_value))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    k_plot = k_grid[:50]  # Focus on lower k
    
    state_labels = [
        ('Bad', 'Unemployed'),
        ('Bad', 'Employed'),
        ('Good', 'Unemployed'),
        ('Good', 'Employed')
    ]
    
    for idx, (Z_label, eps_label) in enumerate(state_labels):
        Z_idx = 0 if Z_label == 'Bad' else 1
        eps_idx = 0 if eps_label == 'Unemployed' else 1
        
        ax = axes[idx // 2, idx % 2]
        kp = policy[:50, K_idx, Z_idx, eps_idx]
        
        ax.plot(k_plot, kp, 'b-', linewidth=2, label="k'(k)")
        ax.plot(k_plot, k_plot, 'k--', alpha=0.5, label='45° line')
        ax.set_xlabel('Current Capital k')
        ax.set_ylabel("Next Period Capital k'")
        ax.set_title(f'{Z_label} State, {eps_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title or f'Policy Function at K = {K_grid[K_idx]:.1f}', fontsize=14)
    plt.tight_layout()
    return fig


def plot_wealth_distribution(k_distribution, title=None):
    """
    Plot cross-sectional wealth distribution.
    
    Parameters:
    -----------
    k_distribution : ndarray (N,) - Individual capital holdings
    title : str - Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(k_distribution, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(k_distribution), color='r', linestyle='--',
                    label=f'Mean = {np.mean(k_distribution):.1f}')
    axes[0].axvline(np.median(k_distribution), color='g', linestyle='--',
                    label=f'Median = {np.median(k_distribution):.1f}')
    axes[0].set_xlabel('Individual Capital k')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Wealth Distribution')
    axes[0].legend()
    
    # Lorenz curve
    k_sorted = np.sort(k_distribution)
    k_cumsum = np.cumsum(k_sorted) / np.sum(k_sorted)
    pop_share = np.arange(1, len(k_sorted)+1) / len(k_sorted)
    
    axes[1].plot(pop_share, k_cumsum, 'b-', linewidth=2, label='Lorenz Curve')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
    axes[1].fill_between(pop_share, k_cumsum, pop_share, alpha=0.3)
    axes[1].set_xlabel('Cumulative Population Share')
    axes[1].set_ylabel('Cumulative Wealth Share')
    axes[1].set_title('Lorenz Curve')
    axes[1].legend()
    
    # Gini coefficient
    gini = 1 - 2 * np.trapz(k_cumsum, pop_share)
    axes[1].text(0.6, 0.2, f'Gini = {gini:.3f}', fontsize=12)
    
    plt.suptitle(title or 'Cross-Sectional Wealth Distribution', fontsize=14)
    plt.tight_layout()
    return fig, gini


def compute_gini(x):
    """Compute Gini coefficient."""
    n = len(x)
    x_sorted = np.sort(x)
    cumsum = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
