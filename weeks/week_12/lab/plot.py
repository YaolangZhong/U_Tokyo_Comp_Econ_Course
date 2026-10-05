"""
Plotting utilities for Lab 12: Actor-Critic Methods

This module provides visualization functions for:
- Training curves
- Policy function comparisons
- Asset distribution plots
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_training_curves(critic_losses: List[float], actor_losses: List[float],
                         rewards_history: List[float]) -> plt.Figure:
    """
    Plot training curves for the actor-critic algorithm.
    
    Parameters
    ----------
    critic_losses : List[float]
        Critic loss history
    actor_losses : List[float]
        Actor loss history
    rewards_history : List[float]
        Average reward history
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].plot(critic_losses)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Critic Loss')
    axes[0].set_title('Critic Loss (Bellman Error)')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(actor_losses)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Actor Loss')
    axes[1].set_title('Actor Loss (-Q value)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(rewards_history)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Average Reward')
    axes[2].set_title('Average Reward (Utility)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_policy_comparison(a_grid: np.ndarray, vfi_policy: np.ndarray,
                           nn_policy: np.ndarray, z_grid: np.ndarray,
                           z_idx: Optional[int] = None) -> plt.Figure:
    """
    Compare policy functions from VFI and neural network.
    
    Parameters
    ----------
    a_grid : np.ndarray
        Asset grid
    vfi_policy : np.ndarray
        VFI policy function (n_a, n_z)
    nn_policy : np.ndarray
        Neural network policy function (n_a,) for given z_idx
    z_grid : np.ndarray
        Productivity grid
    z_idx : int, optional
        Productivity index (if None, plots all)
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_z = len(z_grid)
    
    if z_idx is not None:
        # Single productivity state
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(a_grid, vfi_policy[:, z_idx], 'b-', linewidth=2, label='VFI')
        ax.plot(a_grid, nn_policy, 'r--', linewidth=2, label='Actor-Critic')
        ax.plot(a_grid, a_grid, 'k:', alpha=0.5, label='45° line')
        
        ax.set_xlabel('Current Assets $a$')
        ax.set_ylabel("Next Assets $a'$")
        ax.set_title(f'Policy Function (z = {z_grid[z_idx]:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # All productivity states (nn_policy should be a list or 2D array)
        fig, axes = plt.subplots(1, n_z, figsize=(4 * n_z, 4))
        
        for i_z in range(n_z):
            ax = axes[i_z] if n_z > 1 else axes
            
            ax.plot(a_grid, vfi_policy[:, i_z], 'b-', linewidth=2, label='VFI')
            if isinstance(nn_policy, np.ndarray) and nn_policy.ndim == 2:
                ax.plot(a_grid, nn_policy[:, i_z], 'r--', linewidth=2, label='Actor-Critic')
            ax.plot(a_grid, a_grid, 'k:', alpha=0.5, label='45° line')
            
            ax.set_xlabel('Current Assets $a$')
            ax.set_ylabel("Next Assets $a'$")
            ax.set_title(f'z = {z_grid[i_z]:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Policy Function Comparison: VFI vs Actor-Critic', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_policy_all_z(a_grid: np.ndarray, vfi_policy: np.ndarray,
                      get_nn_policy_func, z_grid: np.ndarray) -> plt.Figure:
    """
    Plot policy comparison for all productivity states.
    
    Parameters
    ----------
    a_grid : np.ndarray
        Asset grid
    vfi_policy : np.ndarray
        VFI policy function (n_a, n_z)
    get_nn_policy_func : callable
        Function that takes (a_grid, z_idx) and returns policy
    z_grid : np.ndarray
        Productivity grid
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_z = len(z_grid)
    fig, axes = plt.subplots(1, n_z, figsize=(4 * n_z, 4))
    
    for i_z in range(n_z):
        ax = axes[i_z] if n_z > 1 else axes
        
        # VFI policy
        ax.plot(a_grid, vfi_policy[:, i_z], 'b-', linewidth=2, label='VFI')
        
        # Neural network policy
        nn_policy = get_nn_policy_func(a_grid, i_z)
        ax.plot(a_grid, nn_policy, 'r--', linewidth=2, label='Actor-Critic')
        
        # 45-degree line
        ax.plot(a_grid, a_grid, 'k:', alpha=0.5, label='45° line')
        
        ax.set_xlabel('Current Assets $a$')
        ax.set_ylabel("Next Assets $a'$")
        ax.set_title(f'z = {z_grid[i_z]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Policy Function Comparison: VFI vs Actor-Critic', fontsize=14)
    plt.tight_layout()
    return fig


def plot_asset_distribution(a_sim: np.ndarray, z_sim: np.ndarray,
                            z_grid: np.ndarray) -> plt.Figure:
    """
    Plot the simulated asset distribution.
    
    Parameters
    ----------
    a_sim : np.ndarray
        Simulated asset holdings
    z_sim : np.ndarray
        Simulated productivity indices
    z_grid : np.ndarray
        Productivity grid
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Overall distribution
    axes[0].hist(a_sim, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(a_sim), color='r', linestyle='--', 
                    label=f'Mean = {np.mean(a_sim):.2f}')
    axes[0].axvline(np.median(a_sim), color='g', linestyle='--', 
                    label=f'Median = {np.median(a_sim):.2f}')
    axes[0].set_xlabel('Assets')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Stationary Asset Distribution')
    axes[0].legend()
    
    # Distribution by productivity
    n_z = len(z_grid)
    for i_z in range(n_z):
        mask = z_sim == i_z
        if np.sum(mask) > 0:
            axes[1].hist(a_sim[mask], bins=30, density=True, alpha=0.5, 
                         label=f'z = {z_grid[i_z]:.2f}')
    axes[1].set_xlabel('Assets')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Asset Distribution by Productivity')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def compute_policy_errors(a_grid: np.ndarray, vfi_policy: np.ndarray,
                          get_nn_policy_func, z_grid: np.ndarray,
                          verbose: bool = True) -> dict:
    """
    Compute approximation errors between VFI and neural network policies.
    
    Parameters
    ----------
    a_grid : np.ndarray
        Asset grid
    vfi_policy : np.ndarray
        VFI policy function (n_a, n_z)
    get_nn_policy_func : callable
        Function that takes (a_grid, z_idx) and returns policy
    z_grid : np.ndarray
        Productivity grid
    verbose : bool
        Whether to print errors
        
    Returns
    -------
    errors : dict
        Dictionary containing error metrics
    """
    n_z = len(z_grid)
    errors = {
        'MAE': [],
        'Max Error': [],
        'Rel Error': [],
        'z': z_grid.tolist()
    }
    
    for i_z in range(n_z):
        nn_policy = get_nn_policy_func(a_grid, i_z)
        vfi_pol = vfi_policy[:, i_z]
        
        mae = np.mean(np.abs(nn_policy - vfi_pol))
        max_error = np.max(np.abs(nn_policy - vfi_pol))
        rel_error = np.mean(np.abs(nn_policy - vfi_pol) / (np.abs(vfi_pol) + 1e-8))
        
        errors['MAE'].append(mae)
        errors['Max Error'].append(max_error)
        errors['Rel Error'].append(rel_error)
        
        if verbose:
            print(f"z = {z_grid[i_z]:.3f}: MAE = {mae:.4f}, "
                  f"Max Error = {max_error:.4f}, Rel Error = {rel_error:.2%}")
    
    if verbose:
        avg_mae = np.mean(errors['MAE'])
        print(f"\nOverall Mean Absolute Error: {avg_mae:.4f}")
    
    return errors
