import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from model import ConsumptionSavingModel

def plot_policy_comparison(model: ConsumptionSavingModel,
                         policies: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         w_idx: int = 0,
                         figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Plot and compare policy functions from different solution methods.
    
    Parameters:
    -----------
    model : ConsumptionSavingModel
        The model instance
    policies : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary of (consumption, next_capital) policies from different methods
    w_idx : int
        Index of wage state to plot (default: 0)
    figsize : Tuple[int, int]
        Size of the figure (default: (12, 5))
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot consumption policies
    for method_name, (c, _) in policies.items():
        ax1.plot(model.k_grid, c[:, w_idx], label=method_name, alpha=0.8)
    
    ax1.set_xlabel('Current Capital (k)')
    ax1.set_ylabel('Consumption (c)')
    ax1.set_title(f'Consumption Policy (w = {model.w_grid[w_idx]:.2f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot savings policies
    for method_name, (_, k_next) in policies.items():
        ax2.plot(model.k_grid, k_next[:, w_idx], label=method_name, alpha=0.8)
    
    # Add 45-degree line for reference
    k_min, k_max = model.k_grid[0], model.k_grid[-1]
    ax2.plot([k_min, k_max], [k_min, k_max], 'k--', alpha=0.3, label='45° line')
    
    ax2.set_xlabel('Current Capital (k)')
    ax2.set_ylabel('Next Period Capital (k\')')
    ax2.set_title(f'Capital Policy (w = {model.w_grid[w_idx]:.2f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_computation_time(times: Dict[str, float],
                         figsize: Tuple[int, int] = (8, 4)) -> None:
    """
    Plot computation times for different methods.
    
    Parameters:
    -----------
    times : Dict[str, float]
        Dictionary of computation times for each method
    figsize : Tuple[int, int]
        Size of the figure (default: (8, 4))
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(times.keys())
    times_list = [times[m] for m in methods]
    
    ax.bar(methods, times_list, alpha=0.7)
    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title('Algorithm Performance Comparison')
    
    # Add time values on top of bars
    for i, v in enumerate(times_list):
        ax.text(i, v + 0.01, f'{v:.3f}s', ha='center')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_convergence_paths(errors: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (10, 5)) -> None:
    """
    Plot convergence paths for different methods.
    
    Parameters:
    -----------
    errors : Dict[str, List[float]]
        Dictionary of error sequences for each method
    figsize : Tuple[int, int]
        Size of the figure (default: (10, 5))
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for method_name, error_seq in errors.items():
        iterations = range(1, len(error_seq) + 1)
        ax.semilogy(iterations, error_seq, label=method_name, alpha=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Policy Change (log scale)')
    ax.set_title('Convergence Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig