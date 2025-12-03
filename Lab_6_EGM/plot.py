import numpy as np
import matplotlib.pyplot as plt

def plot_policy_comparison(model, policies, labels, w_idx=None):
    """
    Plot policy functions from different algorithms
    
    Parameters:
    -----------
    model : ConsumptionSavingModel
        The model object
    policies : list of arrays
        List of policy functions to compare
    labels : list of str
        Labels for each policy function
    w_idx : int, optional
        Index of income state to plot (defaults to middle state)
    """
    if w_idx is None:
        w_idx = model.N // 2
        
    plt.figure(figsize=(10, 6))
    
    # Plot 45 degree line (or borrowing cap) for reference
    # plt.plot(model.k_grid, model.pi_cap[:, w_idx], 'k--', alpha=0.3, label='Max feasible c')
    
    for policy, label in zip(policies, labels):
        plt.plot(model.k_grid, policy[:, w_idx], label=label, linewidth=2, alpha=0.8)
        
    plt.xlabel('Capital Stock ($k_t$)')
    plt.ylabel('Consumption ($c_t$)')
    plt.title(f'Policy Functions Comparison (Income w = {model.w_grid[w_idx]:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gca()

def plot_policy_heat(model, policy, method_name):
    """
    Plot policy function as a heatmap
    """
    plt.figure(figsize=(10, 6))
    
    # Use pcolormesh for better grid handling
    X, Y = np.meshgrid(model.k_grid, model.w_grid)
    
    # Note: policy is (M, N), we need (N, M) for meshgrid if X is k (M) and Y is w (N)
    # But meshgrid usually expects X (rows) Y (cols) or vice versa depending on indexing
    # Let's use imshow for simplicity as in original, or contourf
    
    plt.contourf(X, Y, policy.T, levels=20, cmap='viridis')
    
    plt.colorbar(label='Consumption ($c_t$)')
    plt.xlabel('Capital Stock ($k_t$)')
    plt.ylabel('Income ($w_t$)')
    plt.title(f'Policy Function - {method_name}')
    
    return plt.gca()

def plot_convergence(errors_list, labels):
    """
    Plot convergence paths for different methods
    """
    plt.figure(figsize=(10, 6))
    
    for errors, label in zip(errors_list, labels):
        plt.semilogy(errors, label=label, linewidth=2)
        
    plt.xlabel('Iteration')
    plt.ylabel('Max Error ($||c_{n+1} - c_n||_\infty$)')
    plt.title('Convergence Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    return plt.gca()

def plot_euler_errors(model, policies, labels, w_idx=None):
    """
    Plot Euler equation errors (optional analysis)
    """
    pass # Placeholder
