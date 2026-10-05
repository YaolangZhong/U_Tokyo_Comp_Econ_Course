"""
Shock Generation for Krusell-Smith Model
=========================================
Handles the joint Markov process for aggregate (Z) and idiosyncratic (epsilon) shocks.

The joint process has 4 states: (bad-unemployed, bad-employed, good-unemployed, good-employed)
We decompose this into:
1. Aggregate shock transition: Z -> Z'
2. Idiosyncratic shock transition: epsilon -> epsilon' | (Z, Z')
"""

import numpy as np

def get_stationary_distribution(transition_matrix):
    """
    Compute the stationary distribution of a Markov chain.
    
    The stationary distribution π satisfies: π = π @ P
    This is the eigenvector corresponding to eigenvalue 1.
    """
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    # Normalize to sum to 1
    return stationary / np.sum(stationary)


class ShockSimulator:
    """
    Simulates the joint shock process for aggregate and idiosyncratic states.
    
    Attributes:
    -----------
    Pi : ndarray (4, 4) - Joint transition matrix
    Pi_Z : ndarray (2, 2) - Marginal transition for aggregate state
    Pi_eps_given_Z : ndarray (2, 2, 2, 2) - Conditional transition P(eps'|eps, Z, Z')
    """
    
    def __init__(self, Pi):
        """
        Initialize with the joint transition matrix.
        
        Parameters:
        -----------
        Pi : ndarray (4, 4)
            Joint transition matrix with states ordered as:
            (bad-unemployed, bad-employed, good-unemployed, good-employed)
        """
        self.Pi = Pi
        self.stationary = get_stationary_distribution(Pi)
        
        # Decompose joint process
        self.Pi_Z = self._compute_aggregate_transition()
        self.stationary_Z = get_stationary_distribution(self.Pi_Z)
        self.Pi_eps_given_Z = self._compute_conditional_transition()
        
        # Unemployment rates by aggregate state
        # P(unemployed | bad) = P(bad, unemployed) / P(bad)
        self.ur_bad = self.stationary[0] / self.stationary_Z[0]
        self.ur_good = self.stationary[2] / self.stationary_Z[1]
    
    def _compute_aggregate_transition(self):
        """Compute marginal transition matrix for aggregate state Z."""
        Pi, P = self.Pi, self.stationary
        Pi_Z = np.zeros((2, 2))
        
        # P(bad -> bad) = sum over epsilon states
        Pi_Z[0, 0] = (P[0] * (Pi[0, 0] + Pi[0, 1]) + P[1] * (Pi[1, 0] + Pi[1, 1])) / (P[0] + P[1])
        Pi_Z[0, 1] = 1 - Pi_Z[0, 0]
        
        # P(good -> good)
        Pi_Z[1, 1] = (P[2] * (Pi[2, 2] + Pi[2, 3]) + P[3] * (Pi[3, 2] + Pi[3, 3])) / (P[2] + P[3])
        Pi_Z[1, 0] = 1 - Pi_Z[1, 1]
        
        return Pi_Z
    
    def _compute_conditional_transition(self):
        """
        Compute conditional transition P(epsilon' | epsilon, Z, Z').
        
        Returns array with shape (Z, Z', epsilon, epsilon').
        """
        Pi_cond = np.zeros((2, 2, 2, 2))
        for z in range(2):
            for z_next in range(2):
                for eps in range(2):
                    for eps_next in range(2):
                        # Joint state index: epsilon + 2*Z
                        s = eps + 2 * z
                        s_next = eps_next + 2 * z_next
                        # P(eps'|eps, Z->Z') = P(s->s') / P(Z->Z')
                        Pi_cond[z, z_next, eps, eps_next] = self.Pi[s, s_next] / self.Pi_Z[z, z_next]
        return Pi_cond
    
    def initialize_agents(self, n_agents, Z_init=0, seed=None):
        """
        Initialize agents' employment status given initial aggregate state.
        
        Parameters:
        -----------
        n_agents : int - Number of agents
        Z_init : int - Initial aggregate state (0=bad, 1=good)
        seed : int - Random seed
        
        Returns:
        --------
        Z_idx : int - Initial aggregate state
        eps_idx : ndarray (n_agents,) - Initial employment states
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Draw employment status consistent with unemployment rate
        ur = self.ur_bad if Z_init == 0 else self.ur_good
        eps_idx = (np.random.rand(n_agents) > ur).astype(int)
        
        return Z_init, eps_idx
    
    def transition_step(self, Z_idx, eps_idx):
        """
        Simulate one step of the shock process.
        
        Parameters:
        -----------
        Z_idx : int - Current aggregate state
        eps_idx : ndarray (n_agents,) - Current employment states
        
        Returns:
        --------
        Z_next : int - Next aggregate state
        eps_next : ndarray (n_agents,) - Next employment states
        """
        # Draw next aggregate state
        Z_next = np.random.choice([0, 1], p=self.Pi_Z[Z_idx])
        
        # Draw next employment state for each agent
        # P(eps_next | eps, Z, Z_next)
        probs = self.Pi_eps_given_Z[Z_idx, Z_next, eps_idx]  # Shape (n_agents, 2)
        # probs[:, 0] = P(unemployed), probs[:, 1] = P(employed)
        eps_next = (np.random.rand(len(eps_idx)) > probs[:, 0]).astype(int)
        
        return Z_next, eps_next
    
    def simulate(self, T, n_agents, Z_init=0, seed=0):
        """
        Simulate the full shock process for T periods.
        
        Parameters:
        -----------
        T : int - Number of periods
        n_agents : int - Number of agents
        Z_init : int - Initial aggregate state
        seed : int - Random seed
        
        Returns:
        --------
        Z_history : ndarray (T,) - Aggregate state history
        eps_history : ndarray (T, n_agents) - Employment state history
        """
        np.random.seed(seed)
        
        Z_history = np.zeros(T, dtype=int)
        eps_history = np.zeros((T, n_agents), dtype=int)
        
        # Initialize
        Z_history[0], eps_history[0] = self.initialize_agents(n_agents, Z_init)
        
        # Simulate forward
        for t in range(1, T):
            Z_history[t], eps_history[t] = self.transition_step(
                Z_history[t-1], eps_history[t-1]
            )
        
        return Z_history, eps_history


# Convenience function
def create_shock_simulator(Pi):
    """Create a ShockSimulator instance."""
    return ShockSimulator(Pi)
