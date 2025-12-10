"""
Consumption-Saving Model for Simulation-Based Learning

A discrete-state consumption-saving problem suitable for comparing
DP (VFI) with simulation-based methods (TD, MC).
"""

import numpy as np


class ConsumptionSavingModel:
    """
    Consumption-saving model with discrete assets and income.
    
    State: (k, w) where k is asset level, w is income shock
    Action: c (consumption), which implies next asset k' = R*k + w - c
    
    We discretize both the asset grid and the action space (savings choices).
    """
    
    def __init__(
        self,
        beta: float = 0.96,
        R: float = 1.03,
        sigma: float = 2.0,
        b: float = 0.0,
        k_grid: np.ndarray = None,
        w_grid: np.ndarray = None,
        P: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        beta : float
            Discount factor
        R : float
            Gross interest rate (1 + r)
        sigma : float
            CRRA risk aversion parameter
        b : float
            Borrowing constraint (minimum asset level)
        k_grid : array
            Grid for assets
        w_grid : array
            Grid for income states
        P : array (N x N)
            Markov transition matrix for income
        """
        self.beta = beta
        self.R = R
        self.sigma = sigma
        self.b = b
        
        # Default grids if not provided
        if k_grid is None:
            k_grid = np.linspace(b, 10.0, 50)
        if w_grid is None:
            w_grid = np.array([0.5, 1.5])  # Low and high income
        if P is None:
            # Simple persistent income process (2 states)
            P = np.array([
                [0.8, 0.2],
                [0.2, 0.8]
            ])
        
        self.k_grid = k_grid
        self.w_grid = w_grid
        self.P = P
        
        self.n_k = len(k_grid)
        self.n_w = len(w_grid)
        self.n_states = self.n_k * self.n_w
        
        # Build state indexing
        self._build_state_space()
        
        # Build action space and transition structure
        self._build_action_space()
    
    def _build_state_space(self):
        """Create mappings between (k_idx, w_idx) and flat state index."""
        self.state_to_idx = {}
        self.idx_to_state = {}
        
        idx = 0
        for ik in range(self.n_k):
            for iw in range(self.n_w):
                self.state_to_idx[(ik, iw)] = idx
                self.idx_to_state[idx] = (ik, iw)
                idx += 1
    
    def _build_action_space(self):
        """
        Build discrete action space.
        
        Action = index of next period's asset level k'.
        For each state (k, w), feasible actions are k' such that:
            c = R*k + w - k' >= 0  (non-negative consumption)
            k' >= b               (borrowing constraint)
        """
        self.n_actions = self.n_k  # Each action corresponds to choosing k'
        
        # Precompute feasibility and rewards
        # R[s, a] = u(c) where c = R*k + w - k'
        # P_transition[s, a, s'] = P(w' | w) if k' matches, else 0
        
        self.rewards = np.full((self.n_states, self.n_actions), -np.inf)
        self.transitions = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.feasible = np.zeros((self.n_states, self.n_actions), dtype=bool)
        
        for s in range(self.n_states):
            ik, iw = self.idx_to_state[s]
            k = self.k_grid[ik]
            w = self.w_grid[iw]
            cash = self.R * k + w
            
            for a in range(self.n_actions):  # a = index of k'
                k_next = self.k_grid[a]
                c = cash - k_next
                
                if c > 1e-10 and k_next >= self.b:
                    self.feasible[s, a] = True
                    self.rewards[s, a] = self.u(c)
                    
                    # Transition: next state depends on k' (chosen) and w' (stochastic)
                    for iw_next in range(self.n_w):
                        s_next = self.state_to_idx[(a, iw_next)]
                        self.transitions[s, a, s_next] = self.P[iw, iw_next]
    
    def u(self, c):
        """CRRA utility function."""
        if self.sigma == 1:
            return np.log(np.maximum(c, 1e-10))
        else:
            return (np.maximum(c, 1e-10) ** (1 - self.sigma) - 1) / (1 - self.sigma)
    
    def u_prime(self, c):
        """Marginal utility."""
        return np.maximum(c, 1e-10) ** (-self.sigma)
    
    def get_consumption(self, s, a):
        """Get consumption given state s and action a."""
        ik, iw = self.idx_to_state[s]
        k = self.k_grid[ik]
        w = self.w_grid[iw]
        k_next = self.k_grid[a]
        return self.R * k + w - k_next
    
    def get_asset_and_income(self, s):
        """Get (k, w) values for state s."""
        ik, iw = self.idx_to_state[s]
        return self.k_grid[ik], self.w_grid[iw]
    
    def sample_next_state(self, s, a, rng=None):
        """
        Sample next state given current state and action.
        
        Parameters
        ----------
        s : int
            Current state index
        a : int
            Action index (next asset choice)
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        s_next : int
            Next state index
        reward : float
            Immediate reward u(c)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        ik, iw = self.idx_to_state[s]
        
        # Sample next income state
        iw_next = rng.choice(self.n_w, p=self.P[iw, :])
        
        # Next asset is determined by action
        ik_next = a
        
        s_next = self.state_to_idx[(ik_next, iw_next)]
        reward = self.rewards[s, a]
        
        return s_next, reward
    
    def simulate_episode(self, policy, s0=None, max_steps=1000, rng=None):
        """
        Simulate one episode following a given policy.
        
        Parameters
        ----------
        policy : array of shape (n_states,)
            Policy mapping states to actions
        s0 : int, optional
            Initial state. If None, sample randomly.
        max_steps : int
            Maximum number of steps
        rng : np.random.Generator, optional
            
        Returns
        -------
        states : list
            Sequence of visited states
        actions : list
            Sequence of actions taken
        rewards : list
            Sequence of rewards received
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if s0 is None:
            s0 = rng.integers(0, self.n_states)
        
        states = [s0]
        actions = []
        rewards = []
        
        s = s0
        for _ in range(max_steps):
            a = policy[s]
            s_next, r = self.sample_next_state(s, a, rng)
            
            actions.append(a)
            rewards.append(r)
            states.append(s_next)
            
            s = s_next
        
        return states, actions, rewards


def create_default_model():
    """Create a model instance with default parameters."""
    k_grid = np.linspace(0.0, 15.0, 30)
    w_grid = np.array([0.5, 1.5])  # Low and high income
    P = np.array([
        [0.8, 0.2],
        [0.2, 0.8]
    ])
    
    return ConsumptionSavingModel(
        beta=0.96,
        R=1.03,
        sigma=2.0,
        b=0.0,
        k_grid=k_grid,
        w_grid=w_grid,
        P=P
    )
