"""
Actor-Critic for Aiyagari Model (DP-style)

Following the lecture:
1. Critic: Q(s,a;φ) trained via Bellman equation (policy evaluation)
2. Actor: σ(s;θ) trained to maximize Q(s, σ(s))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
from model import AiyagariParams, StateNormalizer


# =============================================================================
# Neural Networks
# =============================================================================

class Actor(nn.Module):
    """Policy network: state -> action (savings rate in [0,1])"""
    
    def __init__(self, state_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    """Q-network: (state, action) -> Q-value"""
    
    def __init__(self, state_dim: int = 2, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# =============================================================================
# VFI Solver (Benchmark)
# =============================================================================

def make_grid(a_min: float, a_max: float, n: int, curv: float = 1.0) -> np.ndarray:
    """
    Create asset grid with optional curvature.
    
    curv = 1.0: uniform grid
    curv > 1.0: more points near a_min (borrowing constraint)
    curv < 1.0: more points near a_max
    
    Uses: a_grid = a_min + (a_max - a_min) * (x ** curv), where x ∈ [0,1] uniform
    """
    x = np.linspace(0, 1, n)
    return a_min + (a_max - a_min) * (x ** curv)


class VFISolver:
    """Standard Value Function Iteration"""
    
    def __init__(self, params: AiyagariParams, z_grid: np.ndarray, P_z: np.ndarray,
                 n_a: int = 100, tol: float = 1e-6, max_iter: int = 1000, grid_curv: float = 4.0):
        self.params = params
        self.z_grid = z_grid
        self.P_z = P_z
        self.n_a = n_a
        self.tol = tol
        self.max_iter = max_iter
        # Polynomial grid: denser near borrowing constraint (a_min)
        self.a_grid = make_grid(params.a_min, params.a_max, n_a, curv=grid_curv)
        self.n_z = len(z_grid)
    
    def utility(self, c):
        gamma = self.params.gamma
        c = np.maximum(c, 1e-10)
        if gamma == 1:
            return np.log(c)
        return (c ** (1 - gamma) - 1) / (1 - gamma)
    
    def solve(self, verbose=True):
        V = np.zeros((self.n_a, self.n_z))
        policy = np.zeros((self.n_a, self.n_z))
        
        if verbose:
            print("Running VFI...")
        
        for it in range(self.max_iter):
            V_new = np.zeros_like(V)
            
            for i_a, a in enumerate(self.a_grid):
                for i_z, z in enumerate(self.z_grid):
                    cash = (1 + self.params.r) * a + self.params.w * z
                    
                    best_val, best_ap = -np.inf, self.a_grid[0]
                    for i_ap, ap in enumerate(self.a_grid):
                        if ap > cash:
                            break
                        c = cash - ap
                        if c > 0:
                            EV = self.P_z[i_z] @ V[i_ap, :]
                            val = self.utility(c) + self.params.beta * EV
                            if val > best_val:
                                best_val, best_ap = val, ap
                    
                    V_new[i_a, i_z] = best_val
                    policy[i_a, i_z] = best_ap
            
            diff = np.max(np.abs(V_new - V))
            V = V_new
            
            if verbose and (it + 1) % 50 == 0:
                print(f"  Iter {it+1}: diff = {diff:.2e}")
            
            if diff < self.tol:
                if verbose:
                    print(f"  Converged in {it+1} iterations")
                break
        
        return self.a_grid, V, policy


# =============================================================================
# Actor-Critic Solver (DP-style, following lecture notes)
# =============================================================================

class ActorCriticSolver:
    """
    Actor-Critic with DP-style updates.
    
    Critic: Q(s,a;φ) trained via Bellman equation
        L_φ = E[(Q(s,a;φ) - (r + β E[Q(s',a';φ_target)]))²]
    
    Actor: σ(s;θ) trained to maximize Q
        L_θ = -E[Q(s, σ(s;θ); φ)]
    """
    
    def __init__(self, params: AiyagariParams, z_grid: np.ndarray, P_z: np.ndarray,
                 n_a: int = 100, hidden_dim: int = 64, lr: float = 1e-3, grid_curv: float = 4.0):
        self.params = params
        self.z_grid = z_grid
        self.P_z = P_z
        self.n_z = len(z_grid)
        self.n_a = n_a
        # Polynomial grid: denser near borrowing constraint
        self.a_grid = make_grid(params.a_min, params.a_max, n_a, curv=grid_curv)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalizer = StateNormalizer(params.a_min, params.a_max, z_grid)
        
        # Networks
        self.actor = Actor(hidden_dim=hidden_dim).to(self.device)
        self.critic = Critic(hidden_dim=hidden_dim).to(self.device)
        
        # Target network for critic only (stabilizes Bellman target)
        self.critic_target = Critic(hidden_dim=hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Precompute grid data
        self._setup_grid()
        
        # Transition matrix as tensor
        self.P_z_tensor = torch.FloatTensor(P_z).to(self.device)
    
    def _setup_grid(self):
        """Precompute all (a, z) grid states."""
        states, a_vals, z_idxs, cash_vals = [], [], [], []
        
        for i_z in range(self.n_z):
            for i_a, a in enumerate(self.a_grid):
                z = self.z_grid[i_z]
                cash = (1 + self.params.r) * a + self.params.w * z
                s_norm = self.normalizer.normalize(np.array([a]), np.array([i_z]))[0]
                
                states.append(s_norm)
                a_vals.append(a)
                z_idxs.append(i_z)
                cash_vals.append(cash)
        
        self.states = torch.FloatTensor(np.array(states)).to(self.device)
        self.a_vals = np.array(a_vals)
        self.z_idxs = np.array(z_idxs)
        self.cash = torch.FloatTensor(cash_vals).to(self.device)
        self.n_grid = len(states)
    
    def _utility(self, c):
        """CRRA utility."""
        gamma = self.params.gamma
        c = torch.clamp(c, min=1e-10)
        if gamma == 1:
            return torch.log(c)
        return (c ** (1 - gamma) - 1) / (1 - gamma)
    
    def _get_next_states(self, a_next):
        """
        For each grid state and each possible next z', 
        compute the normalized next state (a', z').
        
        Returns: next_states (n_grid, n_z, 2)
        """
        a_next_np = a_next.detach().cpu().numpy()
        next_states = np.zeros((self.n_grid, self.n_z, 2))
        
        for i_z_next in range(self.n_z):
            z_idx_next = np.full(self.n_grid, i_z_next)
            next_states[:, i_z_next, :] = self.normalizer.normalize(a_next_np, z_idx_next)
        
        return torch.FloatTensor(next_states).to(self.device)
    
    def _compute_expected_Q(self, a_next):
        """
        Compute E[Q(s', a') | s] where:
        - s' = (a', z') with z' drawn from P(z'|z)
        - a' = actor(s')  (current actor, not target)
        
        Uses target CRITIC for stability, current ACTOR for next action.
        """
        # Get normalized next states for all possible z'
        next_states = self._get_next_states(a_next)  # (n_grid, n_z, 2)
        
        # Compute Q(s', actor(s')) for each possible z'
        Q_next = torch.zeros(self.n_grid, self.n_z, device=self.device)
        
        for i_z_next in range(self.n_z):
            s_next = next_states[:, i_z_next, :]  # (n_grid, 2)
            with torch.no_grad():
                a_next_policy = self.actor(s_next)  # current actor
                Q_next[:, i_z_next] = self.critic_target(s_next, a_next_policy).squeeze(-1)
        
        # E[Q(s',a')] = sum over z' of P(z'|z) * Q(s', a')
        expected_Q = torch.zeros(self.n_grid, device=self.device)
        for i_z in range(self.n_z):
            mask = (self.z_idxs == i_z)
            expected_Q[mask] = Q_next[mask] @ self.P_z_tensor[i_z]
        
        return expected_Q
    
    def _update_critic(self, stochastic: bool = False, n_action_grid: int = 10, noise_std: float = 0.1):
        """
        Critic update via Bellman equation.
        
        Two modes:
        - stochastic=False (default): Train on grid of actions (Fitted Q-Iteration)
        - stochastic=True: Train on actor's action + noise (DDPG-style)
        """
        if stochastic:
            return self._update_critic_stochastic(noise_std)
        else:
            return self._update_critic_grid(n_action_grid)
    
    def _update_critic_grid(self, n_action_grid: int = 10):
        """
        Grid-based critic update (deterministic DP-style).
        
        For each (state, action) pair on a grid:
            y = r(s,a) + β E[Q(s', actor(s'))]
            Train Q(s,a) → y
        """
        action_grid = torch.linspace(0.01, 0.99, n_action_grid, device=self.device)
        
        total_loss = 0.0
        
        for action_val in action_grid:
            action = torch.full((self.n_grid, 1), action_val, device=self.device)
            a_next = (action.squeeze() * self.cash).clamp(self.params.a_min, self.params.a_max)
            
            c = self.cash - a_next
            reward = self._utility(c)
            
            with torch.no_grad():
                expected_Q_next = self._compute_expected_Q(a_next)
                target = reward + self.params.beta * expected_Q_next
            
            Q_current = self.critic(self.states, action).squeeze(-1)
            total_loss = total_loss + nn.MSELoss()(Q_current, target)
        
        loss = total_loss / n_action_grid
        
        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()
        
        return loss.item()
    
    def _update_critic_stochastic(self, noise_std: float = 0.1):
        """
        Stochastic critic update (DDPG-style with exploration noise).
        
        Train on actor's action + Gaussian noise.
        """
        with torch.no_grad():
            action = self.actor(self.states)
            noise = torch.randn_like(action) * noise_std
            action_noisy = (action + noise).clamp(0.0, 1.0)
        
        a_next = (action_noisy.squeeze() * self.cash).clamp(self.params.a_min, self.params.a_max)
        
        c = self.cash - a_next
        reward = self._utility(c)
        
        with torch.no_grad():
            expected_Q_next = self._compute_expected_Q(a_next)
            target = reward + self.params.beta * expected_Q_next
        
        Q_current = self.critic(self.states, action_noisy).squeeze(-1)
        loss = nn.MSELoss()(Q_current, target)
        
        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()
        
        return loss.item()
    
    def _update_actor(self):
        """
        Actor update: maximize Q(s, actor(s))
        
        L_θ = -E[Q(s, σ(s;θ); φ)]
        """
        # Get action from actor (with gradient)
        action = self.actor(self.states)
        
        # Q-value (critic provides gradient signal)
        Q = self.critic(self.states, action)
        
        # Loss: negative Q (we want to maximize Q)
        loss = -Q.mean()
        
        self.opt_actor.zero_grad()
        loss.backward()
        self.opt_actor.step()
        
        return loss.item()
    
    def _soft_update(self, target, source, tau=0.01):
        """Polyak averaging for target networks."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
    
    def solve(self, n_iterations=500, critic_updates=5, actor_updates=1,
              tau=0.005, stochastic=False, verbose=True):
        """
        Train actor-critic.
        
        Parameters
        ----------
        stochastic : bool
            False (default): Grid-based critic training (deterministic DP-style)
            True: Noisy action critic training (DDPG-style)
        
        Returns: critic_losses, actor_losses
        """
        critic_losses = []
        actor_losses = []
        
        mode = "stochastic" if stochastic else "grid-based"
        if verbose:
            print(f"Training Actor-Critic ({mode})...")
        
        for it in range(n_iterations):
            # Update critic (multiple times for stability)
            c_loss = 0
            for _ in range(critic_updates):
                c_loss += self._update_critic(stochastic=stochastic)
            c_loss /= critic_updates
            
            # Update actor
            a_loss = 0
            for _ in range(actor_updates):
                a_loss += self._update_actor()
            a_loss /= actor_updates
            
            # Soft update target critic
            self._soft_update(self.critic_target, self.critic, tau)
            
            critic_losses.append(c_loss)
            actor_losses.append(a_loss)
            
            if verbose and (it + 1) % 50 == 0:
                print(f"  Iter {it+1}: Critic Loss={c_loss:.4f}, Actor Loss={a_loss:.4f}")
        
        if verbose:
            print("Training complete!")
        
        return critic_losses, actor_losses
    
    def get_policy(self, a_grid, z_idx):
        """Get policy a'(a) for given z."""
        n = len(a_grid)
        z = self.z_grid[z_idx]
        cash = (1 + self.params.r) * a_grid + self.params.w * z
        
        s_norm = self.normalizer.normalize(a_grid, np.full(n, z_idx))
        s_tensor = torch.FloatTensor(s_norm).to(self.device)
        
        with torch.no_grad():
            rate = self.actor(s_tensor).cpu().numpy().squeeze()
        
        a_next = rate * cash
        return np.clip(a_next, self.params.a_min, np.minimum(cash - 1e-6, self.params.a_max))
