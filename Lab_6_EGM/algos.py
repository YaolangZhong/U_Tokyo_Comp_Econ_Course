import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class TISolver:
    def __init__(self, model, tol=1e-6, max_iter=1000, damping=1.0):
        self.model = model
        self.tol = tol
        self.max_iter = max_iter
        self.damping = damping
        
    def solve(self, verbose=True):
        """Time iteration solver"""
        # Initialize at borrowing cap (feasible consumption)
        c = self.model.pi_cap.copy()
        errors = []
        
        if verbose:
            print("Starting Time Iteration method...")
            
        for it in range(self.max_iter):
            c_new = np.zeros_like(c)
            
            # Pre-compute interpolators for the current policy c
            # We need one interpolator per income state w
            interpolators = []
            for n in range(self.model.N):
                # Use linear interpolation with extrapolation
                interp = interp1d(self.model.k_grid, c[:, n], 
                                  kind='linear', fill_value="extrapolate")
                interpolators.append(interp)
            
            # Iterate over all state points
            for m in range(self.model.M):
                for n in range(self.model.N):
                    
                    # Current state
                    k_curr = self.model.k_grid[m]
                    w_curr = self.model.w_grid[n]
                    pi_max = self.model.pi_cap[m, n]
                    
                    # Define the Euler residual function: u'(c) - beta*R*E[u'(c')]
                    def euler_residual(cons):
                        # Implied next capital
                        k_next = self.model.R * k_curr + w_curr - cons
                        
                        # Interpolate next period consumption for each possible w_next
                        # c(k_next, w_next)
                        # We can compute expectation vectorally
                        expected_mu = 0.0
                        for n_prime in range(self.model.N):
                            # Interpolate policy at k_next given w_next state n_prime
                            c_next = interpolators[n_prime](k_next)
                            
                            # Safeguard c_next > 0 for u_prime evaluation
                            # Although extrapolation might yield < 0, Inada usually prevents it
                            # But we clamp to small epsilon just in case
                            c_next = max(c_next, 1e-10)
                            
                            expected_mu += self.model.P[n, n_prime] * self.model.u_prime(c_next)
                            
                        rhs = self.model.beta * self.model.R * expected_mu
                        return self.model.u_prime(cons) - rhs
                    
                    # Solve for root
                    # Check bounds: c in [1e-10, pi_max]
                    # If G(pi_max) > 0, then constraint binds (u'(pi_max) > RHS) => c = pi_max
                    # We should check signs. u'(c) is decreasing. G(c) is decreasing (usually).
                    
                    res_max = euler_residual(pi_max)
                    if res_max >= 0:
                        # Marginal utility at cap is higher than discounted future MU
                        # => Agent wants to consume more but constrained
                        c_new[m, n] = pi_max
                    else:
                        # Interior solution
                        try:
                            # Lower bound close to 0 (Inada condition)
                            c_new[m, n] = brentq(euler_residual, 1e-10, pi_max, xtol=1e-8)
                        except ValueError:
                            # Fallback if no root found (rare if properties hold)
                            c_new[m, n] = pi_max

            # Apply damping
            c_new = (1-self.damping) * c + self.damping * c_new
            
            # Compute error
            error = np.max(np.abs(c_new - c))
            errors.append(error)
            
            if verbose and (it + 1) % 10 == 0:
                print(f"Iteration {it+1}: max error = {error:.2e}")
            
            if error < self.tol:
                if verbose:
                    print(f"Converged after {it+1} iterations!")
                return c_new, np.array(errors)
                
            c = c_new.copy()
        
        if verbose:
            print("Warning: Maximum iterations reached without convergence")
        return c, np.array(errors)


class EGMSolver:
    def __init__(self, model, tol=1e-6, max_iter=1000, damping=1.0):
        self.model = model
        self.tol = tol
        self.max_iter = max_iter
        self.damping = damping
        
    def solve(self, verbose=True):
        """Endogenous Grid Method solver"""
        c = self.model.pi_cap.copy()
        errors = []
        
        if verbose:
            print("Starting Endogenous Grid Method...")
        
        for it in range(self.max_iter):
            # Step 1: Compute Expectation term (RHS of Euler)
            # EGM uses c on the grid points for the expectation
            # c_next(k_grid, w_prime) is just c[:, n_prime]
            
            # Compute marginal utility at all grid points: u'(c_i(k,w))
            uc_next = self.model.u_prime(c) # shape (M, N)
            
            # Expected marginal utility: sum(P * u'(c'))
            # (M, N) dot (N, N)^T -> but P is (N,N) transition from n to n'
            # We want for each (m, n): sum_{n'} P[n, n'] * uc_next[m, n']
            # uc_next is (M, N').
            expect_uc = uc_next @ self.model.P.T 
            # Check dimensions: (M, N) @ (N, N) -> (M, N). Correct.
            
            rhs = self.model.beta * self.model.R * expect_uc
            
            # Step 2: Invert Euler to get endogenous c_tilde and k_tilde
            c_tilde = self.model.u_prime_inv(rhs)
            
            # k_tilde = (k' - w + c_tilde) / R ? No, k' is the grid point k_grid[m]
            # Budget constraint: c + k' = Rk + w => k = (k' + c - w) / R
            # Here k_grid[m] plays the role of k_{t+1}
            # k_tilde[m, n] = (k_grid[m] + c_tilde[m, n] - w_grid[n]) / R
            
            # Reshape k_grid for broadcasting
            k_grid_m = self.model.k_grid[:, np.newaxis] # (M, 1)
            w_grid_n = self.model.w_grid[np.newaxis, :] # (1, N)
            
            k_tilde = (k_grid_m + c_tilde - w_grid_n) / self.model.R
            
            # Step 3: Interpolate back to fixed grid
            c_new = np.zeros_like(c)
            
            for n in range(self.model.N):
                # We have pairs (k_tilde[:, n], c_tilde[:, n])
                # We want to find c_new at self.model.k_grid
                
                # Note: k_tilde is monotonic in m because c_tilde (consumption) 
                # is increasing in future assets (normal goods) and k_grid is increasing
                
                # Using numpy's interp which does linear interpolation
                # We need to handle the constraint k_{t+1} >= b (k_grid[0])
                # If k_fixed < k_tilde[0], it means to get to b (lowest k'), 
                # we need more wealth than we have, or rather, we are constrained.
                # If constrained: c = Rk + w - b (pi_cap)
                
                # interp(x, xp, fp)
                c_interp = np.interp(self.model.k_grid, k_tilde[:, n], c_tilde[:, n], left=np.nan, right=np.nan)
                
                # Handle boundaries
                # For points where k_grid < k_tilde[0], the agent is borrowing constrained
                # The endogenous grid doesn't cover the borrowing constraint region typically
                
                # Check valid region
                mask_constrained = self.model.k_grid < k_tilde[0, n]
                
                c_new[mask_constrained, n] = self.model.pi_cap[mask_constrained, n]
                
                # For interior
                c_new[~mask_constrained, n] = c_interp[~mask_constrained]
                
                # Upper extrapolation (if k_grid > max(k_tilde))
                # Linear extrapolation from last two points or just extend
                # np.interp fills with right (NaN here), we can fill with linear extrap if needed
                # But usually k_tilde covers the space if grid is large enough
                if np.any(np.isnan(c_new[:, n])):
                     # Simple linear extrapolation for upper bound if needed
                     mask_nan = np.isnan(c_new[:, n])
                     slope = (c_tilde[-1, n] - c_tilde[-2, n]) / (k_tilde[-1, n] - k_tilde[-2, n])
                     c_new[mask_nan, n] = c_tilde[-1, n] + slope * (self.model.k_grid[mask_nan] - k_tilde[-1, n])

                # Enforce feasibility (c <= Rk + w - b)
                c_new[:, n] = np.minimum(c_new[:, n], self.model.pi_cap[:, n])
                
            # Apply damping
            c_new = (1-self.damping) * c + self.damping * c_new
            
            error = np.max(np.abs(c_new - c))
            errors.append(error)
            
            if verbose and (it + 1) % 10 == 0:
                print(f"Iteration {it+1}: max error = {error:.2e}")
            
            if error < self.tol:
                if verbose:
                    print(f"Converged after {it+1} iterations!")
                return c_new, np.array(errors)
            
            c = c_new.copy()
            
        if verbose:
            print("Warning: Maximum iterations reached without convergence")
        return c, np.array(errors)


class RolloutSolver:
    def __init__(self, model, tol=1e-6, max_iter=1000, damping=1.0):
        self.model = model
        self.tol = tol
        self.max_iter = max_iter
        self.damping = damping
        
    def solve(self, verbose=True):
        """Forward Rollout (Lagged-Coleman) solver"""
        c = self.model.pi_cap.copy()
        errors = []
        
        if verbose:
            print("Starting Forward Rollout method...")
            
        for it in range(self.max_iter):
            c_new = np.zeros_like(c)
            
            # Pre-compute interpolators for current c
            interpolators = []
            for n in range(self.model.N):
                interpolators.append(interp1d(self.model.k_grid, c[:, n], 
                                            kind='linear', fill_value="extrapolate"))
            
            # Step 1: Forward rollout - compute next capital under OLD policy
            k_grid_m = self.model.k_grid[:, np.newaxis]
            w_grid_n = self.model.w_grid[np.newaxis, :]
            k_next_mat = self.model.R * k_grid_m + w_grid_n - c
            
            # Step 2: Update consumption via Euler equation
            # c_new = (beta * R * E[u'(c(k', w'))])^(-1/sigma)
            
            # We iterate to compute expectation
            for m in range(self.model.M):
                for n in range(self.model.N):
                    k_next = k_next_mat[m, n]
                    
                    expected_mu = 0.0
                    for n_prime in range(self.model.N):
                        c_next_val = interpolators[n_prime](k_next)
                        c_next_val = max(c_next_val, 1e-10) # Safeguard
                        expected_mu += self.model.P[n, n_prime] * self.model.u_prime(c_next_val)
                    
                    rhs = self.model.beta * self.model.R * expected_mu
                    c_target = self.model.u_prime_inv(rhs)
                    
                    # Enforce borrowing constraint
                    c_new[m, n] = min(c_target, self.model.pi_cap[m, n])
            
            # Apply damping
            c_new = (1-self.damping) * c + self.damping * c_new
            
            error = np.max(np.abs(c_new - c))
            errors.append(error)
            
            if verbose and (it + 1) % 10 == 0:
                print(f"Iteration {it+1}: max error = {error:.2e}")
            
            if error < self.tol:
                if verbose:
                    print(f"Converged after {it+1} iterations!")
                return c_new, np.array(errors)
            
            c = c_new.copy()
            
        if verbose:
            print("Warning: Maximum iterations reached without convergence")
        return c, np.array(errors)
