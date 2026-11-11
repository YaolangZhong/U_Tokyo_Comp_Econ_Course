Read algorithms.md and mccall_model.md. Implement the algorithms and solve
the McCall job-search model using the baseline configuration.

Create three Python files:

1. solvers.py
   Implement three algorithm classes: VFI, PFI, and GPI, each with a
   solve(model) method. Follow the pseudocode and common MDP interface in
   algorithms.md. The solvers should work with any finite discounted MDP
   using that interface, independently of the McCall model.
   Allow the convergence tolerance and maximum iterations to be configured.
   For GPI, allow the number of policy evaluation sweeps to be configured,
   with a default of 3. Return the value function, policy, convergence
   information, iteration counts, and runtime.

2. mccall.py
   Implement a McCallModel class from the economic description in
   mccall_model.md. Use its baseline parameters as configurable defaults.
   Represent the model using the common MDP interface in algorithms.md.
   Keep the model definition separate from the solution algorithms.

3. run.py
   Construct the baseline McCall model and run all three algorithms with
   the same convergence tolerance. Compare their state values, policies,
   Bellman residuals, iteration counts, and runtimes. Distinguish policy
   evaluation sweeps from full linear solves when reporting work.
   Report the accepted wage offers and the lowest accepted grid wage.
   Print a clear comparison and save the state values and policies to
   results/comparison.csv.

Use NumPy and the Python standard library. Keep the code readable and
avoid unnecessary dependencies. Check that the returned policies are
feasible, the solutions satisfy the Bellman equation within the requested
tolerance, and the three algorithms agree within numerical accuracy.
Investigate and explain any discrepancies.

Run python run.py, inspect the output, and fix any errors. Report the
command used, the numerical comparison, and whether the checks passed.
If you cannot execute the code, state that clearly.
