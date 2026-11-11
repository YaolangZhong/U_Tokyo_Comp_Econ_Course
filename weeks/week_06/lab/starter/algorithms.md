# VFI, PFI, and GPI

These workflows transcribe the VFI algorithm from Lecture 3 and PFI/MPI from Lecture 4, with the finite-sweep GPI interpretation demonstrated in Week 5. Numerical defaults and the shared Python interface below specify this exercise.

## Common finite-MDP input

Let $S$ be a finite state set, $A(s)$ a nonempty feasible action set, $r(s,a)$ a deterministic one-period reward, $P(s'\mid s,a)$ the transition kernel, and $0\leq\beta<1$ the discount factor.

$$
Q_V(s,a)=r(s,a)+\beta\sum_{s'\in S}P(s'\mid s,a)V(s'),\qquad
(TV)(s)=\max_{a\in A(s)}Q_V(s,a).
$$

For a deterministic policy $\sigma$, define $r^\sigma(s)=r(s,\sigma(s))$ and $P^\sigma(s,s')=P(s'\mid s,\sigma(s))$. Then

$$
T_\sigma V=r^\sigma+\beta P^\sigma V,\qquad
(I-\beta P^\sigma)V^\sigma=r^\sigma.
$$

Use the array layout from Lab 3. A model object supplies the following attributes; the solvers must not depend on a particular model class.

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `states` | length `N` | State labels |
| `actions` | `(N, A)` | Actual action labels in each table column |
| `feasible` | `(N, A)` | Boolean mask of allowed actions |
| `rewards` | `(N, A)` | One-period rewards |
| `P` | `(N, A, N)` | Current state, action column, next state |
| `beta` | scalar | Discount factor |

Each feasible transition row must be nonnegative and sum to one. Every state has a feasible action. Mask infeasible action values with negative infinity before maximizing. Policies used for array indexing contain action **column indices**; also return the corresponding action labels.

Every evaluation sweep is simultaneous: compute all next values from the same old array. Initialize values to zero and the policy to the first feasible action in each state. In greedy selection, retain the current action when it exactly attains the maximum; otherwise choose the first maximizing column. Do not use a broad approximate-equality test to treat inferior actions as ties.

Use target value error `epsilon=1e-6`, residual threshold $\delta=(1-\beta)\epsilon$, and `max_iter=10000`. The Bellman residual is $\rho(V)=\|TV-V\|_\infty$; $\rho(V)\leq\delta$ guarantees $\|V-V^*\|_\infty\leq\epsilon$. Report failure if an iteration cap is reached.

## 1. Value function iteration (VFI)

0. **Input:** the MDP, $\epsilon$, and maximum iterations.
1. **Initialize:** $V_0=0$, $i=0$, and a feasible policy for tie handling.
2. **Compute action values:** $Q_i=Q_{V_i}$ for all feasible state-action pairs.
3. **Update:** set $V_{i+1}(s)=\max_a Q_i(s,a)$, storing it separately.
4. **Measure:** $d_i=\|V_{i+1}-V_i\|_\infty=\rho(V_i)$.
5. **Stop or repeat:** if $d_i\leq\delta$, continue to Step 6. Otherwise set $V_i\leftarrow V_{i+1}$, increment $i$, and return to Step 2.
6. **Return:** $V_i$ and a policy greedy with respect to $Q_i$.

This matches Lecture 3's return convention: return the old array whose residual was just measured, not the unchecked new array. The Week 5 demonstration instead checks the new array's residual and returns that array; either convention is valid when the check and returned value agree. Use the Lecture 3 convention for this exercise.

## 2. Policy function iteration (PFI)

0. **Input:** the MDP, $\epsilon$, and maximum iterations.
1. **Initialize:** a feasible deterministic policy $\sigma_0$; set $i=0$.
2. **Evaluate fully:** form $r^{\sigma_i}$ and $P^{\sigma_i}$, and solve $(I-\beta P^{\sigma_i})V^{\sigma_i}=r^{\sigma_i}$. Use a linear-system solver, not a matrix inverse.
3. **Compute action values:** $Q_i=Q_{V^{\sigma_i}}$.
4. **Improve:** choose $\sigma_{i+1}$ greedily using the tie rule above.
5. **Stop or repeat:** if the policy is unchanged, return its evaluated value and policy. Otherwise set $\sigma_i\leftarrow\sigma_{i+1}$, increment $i$, and return to Step 2.
6. **Numerical check:** require the returned value's residual to be at most $\delta$. Report a numerical failure if policy stability is reached but this check fails.

One linear solve is full policy evaluation; it is not one evaluation sweep.

## 3. Generalized policy iteration (GPI): finite-sweep MPI

GPI describes the interaction of evaluation and improvement. This exercise implements **modified policy iteration (MPI)** as the concrete GPI algorithm, with `evaluation_sweeps=k` and default $k=3$.

0. **Input:** the MDP, $\epsilon$, positive integer $k$, and maximum iterations.
1. **Initialize:** $V_0=0$, a feasible policy, and $i=0$.
2. **Improve:** choose $\sigma_{i+1}$ greedy with respect to $Q_{V_i}$.
3. **Evaluate partially:** hold $\sigma_{i+1}$ fixed. Set $W_0=V_i$, then perform exactly $k$ simultaneous sweeps:

   $$W_{j+1}=r^{\sigma_{i+1}}+\beta P^{\sigma_{i+1}}W_j,\qquad j=0,\ldots,k-1.$$

4. **Update:** set $V_{i+1}=W_k$ and recompute $Q_{V_{i+1}}$ and $\rho(V_{i+1})$.
5. **Stop or repeat:** if $\rho(V_{i+1})\leq\delta$, return $V_{i+1}$ and a policy greedy with respect to these final values. Otherwise increment $i$ and return to Step 2 using the new values.

An unchanged policy alone is insufficient: its value may still be inaccurate. For $k=1$, the value updates reproduce VFI; the return/check conventions above can cause a one-round difference in reported counts.

## Lecture sources

- [Lecture 3: VFI, frames “Value Function Iteration (VFI) Algorithm” and “VFI Algorithm (continued)”](../../../week_03/lecture/slides/tex/Q_Functions_Optimality_and_Value_Function_Iteration.tex)
- [Lecture 4: PFI algorithm, Modified Policy Iteration, and Generalized Policy Iteration](../../../week_04/lecture/slides/tex/Policy_Function_Iteration.tex)
- [Week 5: VFI/PFI/GPI algorithm recap](../../../week_05/lecture/slides/tex/Maze_Value_and_Policy_Updates.tex)
