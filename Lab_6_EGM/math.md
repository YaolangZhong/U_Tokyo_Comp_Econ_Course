# Households' Dynamic Optimization Problem

The representative household chooses $\{c_t,k_{t+1}\}$ to maximize lifetime utility:
  $$
    \max_{\{c_t,k_{t+1}\}_{t\ge0}}
    \sum_{t=0}^{\infty}\beta^t u(c_t)
    \quad\text{s.t.}\quad
    c_t + k_{t+1} = R_tk_t + w_t, \quad k_{t+1}\ge b.
  $$

where $R_t = 1 + r_t$ and $P(w_{t+1} | w_t)$ is taken as given to agent

- State $k_t, w_t$ determine the income
- Controls $c_t, k_{t+1}$ determine the expenditure. We pick $c_t = \pi(k_t, w_t)$ and then $k_{t+1} = R_tk_t + w_t - \pi(k_t, w_t)$


Inputs to the algorithms:

- Utility $u(c)=\tfrac{c^{1-\sigma}}{1-\sigma}$, marginal utility $u_c(c)=c^{-\sigma}$
- Parameters $\beta, R, \sigma, b$; transition matrix $P=(P_{nn'})$
- Grids: $\mathbf{\bar{k}}=\{\bar{k}_1,\ldots,\bar{k}_M\}$ with $\bar{k}_1=b$, and $\mathbf{\bar{w}}=\{\bar{w}_1,\ldots,\bar{w}_N\}$.

---

# Algorithms


## Time Iteration (TI):

### 1. Prerequisites
- Utility $u(c)=\tfrac{c^{1-\sigma}}{1-\sigma}$, marginal utility $u_c(c)=c^{-\sigma}$.
- Parameters $\beta, R, \sigma, b$; transition matrix $P=(P_{nn'})$.
- Grids: $\mathbf{\bar{k}}=\{\bar{k}_1,\ldots,\bar{k}_M\}$ with $\bar{k}_1=b$, and $\mathbf{\bar{w}}=\{\bar{w}_1,\ldots,\bar{w}_N\}$.
- Solver hyperparameters: MaxIter, tolerance $\varepsilon$.



### 2. Initialize
- Borrowing cap (elementwise):
  $$
  \bar{\pi}_{m,n}:=R\,\bar{k}_m+\bar{w}_n-b.
  $$
- Set $c^{(0)}_{m,n}=\bar{\pi}_{m,n}$ for all $(m,n)$.  
  This defines $\pi^{(0)}(k,w;\mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(0)})$.


### 3. Iterate for $i=0,1,\ldots,\text{MaxIter}-1$
- For each node $(m,n)$, define the residual and an NCP reformulation:

$$
\begin{aligned}
G_{m,n}(c)
&:= u_c(c)
   - \beta R \sum_{n'=1}^{N} P_{nn'}\,
      u_c\!\Big(\pi^{(i)}(R\bar{k}_m+\bar{w}_n-c,\ \bar{w}_{n'};\ \mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(i)})\Big),\\[0.25em]
\Psi_{m,n}(c)
&:= \sqrt{\,G_{m,n}(c)^2+(\bar{\pi}_{m,n}-c)^2\,}
    - G_{m,n}(c) - (\bar{\pi}_{m,n}-c).
\end{aligned}
$$


- Solve for $c^{(i+1)}_{m,n}\in[0,\bar{\pi}_{m,n}]$ using either:
  - Bracketing (bisection/Brent) on $G_{m,n}(c)=0$ with clipping at $\bar{\pi}_{m,n}$, or
  - Semismooth Newton on the Fischer–Burmeister equation $\Psi_{m,n}(c)=0$.
- Then set:
$$
k'_{m,n}=R\,\bar{k}_m+\bar{w}_n-c^{(i+1)}_{m,n}.
$$



### 4. Convergence
- If $\displaystyle \max_{m,n}\big|c^{(i+1)}_{m,n}-c^{(i)}_{m,n}\big|<\varepsilon$, stop.
- Otherwise, form $\pi^{(i+1)}(k,w;\mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(i+1)})$ and continue.

---

## Endogenous Grid Method (EGM):

### 1. Prerequisites
- Utility $u(c)=\tfrac{c^{1-\sigma}}{1-\sigma}$, marginal utility $u_c(c)=c^{-\sigma}$.
- Parameters $\beta, R, \sigma, b$; transition matrix $P=(P_{nn'})$.
- Grids: $\mathbf{\bar{k}}=\{\bar{k}_1,\ldots,\bar{k}_M\}$ with $\bar{k}_1=b$, and $\mathbf{\bar{w}}=\{\bar{w}_1,\ldots,\bar{w}_N\}$.
- Solver hyperparameters: MaxIter, tolerance $\varepsilon$.


### 2. Initialize
- Borrowing cap (elementwise):
  $$
  \bar{\pi}_{m,n}:=R\,\bar{k}_m+\bar{w}_n-b.
  $$
- Set $c^{(0)}_{m,n}=\bar{\pi}_{m,n}$ for all $(m,n)$.  
  This defines $\pi^{(0)}(k,w;\mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(0)})$.


### 3. Iterate for $i=0,1,\ldots,\text{MaxIter}-1$
- Backward consumption for $\mathbf{\tilde{c}}^{(i+1)}$. For each $(m',n)$:
  $$
  \tilde{c}^{(i+1)}_{m',n}
  =
  u_c^{-1}\Big(
    \beta R \sum_{n'=1}^{N} P_{nn'}\,
    u_c\big(\pi^{(i)}(\bar{k}_{m'},\,\bar{w}_{n'};\,\mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(i)})\big)
  \Big).
  $$
- Implied current state $\mathbf{\tilde{k}}$. For each $(m',n)$:
  $$
  \tilde{k}^{(i+1)}_{m',n}=\frac{\bar{k}_{m'}-\bar{w}_n+\tilde{c}^{(i+1)}_{m',n}}{R}.
  $$


- Interpolation. For each $(m,n)$:
  - If $\bar{k}_m < \tilde{k}^{(i+1)}_{1,n}$: set $c^{(i+1)}_{m,n}=\bar{\pi}_{m,n}$.
  - Else if $\bar{k}_m > \tilde{k}^{(i+1)}_{M,n}$:  
    use monotone linear extrapolation from the last two pairs  
    $(\tilde{k}^{(i+1)}_{M-1,n},\tilde{c}^{(i+1)}_{M-1,n})$, $(\tilde{k}^{(i+1)}_{M,n},\tilde{c}^{(i+1)}_{M,n})$,  
    or clamp to $c^{(i+1)}_{m,n}=\tilde{c}^{(i+1)}_{M,n}$.
  - Else (interior): find $m'$ such that $\tilde{k}^{(i+1)}_{m',n}\le \bar{k}_m \le \tilde{k}^{(i+1)}_{m'+1,n}$, then
    $$
    \lambda(\bar{k}_m)=\frac{\bar{k}_m-\tilde{k}^{(i+1)}_{m',n}}{\tilde{k}^{(i+1)}_{m'+1,n}-\tilde{k}^{(i+1)}_{m',n}},
    \quad
    c^{(i+1)}_{m,n}=(1-\lambda)\,\tilde{c}^{(i+1)}_{m',n}+\lambda\,\tilde{c}^{(i+1)}_{m'+1,n}.
    $$
  - (Optional) Enforce cap: $c^{(i+1)}_{m,n}\leftarrow \min\{c^{(i+1)}_{m,n},\,\bar{\pi}_{m,n}\}$.
- (Optional) Damping: $c^{(i+1)}\leftarrow (1-\alpha)c^{(i)}+\alpha c^{(i+1)}$, with $\alpha\in(0,1]$.
- Set $k'_{m,n}=R\,\bar{k}_m+\bar{w}_n-c^{(i+1)}_{m,n}$.


### 4. Convergence

- If $\max_{m,n}\lvert c^{(i+1)}_{m,n}-c^{(i)}_{m,n}\rvert<\varepsilon$, stop.
- Otherwise, form $\pi^{(i+1)}(k,w;\mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(i+1)})$ and continue.


--- 


## Forward rollout (lagged-Coleman explicit update):

### 1. Prerequisites
- Utility $u(c)=\tfrac{c^{1-\sigma}}{1-\sigma}$, marginal utility $u_c(c)=c^{-\sigma}$.
- Parameters $\beta, R, \sigma, b$; transition matrix $P=(P_{nn'})$.
- Grids: $\mathbf{\bar{k}}=\{\bar{k}_1,\ldots,\bar{k}_M\}$ with $\bar{k}_1=b$, and $\mathbf{\bar{w}}=\{\bar{w}_1,\ldots,\bar{w}_N\}$.
- Solver hyperparameters: MaxIter, tolerance $\varepsilon$.



### 2. Initialize
- Borrowing cap (elementwise):
  $$
  \bar{\pi}_{m,n}:=R\,\bar{k}_m+\bar{w}_n-b.
  $$
- Set $c^{(0)}_{m,n}=\bar{\pi}_{m,n}$ for all $(m,n)$.  
  This defines $\pi^{(0)}(k,w;\mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(0)})$.


### 3. Iterate for $i=0,1,\ldots,\text{MaxIter}-1$

- For each node $(m,n)$, form the next state using the **old policy (explicit rollout)**:
  $$
  k^{\prime\,(i)}_{m,n} := R\,\bar{k}_m+\bar{w}_n - c^{(i)}_{m,n}.
  $$
- Compute the unconstrained updated consumption via the Euler RHS evaluated under $\pi^{(i)}$:
  $$
  \tilde c^{(i+1)}_{m,n}
  =
  u_c^{-1}\!\Big(
    \beta R \sum_{n'=1}^{N} P_{nn'}\;
    u_c\!\big(\pi^{(i)}(k^{\prime\,(i)}_{m,n},\,\bar{w}_{n'};\,\mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(i)})\big)
  \Big).
  $$
- Enforce feasibility on today’s grid (borrowing cap):
  $$
  c^{(i+1)}_{m,n} = \min\{\tilde c^{(i+1)}_{m,n},\; \bar{\pi}_{m,n}\}.
  $$
- (Optional) Damping: $c^{(i+1)} \leftarrow (1-\alpha)c^{(i)} + \alpha\,c^{(i+1)}$, with $\alpha\in(0,1]$.
- Set $k'_{m,n}=R\,\bar{k}_m+\bar{w}_n-c^{(i+1)}_{m,n}$.


### 4. Convergence
- If $\max_{m,n}\lvert c^{(i+1)}_{m,n}-c^{(i)}_{m,n}\rvert<\varepsilon$, stop.
- Otherwise, form $\pi^{(i+1)}(k,w;\mathbf{\bar{k}},\mathbf{\bar{w}},\mathbf{c}^{(i+1)})$ and continue.


# Request

base on the math.md, use the information about the eocnonics model as well as the pesudo-code for TI, EGM, Rollout, write me the following under the Lab_6_EGM folder:

- the model.py script for the  consumption saving model class 
- the algos.py script for the three solver classes
- the script plot.py that can plot the output policies of all three algorithms for comparison
- give me the code snippet that i can paste into the .ipynb file for the example experiment with all algorithms solve the same example class of model, where the parameters are chosen in the typical values as in the economics literature