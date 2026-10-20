# Lab 3: Value Function Iteration in VS Code

Download and unzip the **Lab 3 working folder** from the course website. Open the extracted folder in VS Code with **File → Open Folder**. Keep the notebook and all Python files together. If you cloned the course repository, open `weeks/week_03/lab/starter`.

## Setup

1. Install Python and VS Code, then install Microsoft's **Python** and **Jupyter** extensions in VS Code.
2. Open **Terminal → New Terminal**. Create an environment and install the packages:

   **macOS / Linux**
   ```sh
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

   **Windows (PowerShell)**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Open `lab_03.ipynb`. Use **Select Kernel → Python Environments** and select `.venv`.
4. Choose **Run All**. The values and figures appear below the cells.

The notebook imports the adjacent `.py` files. After editing a script, save it, restart the notebook kernel, and run the cells again.

## Files

| File | Purpose |
|---|---|
| `lab_03.ipynb` | Set parameters, run experiments, display results |
| `mdp.py` | Shared MDP input: states, actions, rewards, transitions, discount factor |
| `vfi.py` | VFI Steps 0–7, optional initial values, convergence diagnostics, simulation |
| `maze.py` | Lecture maze, rewards, and transitions |
| `mccall.py` | Job-search environment with states `(U, w)` and `(E, w)` |
| `saving.py` | Continuous saving model converted to numerical MDP tables |
| `figures.py` | Plotting helpers |
| `requirements.txt` | Packages installed in `.venv` |

## Shared MDP interface

All three model classes inherit `MDP` from `mdp.py`. For `n` states and up to `m` actions per state:

| Input | Shape | Meaning |
|---|---|---|
| `states` | length `n` | State labels; array rows use Python indices |
| `actions` | `(n, m)` | Actual action represented by each column |
| `feasible` | `(n, m)` | Boolean mask for allowed actions |
| `rewards` | `(n, m)` | Immediate rewards |
| `P` | `(n, m, n)` | Transition probabilities over next states |
| `beta` | scalar | Discount factor strictly between 0 and 1 |

Every feasible transition row sums to one. Every state has at least one feasible action. The solver computes the action values, maximization, and stopping check; model files contain no VFI or Bellman-update routines. Dense transition tables make this interface easy to inspect for these small examples; their storage grows as `n × m × n`.

## Run the same solver on each model

```python
from maze import Maze
from mccall import McCallModel
from saving import ConsumptionSavingModel
from vfi import value_iteration

maze_result = value_iteration(Maze())
search_result = value_iteration(McCallModel())
saving_result = value_iteration(ConsumptionSavingModel(), tolerance=1e-5)
```

Values default to zero for **every state**. To supply an initial vector:

```python
import numpy as np

model = McCallModel()
V0 = np.zeros(len(model.states))
V0[model.n:] = model.wages / (1 - model.beta)
result = value_iteration(model, initial_V=V0)
```

`initial_V` is copied, so the caller's array is preserved. `result["policy"][i]` is the chosen **action**, while `result["policy_indices"][i]` is its table column. Maze actions are codes 0–3 (U, D, L, R), McCall actions are strings, and saving actions are amounts of goods.

The returned dictionary also contains `values`, `history`, `residuals`, `iterations` (Bellman evaluations), `updates` (completed replacements of the value vector), `seconds` (solver time), and `error_bound`. The tolerance targets value error, not policy loss. In accordance with the lecture algorithm, the solver returns the old value vector used to compute the final action values and policy. Failure to converge within `max_iter` raises an error.

## Model interpretation

Begin with the maze, then solve McCall job search and continuous saving. The notebook compares values, policies, convergence, and runtime, and simulates recovered policies using Lab 2's `step(state, action, rng)` pattern.

For saving, the economic state and action remain continuous. The solver receives a numerical approximation: stock and action grids, midpoint integration, and interpolation weights combined into a transition table. Simulations still use continuous income draws. The notebook compares grid resolutions; the residual bounds iteration error for the numerical MDP, excluding approximation error relative to the continuous problem.

Keep `.ipynb`, `.py`, `README.md`, `requirements.txt`, and `.gitignore` in Git. The environment, caches, and disposable `results/` directory are ignored.

## Setup references

- [VS Code: Jupyter notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [VS Code: Python environments](https://code.visualstudio.com/docs/python/environments)

Model references are linked at the end of the notebook.
