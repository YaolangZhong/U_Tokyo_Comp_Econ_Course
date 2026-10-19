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
| `vfi.py` | Shared VFI algorithm and policy simulation |
| `maze.py` | Lecture maze, rewards, and transitions |
| `mccall.py` | Job-search environment with states `(U, w)` and `(E, w)` |
| `saving.py` | Continuous saving model and numerical Bellman update |
| `figures.py` | Plotting helpers |
| `requirements.txt` | Packages installed in `.venv` |

Begin with the maze, then solve McCall job search and continuous saving. The notebook carries over Lab 2's model classes and `step(state, action, rng)` interface; VFI now selects the policy.

For saving, the state and action remain continuous. Stored stock points, an action grid, and midpoint integration approximate the Bellman update. The notebook compares grid resolutions; the VFI residual measures iteration error for the numerical operator, not the full error relative to the continuous problem.

Keep `.ipynb`, `.py`, `README.md`, `requirements.txt`, and `.gitignore` in Git. The environment, caches, and disposable `results/` directory are ignored.

## Setup references

- [VS Code: Jupyter notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [VS Code: Python environments](https://code.visualstudio.com/docs/python/environments)

Model references are linked at the end of the notebook.
