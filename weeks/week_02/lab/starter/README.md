# Value Functions and Dynamic Programming

[Open in Colab](https://colab.research.google.com/github/YaolangZhong/U_Tokyo_Comp_Econ_Course/blob/main/weeks/week_02/lab/starter/value_functions.ipynb) · [Read online](https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/weeks/week_02/lab/starter/value_functions.html)

Continue from Lab 1's employment and cake-eating models. Calculate discounted returns, evaluate fixed policies by linear algebra and iteration, implement the Bellman operator, recover an optimal policy, and use backward induction for a finite horizon.

The notebook includes the required Lab 1 definitions and runs independently. Complete two coding exercises: change employment transition probabilities, then solve a larger cake-eating model. Each has a hidden solution block; select **Show code** in Colab to reveal it. Solution functions leave your exercise definitions unchanged.

## Run locally

Install dependencies with `python -m pip install -r requirements.txt`. Open the notebook in VS Code and select the same environment as its kernel, or run `python value_functions.py`. The script contains the notebook's code cells in order. Call `employment_solution()` or `larger_cake_solution()` separately to check the reference solutions.

The lecture benchmarks are employment values `[72/13, 112/13]`, cake values `[0, 1, 1.9]`, and cake consumption `[0, 1, 1]`. Restart and run all cells after completing your code, then save, commit, and push your work. Keep local environments and caches out of Git.
