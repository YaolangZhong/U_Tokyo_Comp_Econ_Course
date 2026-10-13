# Policy Evaluation: Job Search and Consumption–Saving

[Open in Colab](https://colab.research.google.com/github/YaolangZhong/U_Tokyo_Comp_Econ_Course/blob/main/weeks/week_02/lab/starter/value_functions.ipynb) · [Read online](https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/weeks/week_02/lab/starter/value_functions.html)

Define the McCall job-search and finite-state consumption–saving models, simulate policy rollouts, and evaluate supplied policies by solving their Bellman linear systems. Compare exact values with averages of simulated returns.

The notebook continues Lab 1's `step`, `simulate`, and policy-function structure and includes all definitions needed to run independently. Complete two coding exercises: compare wage-acceptance thresholds and compare saving targets. Each has a hidden solution cell; select **Show code** in Colab to reveal it.

## Run locally

Install dependencies with `python -m pip install -r requirements.txt`. Open the notebook in VS Code and select that environment as its kernel, or run `python value_functions.py`. The script contains the notebook's code cells in order. Call `mccall_solution()` or `savings_solution()` separately to run the reference solutions.

The class `McCallModel` uses status/wage states, IID offers, and permanent employment after acceptance. `ConsumptionSavingModel` uses integer stocks, a saving cap, and uniform IID income, following QuantEcon's **Example: A Growth Model**. Model rewards are current income and current consumption utility, respectively. Numerical parameters are small teaching examples.

Model and reading sources are linked in the notebook and the [course references](https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/references.html). Restart and run all cells before saving, committing, and pushing your work.
