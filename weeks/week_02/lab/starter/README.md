# Policy Evaluation: Job Search and Consumption–Saving

[Open in Colab](https://colab.research.google.com/github/YaolangZhong/U_Tokyo_Comp_Econ_Course/blob/main/weeks/week_02/lab/starter/value_functions.ipynb) · [Read online](https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/weeks/week_02/lab/starter/value_functions.html)

Define two model classes and simulate their policies. Solve a finite-state Bellman linear system for McCall job search; estimate continuous-state consumption–saving values by averaging simulated returns. Two coding exercises vary the acceptance threshold and saving target; hidden solution cells can be opened with **Show code**.

Run the cells in order. For local use, install `requirements.txt` and select that Python environment in your notebook editor. The companion `value_functions.py` contains the same code cells. Call `mccall_solution()` or `savings_solution()` to run the reference solutions.

The notebook links the QuantEcon model specifications and the Sutton–Barto reading. Consumption–saving uses real-valued stock and saving with continuous uniform income; selected initial stocks are evaluation points.
