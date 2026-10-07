# First Steps in Python: Markov Processes and Decisions

[Open in Colab](https://colab.research.google.com/github/YaolangZhong/U_Tokyo_Comp_Econ_Course/blob/main/weeks/week_01/lab/starter/markov_processes.ipynb) · [Read online](https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/weeks/week_01/lab/starter/markov_processes.html)

Follow the employment Markov chain from lists and probability vectors to matrices, simulation, a model class, and plotting. Then build and visualize a maze MDP and compare policies. Finish with two coding exercises: a maze policy and cake eating. The class walkthrough introduces the exercises for completion afterwards.

Save a copy in Drive and run cells in order. Complete the two scaffolds and uncomment their checks. Solution code is hidden by default in Colab; select **Show code** in each solution block to reveal it. The solution functions leave your definitions unchanged.

## Run locally

Copy the notebook, companion script, and `requirements.txt` into your project. Select a Python environment in VS Code and install the dependencies with `python -m pip install -r requirements.txt`. Use the same environment as the notebook kernel.

The script contains the notebook's code in order. Call `maze_solution()` or `cake_solution()` separately to check the reference solutions. The baseline next-month distribution is `[0.50, 0.50]`; both maze policies reach the goal in four moves.

After completing the exercises, restart and run all cells. Save the notebook and code, then commit and push. Exclude local environments and caches.
