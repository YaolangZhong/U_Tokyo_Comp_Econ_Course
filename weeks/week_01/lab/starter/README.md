# First Steps in Python: Markov Processes and Decisions

[Open in Colab](https://colab.research.google.com/github/YaolangZhong/U_Tokyo_Comp_Econ_Course/blob/main/weeks/week_01/lab/starter/markov_processes.ipynb) · [Read online](https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/weeks/week_01/lab/starter/markov_processes.html)

Start with the employment/unemployment chain from the lecture: lists, NumPy arrays, probability vectors, transition matrices, and a `MarkovProcess` class with a `simulate` method. A smaller maze introduces state vectors, an MDP class, and policies. Finish with the independent cake-eating MDP exercise.

Save a copy in Drive before editing in Colab. Run the cells in order, complete the exercise, then restart the runtime and run all cells. Download your completed notebook to keep in your project. The final `CakeMDP` class and policies are student exercises; replace their `pass` placeholders and activate the supplied checks.

## Run locally

Copy `markov_processes.ipynb`, `markov_processes.py`, `requirements.txt`, and `.gitignore` into your project. Create and select a Python environment in VS Code. Install the dependencies into it with `python -m pip install -r requirements.txt`, then select the same environment as the notebook kernel.

Run the notebook and the companion script. The script contains the supplied code cells in notebook order. The baseline next-month probability vector is `[0.50, 0.50]`; changing the job-finding probability to `0.40` gives `[0.38, 0.62]`. The guided maze policy reaches the goal in four moves and then remains there.

Record your Python/package versions, run instructions, results, and completed exercise in your README. Commit and push the files; exclude local environments and caches.
