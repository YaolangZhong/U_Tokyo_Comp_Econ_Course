"""Plotting helpers: keep the experiment notebook short."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE, TEAL, RED = "#2464a0", "#007d85", "#ba4a45"


def plot_convergence(result, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.semilogy(np.maximum(result["residuals"], 1e-16), color=BLUE)
    ax.set(xlabel="Iteration k", ylabel="Bellman residual", title=title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_maze(model, result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    path = model.path(result["policy"])
    for ax, title in zip(axes, ["Greedy policy", "Values and the chosen path"]):
        for y in range(6):
            for x in range(9):
                color = "#928580" if (x, y) in model.walls else "white"
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color,
                                       edgecolor="#cccccc", linewidth=0.5))
        for i, (x, y) in enumerate(model.cells):
            if i in [model.start, model.goal]:
                color = "#f5dea2" if i == model.start else "#b5ddd5"
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color,
                                       edgecolor="#cccccc", linewidth=0.5))
            ax.text(x + .08, y + .90, str(i + 1), va="top", fontsize=6, color="#888888")
            label = (model.actions[result["policy"][i]] if ax is axes[0]
                     else f'{result["values"][i]:.1f}')
            if i == model.goal and ax is axes[0]:
                label = "·"  # No move is needed after reaching the goal.
            ax.text(x + .5, y + .5, label, ha="center", va="center", color="#aaaaaa", fontsize=9)
        for s, next_s in zip(path[:-1], path[1:]):
            x, y = model.cells[s]; u, v = model.cells[next_s]
            ax.annotate("", xy=(u + .5, v + .5), xytext=(x + .5, y + .5),
                        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.6})
        ax.set(xlim=(0, 9), ylim=(0, 6), title=title, aspect="equal")
        ax.axis("off")
    fig.suptitle(f"Maze: beta = {model.beta}; yellow = start, green = goal", fontsize=12)
    fig.tight_layout()
    return fig


def plot_mccall(model, result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for k in [1, 2, 3, 10]:
        if k < len(result["history"]):
            axes[0].plot(model.wages, result["history"][k, :model.n], "o--", alpha=.55, label=f"v_{k}")
    axes[0].plot(model.wages, result["values"][:model.n], "o-", color=BLUE, label="Converged")
    axes[0].set(xlabel="Offer w in state (U, w)", ylabel="Expected discounted income", title="Unemployed-state values")
    accept = result["policy"][:model.n] == "accept"
    axes[1].scatter(model.wages, accept.astype(int), color=TEAL, s=60)
    axes[1].set(xlabel="Offer w in state (U, w)", yticks=[0, 1], yticklabels=["Reject", "Accept"],
                ylim=(-.3, 1.3), title="Recovered policy")
    axes[0].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_saving(model, result, finer_model=None, finer_result=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(model.states, result["values"], color=BLUE, label="Baseline grid")
    axes[1].plot(model.states, result["policy"], color=TEAL, label="Saving")
    axes[1].plot(model.states, model.states - result["policy"], color=BLUE, label="Consumption")
    if finer_model is not None:
        axes[0].plot(finer_model.states, finer_result["values"], "--", color=RED, label="Finer grid")
        axes[1].plot(finer_model.states, finer_result["policy"], "--", color=RED, label="Saving: finer grid")
    axes[0].set(xlabel="Stock s", ylabel="Expected discounted utility", title="Approximate value")
    axes[1].set(xlabel="Stock s", ylabel="Units of goods", title="Approximate policy")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_saving_rollout(states, actions):
    fig, ax = plt.subplots(figsize=(8, 3))
    t = np.arange(len(actions))
    ax.plot(t, states[:-1], color=BLUE, label="Stock")
    ax.plot(t, actions, color=TEAL, label="Saving")
    ax.plot(t, states[:-1] - actions, color=RED, label="Consumption")
    ax.set(xlabel="Period", ylabel="Units of goods", title="One rollout under the recovered policy")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    return fig
