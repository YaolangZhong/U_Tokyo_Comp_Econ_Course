# Week 10: Supervised Learning

Existing materials, to be rebuilt and updated.

## Lecture

- [TEX slides](lecture/slides/tex/)
- [PDF slides](lecture/slides/pdf/)

## Lab

- [TEX slides](lab/slides/tex/)
- [PDF slides](lab/slides/pdf/)
- [Lab_10_ML_Classification.ipynb](lab/Lab_10_ML_Classification.ipynb)
- [Lab_10_ML_Regression.ipynb](lab/Lab_10_ML_Regression.ipynb)

## Build and Git upload

From the repository root, run `./compile_lecture.sh --week 10` (add `--list` to preview or `--part lecture` / `--part lab` to select one section).

- **Include:** slide TeX and assets, published PDFs, lab notebooks/code/teaching data, and this README.
- **Ignore:** root `build/`, lab `results/` and `local/`, Python/Jupyter caches, environments, and LaTeX auxiliary files. Embedded notebook outputs are included unless cleared.
- **Upload:** `git add -- weeks/week_10`, review `git diff --cached`, then commit and push. Include any changed shared dependencies explicitly.

See [build and upload instructions](../../docs/MAINTENANCE.md) for prerequisites, exclusions, and the initial reorganization commit.
