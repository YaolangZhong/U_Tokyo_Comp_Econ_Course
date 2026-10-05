# Course maintenance

## Layout

- `weeks/week_01/` through `weeks/week_13/`: lecture and lab materials.
- Each `lecture/` and `lab/` has `slides/tex/` for sources and `slides/pdf/` for student handouts.
- Lab notebooks, supporting Python modules, and data belong together in the week's `lab/` folder. Start notebooks with that folder as their working directory.
- `topics/`: supplementary Git, Google AI, and AR(1) materials outside the numbered sequence.
- `figures/`: shared figure assets, resolved from the repository root during slide builds.
- `bibliography/`: shared bibliography; reference-system updates are a separate task.
- `Final_Project/`: existing project instructions and paper collection.
- `archive/`: older slide versions and previous build outputs, preserved for comparison.
- `build/`: ignored, reproducible LaTeX intermediates.

Lecture TeX sources are available for weeks 7–13. Weeks 1–6 have PDF handouts only. Lab 1 has a PDF slide deck; the other lab slide directories are placeholders. Placeholder README files identify missing material; do not treat them as completed slides.

## Compile TeX to PDF

Run from the repository root (or invoke the script by absolute path from elsewhere):

```sh
# Preview the decks selected for a week; no TeX installation needed.
./compile_lecture.sh --week 9 --list
# Build every available lecture and lab deck for week 9.
./compile_lecture.sh --week 9
# Build only a lecture or lab.
./compile_lecture.sh --week 9 --part lecture
./compile_lecture.sh --week 9 --part lab
# Build one deck, or all available course and supplementary decks.
./compile_lecture.sh weeks/week_09/lecture/slides/tex/Lecture_9_Parameterization_and_Neural_Network.tex
./compile_lecture.sh --all
```

Requires Python 3.9+, `latexmk`, `pdflatex`, and the LaTeX packages used by the slides. MacTeX's `/Library/TeX/texbin` is detected even when absent from the shell path. No dependencies are automatically installed.

For week 9's lecture, the workflow is:

```text
weeks/week_09/lecture/slides/tex/<deck>.tex   source (Git)
    → build/weeks/week_09/lecture/<deck>/    PDF + logs + auxiliaries (ignored)
    → weeks/week_09/lecture/slides/pdf/<deck>.pdf   published handout (Git)
```

`latexmk` manages repeat passes and bibliography processing when requested by the source. Each deck has its own build directory. A successful, nonempty PDF atomically replaces the handout; a failed build leaves the previous handout intact and returns a nonzero exit code. In a batch, other decks still build and failures are reported. Missing slide sources are explicitly skipped, never fabricated. `--list` previews selection without compilation.

Keep standalone decks directly in `slides/tex/`; place included TeX fragments in a subfolder. Week/all selection recognizes files containing `\documentclass`. The build runs from the repository root for existing shared `figures/` and `bibliography/` paths, and also searches the source directory recursively for local inputs. Avoid reusing asset names between shared and local folders.

In VS Code, open the course repository as the workspace, open a deck, and select the **Course slides (publish PDF)** recipe. It invokes this same driver; automatic builds are disabled. The viewer opens the published PDF. Compilation does not stage, commit, or push files.

Existing handouts may predate their TeX sources. Always inspect the newly built PDF before committing. Compilation has not been verified against a real TeX installation on this machine.

## What is uploaded for each week?

The same rules apply to **all 13 weeks**, for both lecture and lab.

| Path/content | Git policy | Purpose |
| --- | --- | --- |
| `slides/tex/**` (`.tex`, source figures, local `.bib`/`.sty`) | Include | Editable, reproducible slide sources |
| `slides/pdf/*.pdf` | Include | Student handouts |
| `lab/*.ipynb`, `lab/*.py`, teaching data/assets | Include | Exercises and runnable code |
| Weekly `README.md` and placeholder READMEs | Include | Navigation and material status |
| Root `figures/`, `bibliography/`, scripts and configuration | Include when changed | Shared build dependencies |
| Root `build/**` | Ignore | Generated intermediate PDFs, logs, and auxiliary files |
| Any `results/` or `local/` directory | Ignore | Disposable lab results or local-only inputs |
| `__pycache__/`, `.ipynb_checkpoints/`, `.venv/`, `venv/` | Ignore | Runtime caches and environments |
| LaTeX `.aux`, `.log`, `.bbl`, `.blg`, `.nav`, `.snm`, `.toc`, etc. | Ignore | Recreated during compilation |
| `.DS_Store`, `.env`, `.env.*` (except `.env.example`) | Ignore | OS metadata and local configuration |
| `archive/previous_builds/`, `archive/local_cache/` | Ignore | Local historical outputs and caches |
| `archive/alternative_slides/`, migration manifest | Include | Preserved alternative teaching materials and move history |

**Include means eligible for Git, not automatically uploaded.** Only staged changes become part of a commit, and only pushed commits reach GitHub. Notebook outputs embedded inside `.ipynb` are included with that notebook; `.gitignore` cannot filter individual cells. Clear unnecessary outputs before staging. Keep intended teaching datasets outside `results/` and `local/`.

### Review and upload a week

```sh
# Refresh remote knowledge and fast-forward before editing when the checkout is clean.
git pull --ff-only
./compile_lecture.sh --week 9
# Inspect included changes and ignored files separately.
git status --short -- weeks/week_09
git status --short --ignored -- weeks/week_09 build
# Explain the rule for a particular ignored file.
git check-ignore -v --no-index build/weeks/week_09/lecture/example/example.aux
# Stage this week, including its published PDFs and any intentional removals.
git add -- weeks/week_09
# If shared inputs/build settings changed, stage those specific files too.
# Example: git add -- figures/changed-figure.png bibliography/references.bib
git diff --cached --stat
git diff --cached
git commit -m "Update week 09 materials"
git push origin main
```

Do not use `git add -f` for ignored build outputs. `.gitignore` does not untrack previously committed files: tracked caches must be removed from the index once. This cleanup removes the already-missing tracked system/cache files from the index. The pending course reorganization must be committed as a whole before using the week-only update routine; otherwise old paths and new paths would be split across commits.

## Organization record

The 2026-10-05 migration preserves original file contents except the Week 13 notebook's descriptive folder label. `archive/migration-manifest.json` maps moved files and their original SHA-256 hashes. A full pre-migration backup is held outside the repository at `../.course-backups/before-organization-2026-10-05/`.

The old ignore-all Git policy has been replaced so TeX and notebook updates are visible to Git. Python caches, system metadata, virtual environments, and build intermediates remain ignored. Previously tracked, already-missing cache and system files are staged for removal from version control with the organization changes.

## Website publishing

See [Website editing guide](WEBSITE.md). Weekly `index.qmd` pages, root `.qmd` pages, `_quarto.yml`, `styles.scss`, scripts, and `.github/workflows/publish.yml` are source files to commit. `_site/`, `.quarto/`, `downloads/`, and generated `_materials.qmd` includes are ignored. GitHub Actions builds them from source and publishes the `_site/` artifact; they do not belong in commits to `main`.
