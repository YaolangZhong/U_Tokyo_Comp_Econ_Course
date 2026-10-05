# Course maintenance

## Content and file organization

Start with [course design](COURSE_DESIGN.md) for the current teaching frame and [website editing](WEBSITE.md) for publication.

Weekly directories are schedule containers. Slide files and visible titles are topic-based, with no course week/lecture number. `course_plan.json` maps topic IDs to weeks and records the single canonical slide directory for each topic. Reuse a topic ID when it spans weeks. Multiple topics may share a week.

Each week includes lecture and lab `slides/tex/` and `slides/pdf/`, a lab-planning README, and homework reading/exercise Markdown files. `*.placeholder.md` reserves a future deck; it is not a TeX file or a completed handout.

## Compile topic decks

```sh
./compile_lecture.sh --week 6 --list
./compile_lecture.sh --week 6
./compile_lecture.sh --week 6 --part lecture
./compile_lecture.sh weeks/week_06/lecture/slides/tex/Parameterization_and_Neural_Networks.tex
./compile_lecture.sh --all
```

The week command builds standalone TeX sources physically stored under that week. For a topic reused from a different week, compile its canonical source path. `--all` builds all available weekly and supplementary sources once. Neither missing sources nor placeholders are converted to slides.

Requires Python 3.9+, `latexmk`, `pdflatex`, and the packages used by the deck. The driver detects MacTeX in `/Library/TeX/texbin`. VS Code's **Course slides (publish PDF)** recipe uses the same driver. A full TeX compilation remains unverified on this machine because `latexmk` is absent.

```text
weeks/week_NN/lecture/slides/tex/Topic.tex       source
 → build/weeks/week_NN/lecture/Topic/           ignored auxiliaries and PDF
 → weeks/week_NN/lecture/slides/pdf/Topic.pdf   published handout
```

The driver runs from the repository root for shared `figures/` and `bibliography/`, and searches the source directory for local inputs. `latexmk` performs repeat passes as needed. Only a successful nonempty PDF replaces the published handout, atomically; failures preserve the prior handout and report a nonzero status. Build commands never commit or push.

The currently available PDFs were retitled as topics without recomputing their mathematical content. Sources exist for some decks only. The PDF-only MDP, VFI, PFI, time-iteration and EGM decks will need TeX sources when rebuilt. Old author attribution and external course citation identifiers are preserved.

## Included versus ignored, for every week

| Included when staged, committed, and pushed | Ignored/local only |
| --- | --- |
| `index.qmd`, weekly/lab READMEs | Generated `_materials.qmd` includes |
| `homework/readings.md`, `homework/exercises.md` | Generated website `_site/`, `.quarto/`, `downloads/` |
| Topic TeX, assets, explicit `.placeholder.md` files | Root `build/` and LaTeX auxiliaries/logs |
| Published PDFs in `slides/pdf/` | Python/Jupyter caches and environments |
| Intended lab code, notebooks, teaching data | Any `results/` and `local/` directories |
| `course_plan.json`, shared bibliography/figures | `.DS_Store`, local `.env` configuration |
| Website/build scripts and configuration | `archive/previous_builds/`, `archive/local_cache/` |
| Retired slides, legacy labs, historical syllabi | Private local inputs and scratch files in `local/` |

The Git repository can retain archival materials without publishing them as active course content. The site build includes only explicitly selected pages and current downloads. Embedded notebook outputs are part of committed `.ipynb` files unless cleared.

## Review and publish a content update

```sh
./site.sh build
git status --short -- weeks/week_01 course_plan.json
git status --short --ignored -- weeks/week_01 build
git add -- weeks/week_01 course_plan.json
# Also stage any intentionally changed shared sources or documentation.
git diff --cached
git commit -m "Update opening-session content"
git push origin main
```

Git includes only committed changes in a push; files are not automatically uploaded simply because they are not ignored. Do not force-add ignored generated output. The GitHub Action validates and publishes the website after a push.

## History

`archive/migration-manifest.json` records the earlier folder migration. `archive/topic-migration.json` records the later topic/lab moves. These are historical records, not current schedules. Pre-transition originals are also backed up locally outside the repository in `../.course-backups/before-topic-transition-2026-10-05/`.
