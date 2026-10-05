# Maintain course content

The student-facing course website is https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/.
The source repository remains https://github.com/YaolangZhong/U_Tokyo_Comp_Econ_Course.

## Your normal workflow

1. Edit `weeks/week_09/index.qmd` (or another week's page). Use Markdown for explanations, `$...$` for inline math, and `$$...$$` for displayed math.
2. Update that week's slide TeX, compiled PDF, notebooks, or Python modules as needed. Build TeX with `./compile_lecture.sh --week 9`; the website does not compile TeX for you.
3. Run `./site.sh preview` and open the local URL it prints. Run `./site.sh build` before publishing; it renders and checks internal file links.
4. Stage the changed source content and PDF handouts, review the staged diff, commit, and push to `main`. The **Publish course website** GitHub Action builds and deploys the site. The live site changes only after the deployment succeeds.

The website uses Quarto 1.10.18 and Python 3.9+. Quarto was installed locally in the Teaching folder's `.tools/`; `site.sh` detects it there, or uses `quarto` on PATH on other machines. The GitHub Action installs the pinned Quarto version automatically.

## What to edit

| Content | Source |
| --- | --- |
| Home, announcements | `index.qmd` |
| Course structure and assessment | `syllabus.qmd` |
| Installation instructions | `getting-started.qmd` |
| Weekly introduction, objectives, exercises, readings | `weeks/week_NN/index.qmd` |
| Lecture/lab slides | Week's `slides/tex/` and `slides/pdf/` |
| Lab content | Week's `lab/*.ipynb` and supporting files |
| Project requirements | `project.qmd` |
| Shared bibliography | `bibliography/references.bib` |
| Reference-page introduction | `references.qmd` |

Use `@judd1998` (or another bibliography key) to cite a source. Add new sources to the shared `.bib`; the References page lists every entry. The pre-existing final-project paper catalog remains linked rather than duplicated.

All 13 weekly pages share the same structure. Keep the `{{< include _materials.qmd >}}` line: the build regenerates it from the actual files, so PDF links, online notebook links, and complete lab ZIP downloads stay synchronized. Do not edit generated `_materials.qmd` files.

## Rendering and downloads

- Notebook pages display existing saved outputs; the publishing workflow does not execute model code or install lab packages.
- Each lab ZIP preserves the lab folder's files and subdirectories. It excludes caches, environments, `results/`, `local/`, hidden files, and slides. Eligible extensions are explicit in `scripts/prepare_site.py`; extend the list when adding another teaching-data format.
- Archives, instructor maintenance notes, build intermediates, and the final-project paper PDFs are not copied into the website. The public GitHub repository still retains previously committed material.
- Generated `_site/`, `.quarto/`, `downloads/`, and material includes are ignored by Git. Commit source pages, configuration, scripts, source notebooks, and published slide PDFs.
- `site.sh` prepares the generated includes before Quarto discovers pages, including on a clean clone. Use it for local builds; the Action performs the same preparation.

## Publication and checks

GitHub **Settings → Pages → Build and deployment → Source** must be **GitHub Actions**. No personal token is stored in the repository. Deployment uses the workflow's GitHub-provided token with Pages deployment permissions.

Pull requests build and validate the website without publishing. Pushes to `main` build and publish. A failed build leaves the existing deployed website unchanged. Inspect the Actions tab when a change does not appear.

The current site uses the existing teaching sequence. Dates and the final-project deadline for the next offering remain unannounced. Most detailed weekly objectives and exercises are intentionally left for the upcoming content revision.
