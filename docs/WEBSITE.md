# Edit and publish course content

Website: https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/

## What to edit

| Task | Source |
| --- | --- |
| Change week-to-topic allocation or continue a topic next week | `course_plan.json` |
| Weekly introduction and learning objectives | `weeks/week_NN/index.qmd` |
| Lab design | `weeks/week_NN/lab/README.md` and weekly lab-plan text |
| Reading assignment | `weeks/week_NN/homework/readings.md` |
| Exercise assignment | `weeks/week_NN/homework/exercises.md` |
| Topic slide deck | Canonical `slides/tex/` and published `slides/pdf/` |
| Course structure and assessment | `syllabus.qmd` and `docs/COURSE_DESIGN.md` |
| Home and announcements | `index.qmd` |
| Research-tools orientation | `getting-started.qmd` |
| Literature overview / bibliographic records | `references.qmd` / `bibliography/references.bib` |
| Project requirements | `project.qmd` |

Week numbers organize sessions; slide titles and filenames identify topics. In `course_plan.json`, each topic has a stable ID, descriptive title, source directory, filename stem, and status (`existing` or `placeholder`). A week's `lecture_topics` lists the IDs it covers. Repeating an ID in another week reuses the same deck, allowing flexible length without duplicate source files. Update the weekly prose and syllabus table when changing allocations.

For a new topic, add its record and assignment, create its canonical slide directories, and use a topic-only filename. A placeholder has a `.placeholder.md` planning note, no invented `.tex` or PDF. When ready, add a real PDF and change its status to `existing`.

Keep the material/homework include lines in weekly pages. `scripts/prepare_site.py` regenerates download links from the topic manifest and lab files. Do not edit `_materials.qmd` directly. Distinct topic names label each PDF button, including when one week contains several decks.

## Preview and publish

1. Edit content and homework in plain Markdown/Quarto. Cite a shared source using a key such as `@judd1998`.
2. Compile any changed TeX through `compile_lecture.sh`; publishing the website does not compile TeX.
3. Run `./site.sh preview` to inspect locally, or `./site.sh build` to render and validate links.
4. Review the Git diff, commit intended source changes and handout PDFs, and push to `main`.

GitHub Actions builds and deploys after a successful push. Pull requests validate without publishing. A failed build leaves the previous deployment unchanged. No personal token is stored in the repository.

The wrapper uses Quarto 1.10.18 from PATH or `../.tools/bin/quarto`; the action installs the pinned version. Python 3.9+ is required. Placeholder includes are prepared before Quarto discovers its input files.

## Publishing boundaries

Current weekly pages and assigned topic PDFs are public. Old Python labs, retired slides, historical syllabi, maintenance notes, and previous builds are not copied into the website; archives remain in the Git repository. No placeholder is a current assignment, and no agent product or paper is prescribed yet.

Future notebook pages display saved results without executing Python in the publishing workflow. Lab download bundles include the permitted teaching-file types listed in `prepare_site.py` and exclude caches, environments, results, local inputs, and slides. A README-only lab frame does not create a download bundle.

Generated `_site/`, `.quarto/`, `downloads/`, and material includes are ignored. Commit `course_plan.json`, pages, homework, placeholders, assets, scripts, and published PDFs. See [maintenance](MAINTENANCE.md) for the full Git policy.
