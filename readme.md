# Computational Economics — University of Tokyo

**[Course website](https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/)** · [Course design](docs/COURSE_DESIGN.md) · [Editing guide](docs/WEBSITE.md)

The course combines computational economic methods with research workflows. Lectures cover core methods earlier and reserve a closing block for computational literature using machine learning. Labs focus on LLM-agent tools for research alongside Git and other regular tools; Python supports the research task rather than serving as the main lab subject.

## Weekly framework, topic-based slides

`weeks/week_01/` through `weeks/week_13/` organize the schedule. Slide decks have topic names, and their length is independent of the number of class sessions. Each week contains:

```text
index.qmd                    weekly web content
lecture/slides/tex/           topic sources or explicit placeholders
lecture/slides/pdf/           published topic handouts
lab/README.md                research-tools lab planning frame
lab/slides/tex/               topic sources or explicit placeholders
lab/slides/pdf/               published lab handouts
homework/readings.md          reading-assignment placeholder
homework/exercises.md         exercise-assignment placeholder
```

Week 1 contains a **Course Introduction** placeholder and **Introduction to Markov Decision Processes**, moved from the old Week 2. The old standalone computational-concepts introduction is retired.

[The course plan](course_plan.json) maps topic IDs to weeks. Weeks 2–10 are provisionally allocated to core methods and synthesis; Weeks 11–13 are reserved provisionally for ML computational literature. Pacing, tools, papers, and homework will be discussed before authoring. A placeholder does not constitute an assignment.

## Working on content

- Edit `weeks/week_NN/index.qmd`, `lab/README.md`, and the two homework files.
- Use topic names for slide files and titles. A topic can be referenced in several weeks through `course_plan.json`, while its source is stored only once.
- Keep shared sources in `bibliography/references.bib`; the reference page is a background collection until weekly readings are chosen.
- Build a week's available TeX with `./compile_lecture.sh --week 1`, or build a canonical `.tex` source by path. Placeholder Markdown files are not compiled.
- Preview with `./site.sh preview`; validate with `./site.sh build`. Push reviewed commits to `main` to rebuild the site automatically.

See [maintenance and Git inclusion rules](docs/MAINTENANCE.md).

## Preserved material

- `archive/retired_slides/`: abandoned opening lecture, retained for possible integration into later topics.
- `archive/legacy_labs/`: previous Python-focused labs, kept for optional reuse.
- `archive/previous_syllabi/`: historical syllabus documents, not current course requirements.
- `Final_Project/`: existing replication-project material. The current project grading policy is unchanged; homework grading remains undecided.
