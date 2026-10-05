# Course design — working framework

This document records the redesign requested on 2026-10-05. It is a frame for discussing content, not a completed syllabus or assignment list.

## Organization and pacing

- Retain `weeks/week_01/` through `weeks/week_13/` as the schedule structure.
- Use topic names for slide filenames, title slides, and references to this course's other decks. Do not encode a week or lecture number into new decks.
- Store each deck once, under its initial owning week's `lecture/slides/` or `lab/slides/`. `course_plan.json` associates lecture topic IDs with weeks. To continue a topic in another week, reference the same topic ID again; do not copy or renumber the source.
- A week may have several decks. A deck may span multiple weeks. The compiler builds physically stored sources; compile the canonical source explicitly when a topic is reused elsewhere.

## Opening session

Week 1 contains **Course Introduction** (placeholder only) and **Introduction to Markov Decision Processes**, moved from the previous Week 2. The MDP PDF's course lecture-number prefix and old scheduled date have been removed. No source TeX is available yet for that deck.

The old Introduction to Some Computational Concepts is retired in `archive/retired_slides/`. Selected concepts can later be introduced where they help explain a method. No reintegration has been performed; the archive README records possible insertion points for discussion.

## Lecture frame

The provisional allocation moves methods earlier: dynamic programming; Euler-equation methods and endogenous grids; expectations; simulation; approximation; supervised/unsupervised learning; reinforcement learning; heterogeneous-agent applications. Week 10 is a synthesis/pacing buffer. Weeks 11–13 provisionally hold ML computational-literature sessions.

These allocations make space for discussion; they are not a decision about how many topics students can cover in one session. All timing after the explicit Week 1 change remains adjustable in `course_plan.json` and the weekly pages.

## Lab frame

The new focus is LLM agents and related tools for research, alongside Git and conventional research tools. Python is used as needed rather than taught as the main subject. Each week reserves:

- Research question/task and tool workflow.
- Agent/tool and supporting conventional tools, to be selected.
- In-class activity, verification procedure, and deliverable.
- A topic-based lab slide placeholder.

Existing Python labs are preserved, with their modules together, in `archive/legacy_labs/week_NN/`. They are not published as current labs or automatically assigned. We may reuse individual examples later.

## Homework and literature

Every week has `homework/readings.md` and `homework/exercises.md`, included on the weekly web page. Each file explicitly says no assignment has yet been set. They reserve reading sections/questions and an exercise/deliverable, with due dates and submission details undecided.

The literature block will examine recent computational economics using ML. Paper selection, recency criteria, available replication materials, discussion prompts, and exercises are future content decisions. No papers or specific agent products are selected by this redesign.

Maintain bibliographic metadata in `bibliography/references.bib`. The current reference list is background, not the assigned reading list. Add citation keys to a week's reading file when a reading is agreed. Preserve external source identifiers such as ASU lecture numbers: they identify cited material and are not this course's schedule.

## Assessment and historical documents

No grading change is implied by adding homework slots. The existing 100% replication-project policy remains pending an explicit assessment decision. Previous syllabi are in `archive/previous_syllabi/` and are not current registration documents.
