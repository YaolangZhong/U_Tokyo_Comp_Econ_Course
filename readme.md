# Computational Economics Module (Autumn A1A2 Term)

**[Course website](https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/)** · [Website editing guide](docs/WEBSITE.md)

Edit each week's `index.qmd` for its web content. Preview with `./site.sh preview`; build and check with `./site.sh build`. Push a commit to `main` to publish through GitHub Actions.


# 1. Overview
This module is designed to help students understand and apply computational tools that will later facilitate their own research—covering methods from classical approaches to state-of-the-art techniques. It combines theory with hands-on programming practice. Each class is divided into two parts:  

- **Theory (first 50 minutes):** Focused on intuition, applications, and key concepts, with minimal formal proofs. Supplementary readings and materials will be provided for students who wish to explore the mathematical details more deeply.  
- **Practice (second 50 minutes):** Programming implementation, model solving, and empirical applications.

## 1.1 Scope and Approach
- **Research-Oriented Tools:** Equip students with computational skills that directly support independent research projects.  
- **Machine Learning Integration:** Introduction to machine learning methods for tackling the *curse of dimensionality* in dynamic economic models.  
- **LLM-Assisted “Vibe Coding”:** Practice using large language models (LLMs) to streamline coding, debugging, and syntax, saving valuable research time.  
- **Model Estimation:** Depending on student interest, the course may extend to estimation techniques for structural models.  

## 1.2 Programming Languages and Tools
- **Primary Language:** Python  
  - Core libraries: NumPy (arrays, vectorization), Matplotlib (visualization), SciPy (statistics and numerical methods), QuantEcon by Thomas J. Sargent and John Stachurski, which provides a rich set of computational tools tailored for economists
- **Machine Learning:** JAX and PyTorch for machine learning applications  
- **Other Languages:** MATLAB and Julia are not covered in this module, but their syntactic similarity to Python makes translation of code and methods relatively straightforward.  

## 1.3 Learning Outcomes
By the end of the course, students will:  
1. Understand the computational foundations of structural economic models.  
2. Apply Python and modern ML frameworks to solve high-dimensional economic problems.  
3. Gain practical experience in debugging and implementing models efficiently with the assistance of LLMs.  
4. Be prepared to extend these tools to model estimation and empirical analysis.  

# 2. Grading Scheme  

- **Replication Project (100%)**  
  Each student selects one paper in the literature and attempts to replicate its main quantitative results.  

## Remarks
  - **Collaboration**: Students are encouraged to form teams of 2–3 members. Each team must replicate as many papers as there are members. The final grade will be based on the overall quality of the teamwork, with equal marks assigned to all team members.

  - **Marking Criteria**: Exact reproduction of the original quantitative results is not required. Evaluation will focus on the quality of the replication effort itself, including project design, algorithm implementation, and programming skills.

  - **Paper Selection**: Students may propose a paper to replicate, subject to instructor approval. The paper may be published or unpublished and does not need to come directly from computational economics, as long as it is prominent in the student’s field of interest. If source code is available, students must demonstrate novelty by improving upon the existing work—for example, by developing an alternative algorithm, redesigning the code pipeline, or conducting robustness checks. The instructor will also provide a list of suggested candidate papers midway through the term.

  - **Sharing**: After grading is completed, replication projects will be compiled and shared in two stages: (1) internally among the registered students of this module; and (2) optionally in a publicly viewable GitHub repository. Each level of sharing will take place only with the approval of the student group involved.

# 3. Weekly materials

Each week contains `lecture/slides/tex/`, `lecture/slides/pdf/`, `lab/slides/tex/`, and `lab/slides/pdf/`. Lab notebooks and Python modules stay together directly under `lab/`. Empty slide folders identify material to be rebuilt; existing PDFs have not been regenerated.

| Week | Existing topic | Materials |
| --- | --- | --- |
| 01 | Computational Concepts | [Week 01](weeks/week_01/README.md) |
| 02 | Markov Decision Processes | [Week 02](weeks/week_02/README.md) |
| 03 | Value Function Iteration | [Week 03](weeks/week_03/README.md) |
| 04 | Policy Function Iteration | [Week 04](weeks/week_04/README.md) |
| 05 | Time Iteration | [Week 05](weeks/week_05/README.md) |
| 06 | Endogenous Grid Method | [Week 06](weeks/week_06/README.md) |
| 07 | Quadrature and Monte Carlo | [Week 07](weeks/week_07/README.md) |
| 08 | Simulation-Based Learning | [Week 08](weeks/week_08/README.md) |
| 09 | Parameterization and Neural Networks | [Week 09](weeks/week_09/README.md) |
| 10 | Supervised Learning | [Week 10](weeks/week_10/README.md) |
| 11 | Unsupervised Learning | [Week 11](weeks/week_11/README.md) |
| 12 | Reinforcement Learning | [Week 12](weeks/week_12/README.md) |
| 13 | Heterogeneous Agent Models | [Week 13](weeks/week_13/README.md) |

See [course maintenance](docs/MAINTENANCE.md) for editing and building, [supplementary topics](topics/), [final project](Final_Project/), and [archived versions](archive/README.md). The [previous syllabus](docs/syllabus_previous.md) and [registration syllabus](docs/syllabus_registration.md) are retained for reference; they may differ from the existing materials.

# 4. Reference Textbooks and Courses
- Dimitri P. Bertsekas, **Reinforcement learning and optimal control** (textbook and 2025 Spring course at ASU): https://web.mit.edu/dimitrib/www/RLbook.html
- Thomas J. Sargent and John Stachurski, **QuantEcon** online courses: https://quantecon.org/
- Jesús Fernández-Villaverde, courses in computation and macroeconomics: https://www.sas.upenn.edu/~jesusfv/teaching.html
- Zhigang Feng, workshops on AI, Machine Learning for Economists: https://sites.google.com/site/zfeng202/notes
- Kenneth Judd, **Numerical methods in economics** textbook, https://www.business.uzh.ch/dam/jcr:ffffffff-cd5d-ce16-0000-000076c01f71/NumericalMethodsJuddPartIPP1-307.pdf