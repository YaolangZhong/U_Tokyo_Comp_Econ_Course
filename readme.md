# Computational Economics Module (Autumn A1A2 Term)

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

  - **Sharing**: After grading is completed, replication projects will be compiled and shared in two stages: (1) internally among the registered students of this module; and (2) optionally in a publicly viewable GitHub repository, with the possibility of notifying the original paper’s author for comments. Each level of sharing will take place only with the approval of the student group involved.

# 3. Syllabus

## Week 1 (Oct 1)

### Lecture 1: Introduction to Computational Concepts
- Round-off error and truncation error
- Conditioning and stability
- Rates of convergence and Big-O notation
- Direct vs. iterative methods

**Supplementary Reading**  
- Kenneth Judd, *Numerical Methods in Economics*, Chapter 2

### Lab 1: Introduction to Python and VS Code
- Installing Python and VS Code
- Python basics

**Supplementary Online Resources**  
- [Getting Started with Python in VS Code (Official Video)](https://www.youtube.com/watch?v=D2cwvpJSBX4)
- [VSCode Tutorial For Beginners - Getting Started With VSCode](https://www.youtube.com/watch?v=ORrELERGIHs)
- [Coding for Economists](https://aeturrell.github.io/coding-for-economists/intro.html)
- [QuantEcon - Python Programming for Economics and Finance](https://python-programming.quantecon.org/intro.html)

## Week 2 (Oct 8)
### Lecture 2: Introduction to Markov Decision Processes (MDP) 
- Deterministic, Stochastic, Finite horizon and Infinite horizon MDP
- Bellman equation

### Lab 2: Introduction to Git, Economics Modeling with Python
- Git installation, Git command, synergy with Github
- Using Python to construct economics models


## Week 3  
### Lecture 3: Solution Algorithms I  
- Value function iteration  
- Policy function iteration  

### Lab 3: Implementing Solution Algorithms  
- Hands-on coding of value and policy iteration  

## Week 4 
### Lecture 4: Solution Algorithms II  
- Time iteration  
- Endogenous Grid Method (EGM)

### Lab 4: Implementing Solution Algorithms  
- Hands-on coding of time iteration and EGM

---

## Week 5  
### Lecture 5: TBD
### Lab 5: TBD

## Week 6  
### Lecture 6: TBD
### Lab 6: TBD

## Week 7  
### Lecture 7: TBD
### Lab 7: TBD

## Week 8  
### Lecture 8: TBD
### Lab 8: TBD


## Week 9  
### Lecture 9: TBD
### Lab 9: TBD


## Week 10  
### Lecture 10: TBD
### Lab 10: TBD

## Week 11  
### Lecture 11: TBD
### Lab 11: TBD

## Week 12  
### Lecture 12: TBD
### Lab 12: TBD

---

# 3. Reference Textbooks and Courses
- Dimitri P. Bertsekas, **Reinforcement learning and optimal control** (textbook and 2025 Spring course at ASU): https://web.mit.edu/dimitrib/www/RLbook.html
- Thomas J. Sargent and John Stachurski, **QuantEcon** online courses: https://quantecon.org/
- Jesús Fernández-Villaverde, courses in computation and macroeconomics: https://www.sas.upenn.edu/~jesusfv/teaching.html
- Zhigang Feng, workshops on AI, Machine Learning for Economists: https://sites.google.com/site/zfeng202/notes
- Kenneth Judd, **Numerical methods in economics** textbook, https://www.business.uzh.ch/dam/jcr:ffffffff-cd5d-ce16-0000-000076c01f71/NumericalMethodsJuddPartIPP1-307.pdf