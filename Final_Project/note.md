## Final Project: Submission Guidelines

### 1. Intention to Submit

Please email me at **yaolang.zhong@e.u-tokyo.ac.jp** to indicate your intention to submit a final project for grading in this module.

In your email, please specify:
- The paper you intend to replicate, either  
  - a paper selected from the **`paper_list`** folder, or  
  - a paper of your own choice (please attach the PDF).

If you choose your own paper, you are also required to make an appointment with me for a brief discussion of your chosen paper. Formal acceptance of the paper will be granted after this discussion.

---

### 2. Submission Deadline and Method

- **Deadline:** **February 7**  
- **Submission method:** By default, submission will be via **email**.

I will confirm at a later date whether submission through **UTOL/UTAS** is required.

---

### 3. Submission Materials

Your submission should consist of **one compressed folder** containing the following items:

- **Paper PDF**  
  The paper you replicate.

- **Summary Note (`note.md` or `note.pdf`)**  
  A concise technical summary of the paper, focusing on:
  - Research questions  
  - Key equations and models  
  - Data and empirical or computational setup  

- **`README.md`**  
  A clear description of your implementation workflow, including:
  - The role of each script  
  - Which scripts or functions correspond to specific equations, algorithms, or figures in the paper  

- **`requirements.txt`**  
  A list of all required packages for setting up the virtual environment.

- **Auxiliary scripts**  
  All relevant `.py` files used in the project.

- **Execution file (choose one):**
  - A Jupyter notebook (`.ipynb`) demonstrating the implementation step by step, **or**
  - A `main.py` file that can be executed to produce the replication output.

---

### 4. Grading Criteria

The following guidelines indicate how projects will be evaluated:

#### (1) Choice of Paper
- More **recent** or **technically advanced** papers (i.e., not trivial to replicate even with the help of LLM tools) tend to receive **higher marks**.
- More **classical** papers are a **safe choice** for passing or receiving mid-range marks.

#### (2) Technical Understanding
- Your `note.md` and `README.md` should clearly demonstrate your understanding of the **technical aspects** of the paper, such as:
  - Mathematical structure  
  - Algorithms  
  - Data handling and implementation details  

> Emphasis is placed on technical understanding rather than the economic motivation or interpretation.

#### (3) Code Quality: Runability and Readability
- Your code should run **without errors** when the environment is set up using `requirements.txt`.
- Code readability is important and should be supported by:
  - Clear comments  
  - Descriptions of function inputs and outputs  
  - Appropriate use of type hints (where applicable)

#### (4) Results
- This is the **least important** criterion.
- Exact replication of results is encouraged but **not required**.
- Strong performance in points (2) and (3) is sufficient for a **solid grade**.