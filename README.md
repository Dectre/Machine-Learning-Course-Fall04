# Machine Learning Course — Fall 1404 (2025–2026)

[![University of Tehran](https://img.shields.io/badge/University-University%20of%20Tehran-blue.svg)](https://ut.ac.ir/)
[![Department](https://img.shields.io/badge/Department-ECE-red.svg)](https://ece.ut.ac.ir/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete coursework, theoretical problem sets, analytical derivations, comprehensive reports, and programming implementations for the **Machine Learning** course at the **University of Tehran** (Department of Electrical and Computer Engineering), Fall 1404 (2025–2026).

---

## 👨‍🎓 Student & Course Information

- **Course:** Machine Learning (ECE Department)
- **Institution:** University of Tehran
- **Semester:** Fall 1404 (Fall 2025 – 2026)
- **Student:** Amirali Dehghani (`810102443`)

---

## 🗂 Repository Structure

```
Machine-Learning-Course-Fall04/
├── HW1/             # Probabilistic Modeling, Bayes Decision Theory & Naive Bayes
├── HW2/             # Optimization Theory, Search Algorithms & Heuristics
├── HW3/             # Support Vector Machines (SVM) & Decision Trees
├── HW4/             # Dimensionality Reduction, Feature Selection & Clustering
├── HW5/             # Regression, Neural Network Foundations & Deep Learning
├── Final Project/   # Spoken Language Identification (Git Submodule)
├── LICENSE
└── README.md
```

---

## 📚 Coursework & Syllabus Breakdown

Each homework folder contains the original assignment specifications (`.pdf`), complete mathematical and analytical solution reports (`ML-HW*-810102443.pdf`), and practical implementations (`.ipynb`).

---

### 🔹 [HW1 — Probabilistic Modeling & Classification Theory](./HW1)
*Covers Bayesian decision theory, parameter estimation via Maximum Likelihood, probabilistic modeling, and text classification.*

* **Theoretical & Analytical Formulations (PDF):**
  * **Bayes Decision Rule vs. Randomized Decisions:** Risk derivation under zero-one loss ($R_{\text{rand}}$ vs. $R_{\text{Bayes}}$) and proof of the non-inferiority of Bayes decision rule.
  * **Gaussian Decision Boundaries:** Analytical derivation of decision thresholds for 1D Gaussians with unequal priors and costs; derivation of 2D discriminant functions $g_1(x), g_2(x)$ and linear/quadratic boundaries.
  * **Maximum Likelihood Estimation (MLE):**
    * Parameter estimation for Bernoulli Naive Bayes feature likelihoods and class priors (proving MLE matches empirical frequencies).
    * Poisson distribution parameter estimation ($\lambda_{\text{MLE}}$) for count data.
    * Mathematical proof that maximizing Gaussian likelihood under homoscedastic noise is equivalent to minimizing Mean Squared Error (MSE).
  * **Shift in Decision Boundaries:** Analysis of decision boundary movement under changing prior probabilities ($P(\omega_1)$ vs. $P(\omega_2)$).
  * **Bayes Theorem & Discrete Naive Bayes:** Manual step-by-step posterior and prior computation for medical/drug diagnostic scenarios and tabular discrete datasets.
* **Practical & Programming Implementations (Notebook):**
  * **Spam vs. Ham Text Classification:** Tokenization, vocabulary construction, frequency filtering, Laplace smoothing, and comparative analysis of Bernoulli vs. Multinomial Naive Bayes models.
  * **Custom Heuristic Classifier:** Rule-based image classification (Cloud vs. Clear sky) using statistical color-space thresholding, evaluated via Precision, Recall, and Confusion Matrices.

---

### 🔹 [HW2 — Optimization Theory, Search Algorithms & Heuristics](./HW2)
*Focuses on continuous optimization, line search techniques, convex duality, and metaheuristic search algorithms.*

* **Theoretical & Analytical Formulations (PDF):**
  * **Line Search Optimization:** Mathematical definition of descent directions ($\nabla f(x)^T p < 0$), step length strategies (Exact, Armijo / Backtracking Line Search, Wolfe Conditions, Barzilai–Borwein).
  * **SVM Duality & Optimization:** Hard-margin primal problem, Lagrangian formulation, dual problem derivation, KKT complementary slackness, geometric interpretation of support vectors, soft-margin with slack variables ($\xi_i$), and Mercer kernelization.
  * **Bilevel Optimization:** Formulating leader-follower games, analytic substitution of the follower's response $y^*(x)$, and regularity/convexity conditions.
  * **Multi-Objective Optimization & Pareto Front:** Definition of Pareto optimality, weighted-sum scalarization ($F_w$), $\epsilon$-constraint method, and KKT mapping onto the Pareto curve.
  * **First-Order Gradient Methods:** Proof that the negative gradient is the steepest descent direction via first-order Taylor approximation; comparative analysis of Batch Gradient Descent (BGD) vs. Stochastic Gradient Descent (SGD).
  * **Normal Equations & Least-Squares Minimizer:** Formal algebraic proof that the pseudoinverse solution $x^* = (A^T A)^{-1} A^T b$ is the unique global minimizer of $\|Ax - b\|_2^2$.
  * **Second-Order Optimization (Newton & Quasi-Newton):** Derivation of the Newton step, failure modes on non-positive-definite Hessians, secant equation, and BFGS/DFP approximations; proof of quadratic convergence rate near local minima.
  * **Constrained Optimization via Lagrange Multipliers & Newton–Raphson:** Analytic Lagrangian solution and 3×3 nonlinear system Jacobian formulation for equality-constrained problems.
* **Practical & Programming Implementations (Notebook):**
  * **Genetic Algorithm (GA) — Jigsaw Puzzle Solver:** Chromosome modeling of 2D image tiles, spatial edge-compatibility fit scores, roulette-wheel selection, crossover, mutation, and local search routines.
  * **Simulated Annealing (SA) for TSP:** Combinatorial optimization for the Travelling Salesman Problem with cooling schedules and permutation neighborhood operators.
  * **Step-Size Strategies in Gradient Descent:** Empirical implementation and convergence trajectory comparison of Backtracking Line Search vs. RMSProp adaptive learning rate on the ill-conditioned Rosenbrock benchmark.

---

### 🔹 [HW3 — Support Vector Machines & Decision Trees](./HW3)
*Deep dive into maximum-margin classification, kernel methods, optimization in infinite dimensions, and tree-based models.*

* **Theoretical & Analytical Formulations (PDF):**
  * **SVM Dual Foundations & Kernel Trick:** Why the dual formulation is preferred for kernels (inner product dependence), why non-support vectors have zero dual multipliers ($\alpha_i = 0$), and kernel capacity/overfitting tradeoffs.
  * **Infinite-Dimensional Hilbert Spaces (RBF Kernel):** Taylor expansion of the Gaussian kernel $\exp(-\|x-z\|^2 / 2\sigma^2)$ proving it corresponds to an infinite-dimensional feature space.
  * **Exact Dual SVM Geometry:** Manual calculation of optimal dual variables $\alpha_i$ and primal parameters $(w, b)$ for symmetric 4-point configurations.
  * **Explicit Polynomial Feature Mapping:** Multinomial expansion of the cubic kernel $K(x, z) = (1 + x^T z)^3$ and construction of the explicit 10-dimensional feature map $\phi(x) \in \mathbb{R}^{10}$ with combinatorial scaling factors.
  * **ID3 Decision Tree Mechanics:** Step-by-step calculations of dataset entropy $H(S)$, conditional entropy, and Information Gain; top-down vs. bottom-up tree induction.
  * **Optimization Geometry & Jensen’s Inequality:** Proof of strict concavity for entropy $H(p)$ ($\frac{\partial^2 H}{\partial p^2} < 0$), proof that Information Gain is always non-negative ($\text{Gain} \ge 0$) using Jensen's Inequality, and analysis of greedy local optima traps.
  * **Failure of Greedy Search (The XOR Trap):** Proof that standard greedy ID3 fails on the XOR problem ($\text{Gain} = 0$ for all features); inductive bias analysis (Preference Bias vs. Restriction Bias).
* **Practical & Programming Implementations (Notebook):**
  * **End-to-End SVM Diagnostic Pipeline:** Preprocessing biologically implausible missing values, feature scaling sensitivity, linear vs. non-linear kernels (RBF, Polynomial, Sigmoid), grid-search hyperparameter tuning ($C, \gamma$), class-weighted SVM for imbalanced data, and 2D decision boundary visualization.
  * **Primal QP-SVM Solver from Scratch:** Formulating the soft-margin SVM primal objective as a Quadratic Program (QP) and solving directly using numerical optimization (`cvxpy` / `scipy`) without high-level ML libraries.
  * **ID3 Decision Tree Classifier from Scratch:** Handling the high-cardinality trap (`Day_ID`), splitting continuous features via dynamic threshold scanning, recursive tree construction, depth-controlled regularization, and post-pruning analysis.

---

### 🔹 [HW4 — Dimensionality Reduction, Feature Selection & Clustering](./HW4)
*Covers unsupervised learning, subspace projection, manifold learning, and feature selection pipelines.*

* **Theoretical & Analytical Formulations (PDF):**
  * **K-Means Clustering Iteration:** Manual calculation of point-to-centroid Euclidean distances, assignment step, and centroid update equations.
  * **Agglomerative Hierarchical Clustering:** Computational complexity ($O(N^2)$ to $O(N^3)$), chaining effect, and outlier sensitivity in Single vs. Complete Linkage; manual dendrogram construction.
  * **Data Geometry & Dimensionality Reduction Choice:** Matching geometric configurations (linearly correlated Gaussian, concentric circles, high class overlap, noisy features) to PCA, Kernel PCA, and LDA based on objective functions.
  * **Formal Theory of LDA & Kernel PCA:**
    * Mathematical proof that Linear Discriminant Analysis produces at most $C-1$ non-zero discriminant directions via rank properties of the between-class scatter matrix $S_B$.
    * Derivation of the Kernel PCA eigenvalue problem from the feature-space covariance operator and out-of-sample projection using kernel evaluations.
  * **Manual Computation of Fisher’s LDA:** Within-class ($S_W$) and between-class ($S_B$) scatter matrices, solving the generalized eigenvalue problem $(S_W^{-1} S_B) w = \lambda w$, and projecting 2D points onto the optimal 1D discriminant axis.
  * **Feature Selection Taxonomy:** Mechanics of Filter, Wrapper (SFS, SBS, Stepwise/Bidirectional), and Embedded methods; proof of suboptimal convergence due to greedy search behavior.
* **Practical & Programming Implementations (Notebook):**
  * **Linear & Kernel PCA on Structured Data:** Linear PCA on the Iris dataset (eigenvalue spectrum, explained variance ratio); Kernel PCA with RBF kernel for non-linear unfolding of nested circular manifolds.
  * **Feature Selection & Biomarker Discovery:** Exploratory analysis, correlation matrix filtering, univariate mutual information ranking, Recursive Feature Elimination (RFE), Cross-Validated RFECV, and Embedded Lasso ($L_1$) regularization on clinical datasets.
  * **Unsupervised Image Segmentation:** 5D spatial and color feature space engineering ($[X, Y, R, G, B]$), multi-segment image segmentation comparing K-Means clustering against Gaussian Mixture Models (GMM) with Expectation-Maximization (EM).

---

### 🔹 [HW5 — Regression, Neural Network Foundations & Deep Learning](./HW5)
*Covers linear/polynomial regression, regularization, optimization in deep architectures, weight initialization, and neural network theory.*

* **Theoretical & Analytical Formulations (PDF):**
  * **MLE for Power-Law Distributions:** Analytical derivation of $\theta_{\text{MLE}}$ for Pareto/power distributions $P(x|\theta) = \theta x^{-\theta-1}$.
  * **Logistic Regression & Gradient Derivation:** Sigmoid derivative properties, binary cross-entropy loss, and step-by-step derivation of the SGD parameter update equation.
  * **$L_2$-Regularized Regression with Newton’s Method:** Matrix calculus proof that a single Newton step from any initial weight vector $w_0$ lands exactly on the optimal closed-form Ridge solution $w^* = (X^T X + \lambda I)^{-1} X^T y$.
  * **Perceptron Convergence & Linear Separability:** Perceptron weight updates, cycle behavior on non-linearly separable data, and feature transformations $[x, y] \to [x, y, x^2, 1]$ to achieve linear separability.
  * **Neural Network Expressive Power:** Determining representational capacity for linear, affine, ReLU, and composite computation graphs ($G_1 - G_5$).
  * **Loss Functions & Cross-Entropy Equivalence:** Derivation of multi-class Cross-Entropy from Maximum Likelihood; loss formulations for noisy label distributions and $[-1, +1]$ target encodings.
  * **Stochastic Stability & The "Dead ReLU" Phenomenon:** Statistical derivation of the pre-activation distribution $z \sim \mathcal{N}(b, \sigma_z^2)$, activation probability $P(a > 0) = \Phi(b/\sigma_z)$, and proof of permanent neuron inactivation under negative bias initialization ($b = -3\sigma_z$).
  * **Variance Preservation & Xavier (Glorot) Initialization:** Forward-pass variance preservation ($\text{Var}(W) = 1/n_{\text{in}}$), backward-pass gradient variance preservation ($\text{Var}(W) = 1/n_{\text{out}}$), derivation of the Xavier variance $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$, and proof of exponential variance vanishing/exploding in deep architectures.
* **Practical & Programming Implementations (Notebook):**
  * **Polynomial Regression & Regularization:** Empirical polynomial degree exploration ($d \in [1, 9]$), Bias-Variance tradeoff analysis, overfitting demonstration on training vs. validation splits, and Ridge ($L_2$) penalty regularization tuning ($\lambda$).
  * **Transfer Learning & Deep CNN Benchmarking:** CIFAR-10 data pipeline, Multi-Layer Perceptron (MLP) baseline, and Transfer Learning using a pre-trained VGG-11 convolutional network (comparing frozen feature extractors vs. fine-tuned networks).

---

### 🔹 [Final Project — Spoken Language Identification (LID)](./Final%20Project)
> *Integrated as a Git Submodule pointing to the dedicated project repository: [Spoken-Language-Identification](https://github.com/Dectre/Spoken-Language-Identification).*

* **Task:** End-to-end acoustic Spoken Language Identification across **German**, **Italian**, **Korean**, and **Spanish**.
* **Dataset & Audio Preprocessing:**
  * 10,800+ audio segments resampled to 16 kHz Mono.
  * Voice Activity Detection (VAD at 20 dB threshold) for silent pause removal.
  * Source-based tracking (`group_id`) to eliminate data leakage across train/test sets using `GroupShuffleSplit`.
* **Feature Extraction:** 44 temporal and spectral acoustic descriptors (20 MFCCs, Spectral Centroid, Zero-Crossing Rate, aggregations over time).
* **Supervised & Unsupervised Modeling:**
  * Supervised multi-class classification with K-Nearest Neighbors (KNN), Random Forest (RF), and Support Vector Machines (SVM).
  * Cross-gender domain adaptation experiment (evaluating speaker-specific vs. language-invariant acoustic cues).
  * Unsupervised linguistic clustering via PCA, K-Means, and Agglomerative Hierarchical Clustering (Ward linkage).

---

## ⚙️ Submodule Setup & Cloning

To clone this repository with all its submodules (including the Final Project), run:

```bash
git clone --recursive https://github.com/Dectre/Machine-Learning-Course-Fall04.git
```

If the repository has already been cloned without the submodule:

```bash
git submodule update --init --recursive
```

---

## 🛠 Prerequisites & Environment

- **Python:** 3.10+
- **Core & Scientific Computing:** `numpy`, `scipy`, `pandas`, `cvxopt` / `cvxpy`
- **Machine Learning & Preprocessing:** `scikit-learn`
- **Deep Learning:** `torch`, `torchvision`
- **Audio Processing (Final Project):** `librosa`, `soundfile`
- **Visualization:** `matplotlib`, `seaborn`
- **Notebook Environment:** `jupyter`, `ipykernel`

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).