# Chaid---CART---BAGGING---Random-Forest

# Tree-Based Classification of Wine Quality (CHAID, CART, Bagging, Random Forest)

## 1. Project Overview

This project develops and compares **tree-based classification models** to segment wines into ordered quality classes using only their **chemical composition**.

Starting from a dataset of **178 wines** with **13 quantitative chemical attributes** and an original 5-class quality label, the analysis:

- recodes the target into **three macro-classes** (high, medium, low quality);
- builds **CHAID decision trees** under different discretization schemes;
- evaluates models via **Train/Test split** and **5-fold Cross Validation**;
- explores **CART with optimal pruning**, **Bagging**, and **Random Forest**;
- identifies the **best-performing model** and compares its rules with interpretable CHAID trees.

The code is implemented in **R** in a single script:

- `Chaid_Cart_Bagging_RF.R`

and is accompanied by a full written report:

- `Report.pdf` – detailed methodology, tables, and graphical analysis.

---

## 2. Data Description

The dataset (assumed to be stored as `wines2.txt`) contains:

- **178 observations** (wines);
- **13 continuous predictors** describing chemical composition:

  - `Alcohol`
  - `Malic.acid`
  - `Ash`
  - `Alcalinity.of.ash`
  - `Magnesium`
  - `Total.phenols`
  - `Flavanoids`
  - `Nonflavanoid.phenols`
  - `Proanthocyanins`
  - `Color.intensity`
  - `Hue`
  - `OD280.OD315.of.diluted.wines`
  - `Proline`

- **Original target variable**: `Classificazione` with 5 ordered levels.

### 2.1 Target Recoding

The response variable is recoded into **three ordered macro-classes**:

- **First Class (0,2]** – aggregation of original classes 1 and 2 (“high quality” wines);
- **Second Class (2,3]** – original class 3 (“medium quality” wines);
- **Third Class (3,5]** – aggregation of original classes 4 and 5 (“low quality” wines).

This recoding simplifies interpretation while preserving the underlying rank structure of wine quality.

Only **chemical variables** are used as predictors: subjective sommelier ratings and demographic preferences are deliberately excluded to maintain **objectivity, generalisability and reproducibility** of the models.

---

## 3. CHAID Methodology and Discretization Strategies

The core of the project is the application of **CHAID** (Chi-squared Automatic Interaction Detection), which builds multiway decision trees based on **χ² tests** on categorical predictors. Since the predictors are continuous, they must first be **discretized**.

The script defines a set of **helper functions** to discretize continuous variables and evaluate performance:

- `find_split_chisq()` – optimal binary split based on χ² statistic;
- `find_split_gini()` – optimal split based on Gini impurity;
- `find_split_entropy()` – optimal split based on information gain (entropy);
- `apply_binary()` – Low/High factor encoding for a given cut-point;
- `get_robust_breaks()` – robust quartile-based breaks extended to `(-Inf, +Inf)` to handle out-of-range future values;
- `print_performance()` – confusion matrix, accuracy and error rate for any fitted model.

### 3.1 Binary CHAID (Chi-square Split)

Each chemical variable is discretized into two classes:

- **Low** vs **High**, with the cut-point chosen to **maximise χ²** with respect to the target.

Three performance scenarios are considered:

1. **Full-sample training (resubstitution)**  
   - high apparent accuracy (~91%) but optimistic due to overfitting.

2. **Train/Test split (70/30)**  
   - error increases to about 17%, revealing the rigidity of a single binary cut and the loss of information.

3. **5-fold Cross Validation**  
   - average error ≈ 18%; confirms that forced binarisation leads to unstable models and poor generalisation.

### 3.2 CHAID with Quantile (Quartile) Discretization

Here each continuous predictor is discretized into up to **four ordered levels**:

- `Q1`, `Q2`, `Q3`, `Q4` (quartiles).

To ensure robustness for future data outside the observed range, the first and last intervals are extended to `(-Inf, Q1]` and `(Q4, +Inf)`.

Results:

- **Resubstitution accuracy** ≈ 89.9% (error ≈ 10.1%);
- **Train/Test error** ≈ 17.3%;
- **5-fold CV error** ≈ 13.5%, but with strong variability across folds (2.8%–19.4%).

The trees become **deep and fragmented**, with many small terminal nodes and unstable secondary splits, particularly affecting the **intermediate class (2,3]**.

### 3.3 Advanced Discretization Methods: Gini, Entropy, Interval

To improve robustness and predictive power, three further discretization schemes are evaluated via 5-fold CV:

1. **Gini-based splits** – supervised, choose thresholds minimising Gini impurity.
2. **Entropy-based splits** – supervised, maximise information gain.
3. **Interval discretization** – unsupervised, divides the range into **three fixed-width intervals** (Low, Medium, High), independent of the target.

Average cross-validated errors:

- **Interval** – ~11.25% (**best performer**)
- Entropy – ~14.6%
- Gini – ~15.8%
- Quartiles – ~13.5%
- Binary (χ²) – ~17.9%

The **Interval** method works as a **natural regulariser**: by ignoring the target when defining intervals, it avoids overfitting to the training distribution and yields **more stable and generalisable cut-points**.

---

## 4. Final CHAID Model (Interval Discretization)

Once Interval is identified as the best discretization method, all chemical variables are discretized into three fixed intervals on the **full dataset**, and a final CHAID tree is fitted.

### 4.1 Performance

Confusion matrix (resubstitution):

- Global **accuracy** ≈ 89.9%, **error** ≈ 10.1%.
- Excellent discrimination of extreme classes:
  - high-quality wines and low-quality wines are correctly identified with very high purity;
  - remaining misclassifications concentrate in the **medium class (2,3]**, typical of ordinal problems with continuous predictors.

### 4.2 Variable Significance (χ² Tests)

A global **Chi-squared independence test** is computed between the recoded target and each discretized predictor. Highly significant variables include:

- `OD280.OD315.of.diluted.wines`
- `Proline`
- `Hue`
- `Flavanoids`
- `Color.intensity`
- `Alcalinity.of.ash`
- `Proanthocyanins`
- `Nonflavanoid.phenols`
- `Magnesium`
- `Malic.acid` (just significant)

Variables such as `Alcohol`, `Ash`, and `Total.phenols` show **non-significant** p-values and do not appear among the main splitters.

### 4.3 Tree Structure and Decision Rules

The final CHAID tree (Interval) has a **clean hierarchical structure** up to five levels. Key rules:

- **Root node – `OD280` (diluted wine spectrum)**  
  - Low OD280 → left branch: associated with lower structural quality.
  - High OD280 → right branch: associated with structurally richer wines.

- **Left branch – alkaline and colour-based segmentation**
  - High `Alcalinity.of.ash` and high `Color.intensity` isolate a **pure low-quality cluster**.

- **Right branch – aminoacid richness and flavonoids**
  - High `Proline` identifies a **premium cluster** with almost only high-quality wines.
  - For medium/low `Proline`, `Flavanoids` separates **standard** wines from residual high-quality cases.

This results in:

- a **pure low-quality node** (100% class (3,5]);
- a **dominant high-quality node** (mostly class (0,2], no low-quality wines);
- a main **medium-quality node** collecting the majority of standard wines.

### 4.4 Simulation on Blind (Unlabelled) Wines

A small **blind dataset** of 5 artificial wines (with new combinations and out-of-range values) is classified using the final CHAID tree:

- structurally strong wines (high OD280) are classified as **Premium**;
- weak wines (low OD280) are classified as **Entry-level**;
- a borderline case with high Proline but low OD280 is conservatively assigned to **Standard quality**.

The simulation confirms that the model behaves as a **conservative quality control system**, where insufficient structural properties prevent Premium classification regardless of other favourable attributes.

---

## 5. CART, Bagging and Random Forest

To benchmark CHAID against other tree-based methods, the script implements:

- **CART** (Classification and Regression Trees) with **cost-complexity pruning**;
- **Bagging** (bootstrap aggregation of trees);
- **Random Forest**.

The working dataset for these models uses the **original continuous predictors** and the 3-class target `Class3`.

### 5.1 CART with Optimal Pruning

Using `rpart`:

1. A **maximal tree** is grown on a 70% training set (`cp` very small).
2. The **CP table** is inspected and the CP value with lowest cross-validated error (`xerror`) is selected.
3. The tree is **pruned** (simplified) at this optimal CP.
4. Performance is evaluated on the 30% **test set**, together with variable importance.

This provides a baseline for single-tree performance.

### 5.2 Bagging

Using `randomForest` with `mtry = number_of_predictors`:

- many full trees are grown on bootstrap samples;
- predictions are aggregated by majority vote;
- **OOB (out-of-bag) error** and **test-set accuracy** are computed;
- variable importance is assessed via Gini decrease or permutation importance.

### 5.3 Random Forest

Random Forest is trained with:

- `mtry` smaller than the number of predictors (random subset of variables at each split);
- a sufficiently large number of trees (e.g. 500).

Key outputs:

- **OOB error rate**;
- confusion matrix on the **test set**;
- **variable importance plot**, highlighting which chemical properties drive predictive power.

---

## 6. Final Model Comparison

All models are compared in terms of:

- **misclassification error** on the test set;
- stability across resampling;
- interpretability of decision rules.

Main conclusions:

- The **CHAID Interval** model provides **very clear and interpretable rules**, with excellent separation of extreme quality classes and robust discretization.
- **Random Forest** typically achieves the **lowest error rate**, leveraging ensemble averaging and random subspace selection.
- Bagging improves over a single CART tree but offers less gain compared to Random Forest.
- There is a trade-off between **predictive accuracy** (Random Forest) and **interpretability & rule transparency** (CHAID Interval).

The report concludes with a detailed analytical comparison between **CHAID Interval** and **Random Forest**, emphasising when a fully interpretable model may be preferred to a black-box ensemble in business or quality-control contexts.

---

## 7. Repository Structure

A suggested structure for the GitHub repository is:

```text
.
├─ README.md                     # This file
├─ Report.pdf                    # Full written report (Italian)
├─ Chaid_Cart_Bagging_RF.R       # R script with all models
├─ wines2.txt                    # Wine dataset (178 rows, 13 variables + target)
└─ figs/                         # (Optional) exported trees and plots
   ├─ chaid_interval_tree.png
   ├─ cart_pruned_tree.png
   ├─ rf_var_importance.png
   └─ ...
