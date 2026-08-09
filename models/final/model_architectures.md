# 🏗️ Model Architectures & Detailed Execution Pipeline

This document provides the full technical specifications, hyperparameter search details, and comprehensive execution results for the ransomware detection models.

## 🧪 Data Processing Pipeline (Common to All)

All models share the following preprocessing pipeline to ensure consistency and prevent data leakage:

1. **Normalization:** Raw execution traces are normalized to **100 rows** using linear interpolation.
2. **Temporal Windowing:** A sliding window (Size=10, Stride=5) is applied.
3. **Statistical Summary:** For each window, 4 statistics are computed for every feature: $\text{Mean}, \text{Std}, \text{Max}, \text{Min}$.
4. **Feature Selection:** 
   - **Binary Task:** Top 10 features selected via `ExtraTreesClassifier` importance on the training set.
   - **Multiclass Task:** Top 30 features selected via `ExtraTreesClassifier` importance on the training set.
5. **Scaling:** `MinMaxScaler` (0 to 1) is applied to all features before entering Quantum or SVM models.

---

## 📉 Binary Classification Models

Evaluation conducted on an independent test set containing real-world ransomware samples.

### 1. Classical RF (ExtraTrees)
- **Architecture:** `ExtraTreesClassifier` with balanced class weights.
- **Grid Search:** `n_estimators` [100, 500], `max_depth` [None, 10], `min_samples_split` [2, 5], `max_features` ["sqrt", None].
- **Best Fit:** `{'rf__max_depth': None, 'rf__max_features': 'sqrt', 'rf__min_samples_split': 5, 'rf__n_estimators': 100}`.
- **Detailed Metrics:**
  - **Accuracy:** 72.73%
  - **Goodware:** Precision: 1.00 | Recall: 0.40 | F1: 0.57
  - **Ransomware:** Precision: 0.67 | Recall: 1.00 | F1: 0.80

### 2. Quantum SVM (QSVM)
- **Architecture:** Learnable Projected Quantum Kernel $\rightarrow$ SVC.
- **Kernel Logic:** Optimized via **Kernel Target Alignment (KTA)** loss.
- **Grid Search:** `n_qubits` [4, 8], `n_layers` [2, 3], `lr` [0.01, 0.03], `C` [1.0, 10.0].
- **Best Fit:** `{'C': 10.0, 'lr': 0.01, 'n_layers': 2, 'n_qubits': 4}`.
- **Detailed Metrics:**
  - **Accuracy:** 73.68%
  - **Note:** Achieved near-perfect ransomware recall (~100%) with moderate specificity.

### 3. Classical SVM (Baseline)
- **Architecture:** `SVC` with RBF Kernel.
- **Grid Search:** `C` [0.1, 1, 10, 100], `gamma` ["scale", "auto", 0.001, 0.01, 0.1].
- **Best Fit:** `{'svm__C': 10, 'svm__gamma': 'auto'}`.
- **Detailed Metrics:**
  - **Accuracy:** 25.84%
  - **Goodware:** Precision: 0.27 | Recall: 0.37 | F1: 0.31
  - **Ransomware:** Precision: 0.24 | Recall: 0.17 | F1: 0.20
  - **Analysis:** Severe overfitting (CV Acc: 98.41% vs Test Acc: 25.84%).

---

## 📂 Multiclass Classification Models

Evaluation conducted using **Leave-One-Run-Out (3-fold)** cross-run strategy on the same set of binaries.

### 1. Classical RF (ExtraTrees)
- **Architecture:** `ExtraTreesClassifier` with balanced class weights.
- **Grid Search:** (Same as Binary).
- **Detailed Metrics:**
  - **Fold 1 Accuracy:** 81.82%
  - **Fold 2 Accuracy:** 91.39%
  - **Fold 3 Accuracy:** 91.39%
  - **Mean Window Acc:** **88.20%**

### 2. Quantum SVM (QSVM)
- **Architecture:** Learnable Projected Quantum Kernel $\rightarrow$ SVC.
- **Grid Search:** `n_qubits` [4, 8], `n_layers` [2], `lr` [0.01, 0.03].
- **Detailed Metrics:**
  - **Fold 1 Accuracy:** 86.12%
  - **Fold 2 Accuracy:** 87.08%
  - **Fold 3 Accuracy:** 80.38%
  - **Mean Window Acc:** **84.53%**

### 3. Classical SVM (Baseline)
- **Architecture:** `SVC` with RBF Kernel.
- **Grid Search:** (Same as Binary).
- **Detailed Metrics:**
  - **Fold 1 Accuracy:** 89.47%
  - **Fold 2 Accuracy:** 78.95%
  - **Fold 3 Accuracy:** 78.95%
  - **Mean Window Acc:** **82.46%**

---

## 🧪 Hybrid Model (Quantum $\rightarrow$ RF)

- **Architecture:** Fixed Quantum Feature Map $\rightarrow$ `ExtraTreesClassifier`.
- **Process:** Input $\rightarrow$ Hilbert Space $\rightarrow$ Observables $\rightarrow$ RF.
- **Detailed Metrics:**
  - **Binary Accuracy:** 44.02% (Precision Good: 0.43, Recall Good: 0.77).
  - **Multiclass Accuracy:** 16.91%.
- **Conclusion:** Ineffective without a learnable kernel (KTA).

---

## 📊 Final Performance Summary

| Task | RF (Classical) | QSVM (Quantum) | SVM (Baseline) |
| :--- | :---: | :---: | :---: |
| **Binary Acc** | 72.73% | **73.68%** | 25.84% |
| **Multiclass Acc** | **88.20%** | 84.53% | 82.46% |
