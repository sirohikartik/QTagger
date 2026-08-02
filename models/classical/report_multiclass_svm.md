# Classical SVM Cross-Run Evaluation (Multiclass)

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Classifier | Support Vector Machine (RBF Kernel) |
| Hyperparameter Search | GridSearchCV (5-fold) |
| Feature Selection | ExtraTrees Feature Importance |
| Selected Features | Top 30 |
| C Search Space | {0.1, 1, 10, 100, 1000} |
| Gamma Search Space | {scale, auto, 0.001, 0.01, 0.1, 1} |
| Missing Value Handling | Median Imputation |
| Feature Scaling | Min-Max Scaling [0,1] |
| Window Size | 10 |
| Window Stride | 5 |
| Normalized Rows | 100 |
| Random State | 42 |
| Evaluation | Leave-One-Run-Out (3 Folds) |

---

## Cross-Run Results

| Fold | Training Runs | Testing Run | Best Parameters | Window Accuracy | Sample Accuracy |
|------|---------------|-------------|----------------|----------------:|----------------:|
| Fold 1 | features2 + features3 | features | C=100, γ=scale | **80.86%** | **81.82%** |
| Fold 2 | features + features3 | features2 | C=1000, γ=scale | **81.34%** | **81.82%** |
| Fold 3 | features + features2 | features3 | C=1000, γ=scale | **83.73%** | **81.82%** |

---

## Overall Performance

| Metric | Value |
|--------|------:|
| Mean Window Accuracy | **81.98%** |
| Std. Window Accuracy | **1.24%** |
| Mean Sample Accuracy | **81.82%** |
| Std. Sample Accuracy | **0.00%** |

---

## Best Hyperparameters

| Fold | C | Gamma |
|------|--:|--------|
| Fold 1 | 100 | scale |
| Fold 2 | 1000 | scale |
| Fold 3 | 1000 | scale |

---

## Observations

- The classical SVM generalized well across independent executions of the same binaries.
- Mean window-level accuracy reached **81.98%** across the three folds.
- Sample-level accuracy remained constant at **81.82%** for all folds.
- Feature selection was performed independently within each fold using only the corresponding training data, preventing information leakage.
- Hyperparameters were optimized separately for every fold using GridSearchCV.
- The classifier achieved strong performance for several binaries (e.g., `sqlite`, `stressng`, `curl`, `rsync`) while consistently confusing certain ransomware families with each other.
- Compared with the binary ransomware detection task, multiclass identification proved substantially easier for the SVM because each binary exhibits a distinctive execution profile.
