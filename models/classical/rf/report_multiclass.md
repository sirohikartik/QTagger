# Random Forest Cross-Run Evaluation

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Classifier | ExtraTreesClassifier |
| Number of Trees | 500 |
| Feature Selection | ExtraTrees Importance |
| Selected Features | Top 10 |
| Missing Value Handling | Median Imputation |
| Window Size | 10 |
| Window Stride | 5 |
| Normalized Rows | 100 |
| Random State | 42 |
| Evaluation | Leave-One-Run-Out (3 Folds) |

---

## Cross-Run Results

| Fold | Training Runs | Testing Run | Window Accuracy | Sample Accuracy |
|------|---------------|-------------|----------------:|----------------:|
| Fold 1 | features2 + features3 | features | **81.82%** | **81.82%** |
| Fold 2 | features + features3 | features2 | **90.43%** | **90.91%** |
| Fold 3 | features + features2 | features3 | **90.43%** | **90.91%** |

---

## Overall Performance

| Metric | Value |
|--------|------:|
| Mean Window Accuracy | **87.56%** |
| Std. Window Accuracy | **4.06%** |
| Mean Sample Accuracy | **87.88%** |
| Std. Sample Accuracy | **4.28%** |

---

## Observations

- Fold 1 produced the lowest performance (81.82%).
- Folds 2 and 3 achieved identical window accuracies (90.43%).
- Mean cross-run accuracy was 87.56%, indicating good robustness across independent executions.
- Feature selection was performed independently within each fold using only the corresponding training data, preventing data leakage.
- The Random Forest generalized well across different executions of the same binaries.
