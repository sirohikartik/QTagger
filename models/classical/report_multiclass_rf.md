# Random Forest Cross-Run Evaluation

## Experimental Setup

| Parameter              | Value                       |
| ---------------------- | --------------------------- |
| Classifier             | ExtraTreesClassifier        |
| Number of Trees        | 500                         |
| Feature Selection      | ExtraTrees Importance       |
| Selected Features      | Top 30                      |
| Missing Value Handling | Median Imputation           |
| Window Size            | 10                          |
| Window Stride          | 5                           |
| Normalized Rows        | 100                         |
| Random State           | 42                          |
| Evaluation             | Leave-One-Run-Out (3 Folds) |

---

## Cross-Run Results

| Fold   | Training Runs         | Testing Run | Window Accuracy | Sample Accuracy |
| ------ | --------------------- | ----------- | --------------: | --------------: |
| Fold 1 | features2 + features3 | features    |      **79.90%** |      **81.82%** |
| Fold 2 | features + features3  | features2   |      **89.47%** |      **90.91%** |
| Fold 3 | features + features2  | features3   |      **89.47%** |      **90.91%** |

---

## Overall Performance

| Metric               |      Value |
| -------------------- | ---------: |
| Mean Window Accuracy | **86.28%** |
| Std. Window Accuracy |  **4.51%** |
| Mean Sample Accuracy | **87.88%** |
| Std. Sample Accuracy |  **4.28%** |

---

## Observations

* Fold 1 produced the lowest performance (79.90% window accuracy).
* Folds 2 and 3 achieved identical window accuracies (89.47%) and sample accuracies (90.91%).
* The Random Forest achieved a mean window accuracy of **86.28%** and a mean sample accuracy of **87.88%** across the three leave-one-run-out folds.
* Feature selection was performed independently within each fold using only the corresponding training runs, preventing data leakage.
* Increasing the number of selected features from **10 to 30** resulted in only a marginal change in overall performance, indicating that the additional features contributed limited discriminative information beyond the most informative subset.
* The classifier generalized well across independent executions of the same binaries, demonstrating robustness to run-to-run behavioural variability.
