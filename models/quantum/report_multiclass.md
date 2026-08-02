# Quantum Support Vector Machine (QSVM) Cross-Run Evaluation

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Classifier | Learnable Projected Quantum Kernel + SVM |
| Quantum Simulator | PennyLane Lightning Qubit |
| Number of Qubits | 8 |
| Quantum Layers | 2 |
| Training Epochs | 60 |
| Learning Rate | 0.03 |
| SVM Regularization (C) | 10.0 |
| Feature Selection | ExtraTrees Importance |
| Selected Features | Top 30 |
| Missing Value Handling | Median Imputation |
| Window Size | 10 |
| Window Stride | 5 |
| Normalized Rows | 100 |
| Evaluation | Leave-One-Run-Out (3 Folds) |

---

# Cross-Run Results

| Fold | Training Runs | Testing Run | Window Accuracy | Sample Accuracy |
|------|---------------|-------------|----------------:|----------------:|
| Fold 1 | features2 + features3 | features | **72.25%** | **72.73%** |
| Fold 2 | features + features3 | features2 | **77.51%** | **81.82%** |
| Fold 3 | features + features2 | features3 | **77.99%** | **81.82%** |

---

# Overall Performance

| Metric | Value |
|--------|------:|
| Mean Window Accuracy | **75.92%** |
| Std. Window Accuracy | **2.61%** |
| Mean Sample Accuracy | **78.79%** |
| Std. Sample Accuracy | **4.28%** |

---

# Observations

- The proposed learnable QSVM achieved a mean **window-level accuracy of 75.92%** across the three independent evaluation folds.
- Mean **sample-level accuracy reached 78.79%**, demonstrating that aggregating predictions across execution windows improves robustness.
- Fold 1 produced the lowest performance (72.25%), while Folds 2 and 3 exceeded 77% window accuracy.
- Feature selection was performed independently within each fold using only the corresponding training data, preventing data leakage.
- The learnable quantum embedding generalized consistently across different executions of the same binaries despite runtime variability.
- Compared with the binary ransomware detection task, multiclass classification proved substantially more challenging because the classifier must distinguish between multiple ransomware families and several benign applications rather than separating only benign and malicious behavior.
- Increasing the number of selected features from 20 to 30 produced only a marginal improvement in window accuracy, suggesting that the quantum feature map had largely saturated and additional classical features contributed limited discriminative information.
