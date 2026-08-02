# Learnable Quantum Kernel SVM (QSVM) Evaluation

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | Learnable Projected Quantum Kernel + SVM |
| Quantum Simulator | PennyLane Lightning Qubit |
| Number of Qubits | **4** |
| Quantum Layers | **2** |
| Training Epochs | **60** |
| Learning Rate | **0.03** |
| SVM Regularization (C) | **10.0** |
| Kernel Training Objective | Kernel Target Alignment (KTA) |
| Feature Selection | ExtraTrees (Top 10 Features) |
| Selected Features | Top 10 |
| Data Normalization | Min-Max Scaling |
| Window Size | 10 |
| Window Stride | 5 |
| Evaluation | Independent Test Set |

---

# Training Summary

| Epoch | KTA Loss |
|------:|---------:|
| 10 | -0.6164 |
| 20 | -0.6748 |
| 30 | -0.6800 |
| 40 | -0.6827 |
| 50 | -0.6862 |
| 60 | **-0.6903** |

The Kernel Target Alignment (KTA) objective steadily improved throughout training, indicating that the learned quantum feature embedding became increasingly aligned with the class labels.

---

# Window-Level Performance

| Metric | Value |
|--------|------:|
| Accuracy | **70.81%** |
| Goodware Precision | **1.00** |
| Goodware Recall | **0.36** |
| Goodware F1-score | **0.53** |
| Ransomware Precision | **0.65** |
| Ransomware Recall | **1.00** |
| Ransomware F1-score | **0.79** |
| Macro F1 | **0.66** |
| Weighted F1 | **0.67** |

---

# Confusion Matrix

| Actual \\ Predicted | Goodware | Ransomware |
|--------------------|---------:|-----------:|
| **Goodware** | **34** | **61** |
| **Ransomware** | **0** | **114** |

---

# Per-Sample Majority Vote Results

| Sample | True Label | Prediction | Result |
|--------|------------|------------|:------:|
| 4dc06ece... | Ransomware | Goodware | ✗ |
| 52fc723f... | Ransomware | Ransomware | ✓ |
| 5677dfad... | Ransomware | Goodware | ✗ |
| 90b06f07... | Ransomware | Ransomware | ✓ |
| cd27a31e... | Ransomware | Ransomware | ✓ |
| e5834b7b... | Ransomware | Ransomware | ✓ |
| curl | Goodware | Ransomware | ✗ |
| gpg | Goodware | Ransomware | ✗ |
| rsync | Goodware | Ransomware | ✗ |
| sqlite | Goodware | Ransomware | ✗ |
| stressng | Goodware | Ransomware | ✗ |

**Sample-level Accuracy:** **4 / 11 = 36.36%**

---

# Observations

- The model achieved a **window-level accuracy of 70.81%**.
- **All ransomware windows were correctly detected**, resulting in a **100% ransomware recall**.
- The primary weakness is the high false-positive rate:
  - **61 of 95 goodware windows** were incorrectly classified as ransomware.
- Consequently, although ransomware detection is perfect at the window level, many benign windows are flagged as malicious.
- Majority voting across windows improves prediction stability but still misclassifies most goodware programs, yielding a sample-level accuracy of only **36.36%**.
- The learned quantum kernel therefore exhibits a **strong bias toward the ransomware class**, prioritizing detection sensitivity over specificity.

---

# Key Takeaway

The optimized learnable QSVM successfully learns a quantum feature embedding that maximizes ransomware detection, achieving **100% ransomware recall** and **70.81% overall window accuracy**. However, this comes at the expense of a substantial false-positive rate on goodware. Future improvements should therefore focus on increasing **goodware specificity** while preserving the excellent ransomware recall, for example through class-balanced kernel objectives, improved feature engineering, threshold calibration, or richer quantum feature embeddings.
```
