# Classical SVM (RBF Kernel) – Binary Ransomware Detection

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Classifier | Support Vector Machine (RBF Kernel) |
| Hyperparameter Search | GridSearchCV (5-fold) |
| Feature Selection | ExtraTrees Feature Importance |
| Selected Features | Top 10 |
| C Search Space | {0.1, 1, 10, 100, 1000} |
| Gamma Search Space | {scale, auto, 0.001, 0.01, 0.1, 1} |
| Best Parameters | C = 10, γ = auto |
| Missing Value Handling | Median Imputation |
| Feature Scaling | Min-Max Scaling [0,1] |
| Window Size | 10 |
| Window Stride | 5 |
| Normalized Rows | 100 |
| Random State | 42 |
| Evaluation | Independent Train/Test Split |

---

## Overall Performance

| Metric | Value |
|--------|------:|
| Window Accuracy | **36.36%** |

---

## Classification Report

| Class | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| Goodware | 0.38 | 0.60 | 0.46 |
| Ransomware | 0.33 | 0.17 | 0.22 |

---

## Confusion Matrix

| Actual \\ Predicted | Goodware | Ransomware |
|---------------------|---------:|-----------:|
| Goodware | **57** | 38 |
| Ransomware | **95** | 19 |

---

## Sample-Level Results

| Metric | Value |
|--------|------:|
| Correct Samples | 6 / 11 |
| Sample Accuracy | **54.55%** |

Correct Predictions

- ✓ 90b06...
- ✓ cd27...
- ✓ curl
- ✓ gpg
- ✓ rsync
- ✓ sqlite

Incorrect Predictions

- ✗ 4dc06...
- ✗ 52fc72...
- ✗ 5677df...
- ✗ e5834...
- ✗ stressng

---

## Observations

- The optimized RBF SVM struggled to separate ransomware from benign software.
- The model exhibited high recall for the goodware class (60%) but detected only 17% of ransomware windows.
- Most ransomware windows were incorrectly classified as goodware.
- Although six of the eleven binaries were classified correctly at the sample level using majority voting, overall binary detection performance remained substantially below the Random Forest baseline.
