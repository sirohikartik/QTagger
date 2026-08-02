T# Model Evaluation Report (Top 10 Features)

## 📊 Overview & Configuration
- **Task:** Binary classification (Goodware vs. Ransomware)
- **Evaluation Level:** Window-level and Sample-level
- **Decision Threshold (Sample Level):** `0.40`
- **Window-level Accuracy:** `0.4545`

---

## 🔍 Top 10 Features Selected
- `io_burstiness_std_std`
- `io_burstiness_std_max`
- `window_start_s_std`
- `file_deletion_rate_mean`
- `file_deletion_rate_max`
- `file_deletion_rate_min`
- `time_to_first_encryption_s_std`
- `window_start_s_max`
- `file_deletion_rate_std`
- `window_start_s_mean`

---

## 📈 Window-Level Evaluation

### Classification Report
```text
              precision    recall  f1-score   support

    goodware       0.41      0.46      0.44        95
  ransomware       0.50      0.45      0.47       114

    accuracy                           0.45       209
   macro avg       0.46      0.46      0.45       209
weighted avg       0.46      0.45      0.46       209

### Confusion Matrix
| | Pred Goodware | Pred Ransomware |
| :--- | :---: | :---: |
| **Actual Goodware** | 44 | 51 |
| **Actual Ransomware** | 63 | 51 |

---

## 🎯 Per-Sample Results (Probability Average @ 0.40)

| Sample Name | True Label | Pred Label | Result |
| :--- | :--- | :--- | :---: |
| `4dc06ecee904b9165fa699b026045c1b6408cc7061df3d2a7bc2b7b4f0879f4d` | ransomware | ransomware | ✓ |
| `52fc723f7e0c4202c97ac5bc2add2d1d3daa5c3f84f3d459a6a005a3ae380119` | ransomware | goodware | ✗ |
| `5677dfad26045e271272bc98be2fd24e2f6d13737850ab1d9857fd58de05e9f9` | ransomware | ransomware | ✓ |
| `90b06f07eb75045ea3d4ba6577afc9b58078eafeb2cdd417e2a88d7ccf0c0273` | ransomware | ransomware | ✓ |
| `cd27a31e618fe93df37603e5ece3352a91f27671ee73bdc8ce9ad793cad72a0f` | ransomware | goodware | ✗ |
| `curl` | goodware | goodware | ✓ |
| `e5834b7bdd70ec904470d541713e38fe933e96a4e49f80dbfb25148d9674f957` | ransomware | ransomware | ✓ |
| `gpg` | goodware | ransomware | ✗ |
| `rsync` | goodware | goodware | ✓ |
| `sqlite` | goodware | ransomware | ✗ |
| `stressng` | goodware | ransomware | ✗ |
