# 📊 Results

## Top 10 Selected Features

| Rank | Feature |
|-----:|---------|
| 1 | `rw_volume_ratio_min` |
| 2 | `window_start_s_std` |
| 3 | `syscall_sequence_entropy_min` |
| 4 | `file_deletion_rate_max` |
| 5 | `file_deletion_rate_mean` |
| 6 | `bytes_read_per_sec_min` |
| 7 | `seq_vs_random_disk_ratio_max` |
| 8 | `seq_vs_random_disk_ratio_mean` |
| 9 | `execve_rate_max` |
| 10 | `syscall_sequence_entropy_mean` |

---

## Window-Level Performance

### Accuracy

| Metric | Value |
|--------|------:|
| Accuracy | **72.73%** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:-------------|----------:|-------:|---------:|--------:|
| Goodware | 1.00 | 0.40 | 0.57 | 95 |
| Ransomware | 0.67 | 1.00 | 0.80 | 114 |

| Overall Metric | Value |
|----------------|------:|
| Accuracy | **72.73%** |
| Macro Avg F1 | **0.69** |
| Weighted Avg F1 | **0.70** |

---

## Confusion Matrix

| | Predicted Goodware | Predicted Ransomware |
|:---------------------|--------------------:|---------------------:|
| **Actual Goodware** | 38 | 57 |
| **Actual Ransomware** | 0 | 114 |

---

## Execution-Level Results

| Sample | True Label | Prediction | Status |
|:-------|:-----------|:-----------|:------:|
| `4dc06ece...` | Ransomware | Ransomware | ✅ |
| `52fc723f...` | Ransomware | Ransomware | ✅ |
| `5677dfad...` | Ransomware | Ransomware | ✅ |
| `90b06f07...` | Ransomware | Ransomware | ✅ |
| `cd27a31e...` | Ransomware | Ransomware | ✅ |
| `e5834b7b...` | Ransomware | Ransomware | ✅ |
| `curl` | Goodware | Goodware | ✅ |
| `gpg` | Goodware | Ransomware | ❌ |
| `rsync` | Goodware | Goodware | ✅ |
| `sqlite` | Goodware | Ransomware | ❌ |
| `stressng` | Goodware | Ransomware | ❌ |

---

## Summary

| Metric | Result |
|--------|-------:|
| Window-Level Accuracy | **72.73%** |
| Execution-Level Accuracy | **72.73% (8/11)** |
| Real Ransomware Detected | **6 / 6 (100%)** |
| Goodware Correctly Classified | **2 / 5 (40%)** |
| False Positives | **3** |
| False Negatives | **0** |

> **Key Result:** The model successfully detected **all six previously unseen real ransomware samples** while maintaining an overall **72.73% execution-level accuracy**. The remaining errors were false positives on three benign applications (`gpg`, `sqlite`, and `stressng`).
