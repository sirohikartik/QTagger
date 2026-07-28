# 🛡️ Ransomware Detection: Generalizing from PoC to Real Samples

## Objective

This experiment evaluates whether a machine learning model trained **only on proof-of-concept (PoC) ransomware** can successfully detect **previously unseen real-world ransomware** based on runtime behavioral features.

---

# 📂 Dataset

## Training Set

### Goodware (5)

- ffmpeg
- grep
- http
- openssl
- tar

### PoC Ransomware (5)

- c_cry
- gocry
- jimmy
- lollocker
- nooled

---

## Test Set (Completely Unseen)

### Goodware (5)

- curl
- gpg
- rsync
- sqlite
- stressng

### Real Ransomware (6)

Six real ransomware samples identified by SHA-256 hashes.

> **None of the test samples were used during training.**

---

# ⚙️ Feature Processing

Every execution trace was normalized to **100 windows**.

Sliding window parameters:

```text
Window Size : 10
Stride      : 5
```

Instead of using raw window values, four statistical summaries were computed for every runtime feature:

- Mean
- Standard Deviation
- Maximum
- Minimum

This produces a compact behavioral representation while preserving temporal information.

---

# 🤖 Model

Classifier:

```text
ExtraTreesClassifier
```

Workflow:

1. Train on all summary features.
2. Rank features using feature importance.
3. Select the **Top 10** features.
4. Retrain the classifier using only those features.
5. Evaluate on the unseen test set.

---

# ⭐ Top 10 Features

| Feature | Description |
|---------|-------------|
| rw_volume_ratio_min | Minimum read/write ratio |
| window_start_s_std | Timing variability |
| syscall_sequence_entropy_min | Lowest syscall entropy |
| file_deletion_rate_max | Maximum deletion rate |
| file_deletion_rate_mean | Average deletion rate |
| bytes_read_per_sec_min | Minimum disk read throughput |
| seq_vs_random_disk_ratio_max | Peak sequential disk access |
| seq_vs_random_disk_ratio_mean | Average sequential disk access |
| execve_rate_max | Maximum process creation rate |
| syscall_sequence_entropy_mean | Average syscall entropy |

Most selected features describe **filesystem**, **disk I/O**, and **system call behavior**, indicating that these behaviors are effective for distinguishing ransomware from benign software.

---

# 📊 Results

## Window-Level

| Metric | Value |
|--------|------:|
| Accuracy | **72.73%** |

---

## Execution-Level

| Metric | Result |
|--------|-------:|
| Real ransomware detected | **6 / 6 (100%)** |
| Goodware correctly identified | **2 / 5** |
| Overall execution accuracy | **8 / 11 (72.73%)** |

---

# 💡 Key Observation

Although the model was trained **only on PoC ransomware**, it successfully detected **all unseen real ransomware samples**.

The remaining challenge is reducing **false positives**, as several benign applications were classified as ransomware.

---

# 🚀 Future Work

- Reduce false positives through threshold tuning.
- Increase training diversity with additional ransomware families.
- Evaluate against larger goodware datasets.
- Compare ExtraTrees with Gradient Boosting and XGBoost.
- Investigate temporal models (e.g., LSTM or Transformer) for behavioral sequences.

---

*This experiment demonstrates that behavioral runtime features can generalize beyond PoC ransomware and identify previously unseen ransomware families.*
