# Repository Overview

## Table of Contents
1. [Repository Overview](#1-repository-overview)
2. [Core Pipeline Components](#2-core-pipeline-components)
   - 2.1 [monitor.sh – the behavioral tracer](#21-monitorsh--full-system-behavioral-tracer)
   - 2.2 [parse_logs.py – feature extraction](#22-parse_logspy--from-raw-logs-to-fixed-width-feature-matrix)
   - 2.3 [goodware_pipeline.py / goodware_pipeline_set2.py – goodware data generation](#23-goodware-pipeline-scripts)
   - 2.4 [normalize.py – temporal normalisation to 100 rows](#24-normalizepy--fixed-length-normalisation)
3. [Analysis Tooling](#3-analysis-tooling)
   - 3.1 [analysis/eda_features.py – exploratory data analysis per sample](#31-analysiseda_featurespy--per-sample-exploratory-report)
   - 3.2 [analysis/collect.py – aggregation of EDA reports](#32-analysiscollectpy--aggregate-eda-reports)
4. [ELF Binary Analysis Artifacts](#4-elf-binary-analysis-artifacts)
5. [Modeling & Experiments](#5-modeling--experiments)
   - 5.1 [Classical ML (models/classical.py)](#51-classical-ml-pipeline-modelsclassicalpy)
   - 5.2 [Quantum-enhanced models (models/quantum/…)](#52-quantum-enhanced-models)
   - 5.3 [Utility scripts (models/qsvm*.py, models/svm.py, …)](#53-other-utility-scripts)
6. [Dataset Layout](#6-dataset-layout)
7. [Supporting Files](#7-supporting-files)
8. [How It All Fits Together](#8-how-it-all-fits-together--end-to-end-example)

---

## 1. Repository Overview

| Directory / File | Purpose |
|---|---|
| `analysis/` | Exploratory Data Analysis (EDA) scripts that walk each sample's `features.csv` and produce a detailed textual report (`eda_report.txt`). |
| `detection_pipeline/` | The heart of the pipeline:<br>• `monitor.sh` – launches the sample under strace/ltrace, QEMU (if needed), seeds dummy files, captures packets, perf, entropy, etc.<br>• `parse_logs.py` – turns the raw logs into a fixed-window feature CSV (`features.csv`).<br>• `goodware_pipeline*.py` – scripts that run a suite of benign workloads (openssl, ffmpeg, HTTP server, etc.) to generate a goodware baseline.<br>• `normalize.py` – downsamples or interpolates each sample's feature matrix to exactly 100 rows (fixed-length representation). |
| `elf_binaries_analysis/` | One sub-directory per analysed ELF binary (identified by its SHA-256). Each contains the raw artefacts produced by a single `monitor.sh` run: logs, dumps, PCAP, entropy timeline, proc timeline, meta-JSON, and the final `features.csv` + `eda_report.txt`. |
| `logs_goodware_v2` / `logs_goodware_v2_1` | Output directories from the goodware-pipeline runs (raw logs, PCAPs, etc.). |
| `models/` | Classical and quantum ML scripts, trained models (`best_quantum_model.pt`), feature-importance reports, and utility scripts for training / evaluation. |
| `train_set/`, `test_set/`, `train_set_org/` | Organised datasets (goodware vs ransomwarePoC) each containing per-sample folders with a `features.csv` ready for modelling. |
| `README.md` | Quick-start guide (setup, run a sample, parse logs, verify output). |
| `.gitignore` | Standard Python / build artefacts ignore rules. |

---

## 2. Core Pipeline Components

### 2.1 monitor.sh – Full-system Behavioral Tracer

**Location:** `detection_pipeline/monitor.sh`

#### High-level flow

| Step | Description |
|---|---|
| Setup (`sudo ./monitor.sh setup`) | Installs tracing utilities (strace, ltrace, tcpdump, perf, bpftrace, qemu-user, xvfb, etc.) and tweaks kernel perf settings. |
| Architecture detection | Uses `file` to detect if the binary is x86-64 on an ARM64 host → launches `qemu-x86_64` automatically. |
| Static analysis | Runs `strings`, `readelf -a`, `objdump -d`, `file`, and optional YARA rules → stored in `<outdir>/static_*`. |
| Dummy file seeding (optional) | Populates `$WATCHDIR` with a mixed set of document, image, archive, and media files (≈85 files by default) to give ransomware something to encrypt. |
| GUI simulation (optional) | Starts `Xvfb :99` and uses `xdotool` to dismiss simple pop-ups (e.g., Tkinter dialogs). |
| Entropy sampler (`_entropy_sampler.py`) | AWKS every 250 ms, computes Shannon entropy of the first 64 KB of each file in `$WATCHDIR`, and writes `entropy_timeline.csv`. |
| Network capture | `tcpdump -i any -w net.pcap -U` runs in background. |
| Process launch | Binary executed under either `strace -f -tt -T -s 256 -o strace.log` or `ltrace -f -tt -T -S -o ltrace.log` (fallback to strace under QEMU). |
| Perf counters | `perf stat -e cycles,cache-misses,page-faults,minor-faults,major-faults -p <pid> -o perf.log`. |
| /proc polling | Samples `VmRSS`, `read_bytes`, `write_bytes` from `/proc/<pid>/io` and `/proc/<pid>/status` every 100 ms → `proc_timeline.csv`. |
| Teardown | Kills all helper processes, writes `meta.json` (binary hash, architecture, QEMU flag, tracer used, timestamps, PID, arguments, timeout). |
| Post-run hint | Prints the command to run `parse_logs.py` on the generated directory. |

#### Key Design Points

- Works for native ARM64 and x86_64 via QEMU user-mode (transparent to the analyst).
- Collects syscall-level, library-call, network, CPU, memory, file-system, and entropy signals.
- All artefacts are written to a timestamped folder (`logs_<datetime>`) for easy traceability.
- The script is deliberately idempotent – re-running with the same outdir simply overwrites logs (useful for iterative tuning).

### 2.2 parse_logs.py – From Raw Logs to Fixed-Width Feature Matrix

**Location:** `detection_pipeline/parse_logs.py`

#### What it consumes (all located in `<logdir>/`)

| File | Content |
|---|---|
| `strace.log` or `ltrace.log` | Raw syscall / library-call trace output lines with timestamps (either epoch or HH:MM:SS.us). |
| `entropy_timeline.csv` | Per-sample entropy metrics (`epoch,mean_entropy,file_count,total_bytes`). |
| `proc_timeline.csv` | Per-sample memory & I/O (`epoch,vmrss_kb,read_bytes,write_bytes`). |
| `meta.json` | Binary metadata (SHA-256, arch, QEMU flag, etc.). |

#### Core Algorithm

1. Parse the trace (strace or ltrace) line-by-line using two regexes:
   - One extracts timestamp (wall-clock or epoch), PID (optional for `-f`), syscall name, return value.
   - Another extracts file-path strings, IP addresses, port numbers, mmap size, etc.
2. Bucket events into 100 ms windows (`WINDOW_SIZE = 0.1 s`). For each window we increment counters:
   - `bytes_read`, `bytes_written`, `urandom_bytes` (if read from `/dev/urandom`),
   - `mmap_bytes`, `mprotect_calls`, `futex_waits`,
   - file-system events (unlink, rename, open on target extensions),
   - network activity (sendto/recvfrom, DNS queries, connection failures),
   - process creation (fork, clone, execve).
3. Compute per-window rates (divide by window length).
4. Merge with auxiliary streams (entropy, proc) by aligning on epoch timestamps (nearest window).
5. Derive global features (time-to-first-encryption, encryption slope via linear regression on bytes written vs. time, entropy delta).
6. Emit a CSV where each row = one 100 ms window, columns = 31 temporal features (see header in any `features.csv`).

#### Output Example (features.csv header)

```
window_idx,window_start_s,bytes_read_per_sec,bytes_written_per_sec,rw_volume_ratio,
file_deletion_rate,file_rename_rate,target_extension_velocity,seq_vs_random_disk_ratio,
urandom_bytes_per_sec,mmap_brk_rate,mprotect_rate,futex_rate,net_packet_rate,
net_byte_volume,unique_dest_ip_rate,dns_query_rate,connection_failure_rate,
child_spawn_rate,execve_rate,syscall_sequence_entropy,io_burstiness_std,dir_ops,
drop_rate,ransom_note_hits,library_call_rate_openssl,vmrss_kb,mean_file_entropy,
time_to_first_encryption_s,encryption_slope,file_entropy_delta
```

### 2.3 Goodware Pipeline Scripts

- **`goodware_pipeline.py`** – runs a fixed set of benign, I/O-heavy workloads (OpenSSL speed test, FFmpeg transcoding, HTTP server + wget, recursive grep, tar archiving).
- **`goodware_pipeline_set2.py`** – a second suite that stresses different subsystems (GPG encryption/libgcrypt, rsync, SQLite transactions, curl download, stress-ng CPU/VM pressure).

Both scripts:

1. Create a fresh corpus directory under `/home/ubuntu/goodware_corpus(_v2)`.
2. Seed it with dummy files (via an embedded Python snippet).
3. Launch each workload via `sudo ./monitor.sh run …`, capturing logs into `logs_goodware_<name>` (or `logs_goodware_v2_<name>`).
4. Enforce a minimum total runtime (default 50 s) by re-running the workload if it finishes early.

The generated logs become the goodware training set (`train_set/goodware/*`).

### 2.4 normalize.py – Fixed-Length Normalisation

**Location:** `detection_pipeline/normalize.py`

Many malware samples run for a few seconds; others run for minutes. To feed a uniform-size tensor to ML models we resample each sample's feature matrix to exactly 100 rows:

- If the original has **> 100 rows** → uniform down-sampling (pick indices `linspace(0, N-1, 100)`).
- If **< 100 rows** → linear interpolation for numeric columns, forward/backward fill for categorical columns.
- The column `window_idx` is reset to `0…99` to preserve temporal ordering.

The script walks a source tree (`dataset/`) containing `<category>/<sample>/features.csv` and writes the normalized version to `dataset_normalized_100r/` preserving the same hierarchy.

---

## 3. Analysis Tooling

### 3.1 analysis/eda_features.py – Per-Sample Exploratory Report

**Location:** `analysis/eda_features.py`

**Purpose:** For each sub-folder that contains a `features.csv`, produce a human-readable `eda_report.txt` covering:

- **Overview** – row/column count, duplicate rows, null counts, numeric vs categorical column split.
- **Nullity Summary** – lists columns with missing values and percentages.
- **Sparsity Summary** – zero-count & percentage for each numeric column (helps spot sparsity-driven features).
- **Constant / Near-Constant Columns** – flags features with 1 or ≤5 unique values.
- **Per-Column Distributions** – for numeric: count, mean, std, skewness, kurtosis, percentiles (0,1,5,25,50,75,95,99,100); for categorical: top-10 values + counts & percentages.
- **Additional derived metrics** – e.g., entropy of syscall sequence, burstiness (std of I/O rates).

The script is invoked automatically by the analysis wrapper (`analysis/collect.py`) or can be run manually:

```bash
python3 analysis/eda_features.py /path/to/root_with_subfolders
```

### 3.2 analysis/collect.py – Aggregate EDA Reports

**Location:** `analysis/collect.py`

**What it does:**

- Walks the immediate sub-directories of a given root (default: current directory).
- In each sub-directory it looks for `eda_report.txt` (generated by `eda_features.py`).
- Concatenates all reports into a single `eda_comparison.txt` file, separated by clear banners and a header that lists the sample name (directory name).

**Use case:** after running the malware pipeline on dozens of samples, you can run `collect.py` to get one massive diff-style report that highlights divergences in feature distributions between ransomware and goodware.

---

## 4. ELF Binary Analysis Artifacts

Each sub-directory under `elf_binaries_analysis/` corresponds to a distinct binary (identified by its SHA-256 hash). Inside you will find the complete set of artefacts produced by a single `monitor.sh` run:

| File / Folder | Description |
|---|---|
| `static_strings.txt` | Output of `strings -a <binary>`. |
| `static_readelf.txt` | Full ELF header, program sections, dynamic symbols (`readelf -a`). |
| `static_objdump.txt` | Disassembly of all executable sections (`objdump -d`). |
| `file_info.txt` | Output of `file -b <binary>`. |
| `yara_matches.txt` (if `./yara_rules` exists) | YARA rule hits. |
| `net.pcap` | Raw packet capture (tcpdump). |
| `strace.log` or `ltrace.log` | Main syscall / library trace. |
| `xvfb.log` | Xvfb server output (if GUI simulation enabled). |
| `entropy_timeline.csv` | Time-series of file-entropy (see §2.2). |
| `proc_timeline.csv` | Process memory & I/O counters (see §2.2). |
| `perf.log` | Hardware counter output (cycles, cache-misses, page-faults). |
| `meta.json` | JSON metadata (see §2.1). |
| `features.csv` | The final 31-column, 100-row feature matrix ready for ML. |
| `eda_report.txt` | Exploratory data analysis report (see §3.1). |

These directories let you re-run the analysis (e.g., tweak feature extraction) or inspect raw behavioural evidence without re-executing the potentially risky binary.

---

## 5. Modeling & Experiments

### 5.1 Classical ML Pipeline (models/classical.py)

**Location:** `models/classical.py`

**Highlights**

- **Data Loading** – walks `../train_set` and `../test_set`, loads each `<category>/<sample>/features.csv`, normalises each sample to 100 rows via `normalize_rows()` (see §2.4).
- **Windowing** – slides a window of size `WINDOW_SIZE=10` rows with stride `STRIDE=5` over the normalised matrix, computing mean, std, min, max for each original feature inside the window → yields a high-dimensional feature vector per window.
- **Labeling** – each window inherits the binary label of its source sample (goodware / ransomware).
- **Feature Dropping** – a list of temporally-specific features (`encryption_slope`, `file_entropy_delta`, etc.) is removed because they are deemed leakage or are not stationary across windows (as described in the original script).
- **Model** – `Pipeline([SimpleImputer(strategy='median'), ExtraTreesClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1)])`.
- **Evaluation** – prints accuracy, classification report, confusion matrix, and per-sample RMS scores (average predicted ransomware probability > 0.4 → ransomware).
- **Feature Importance** – saved to `feature_importance.csv`.
- **Iterative Feature Selection** – reports accuracy when using the top N features (N = all, 15, 10).

### 5.2 Quantum-Enhanced Models

The `models/quantum/` directory contains a variety of experiments that embed the classical feature vectors into quantum kernels or variational circuits:

| Script | Purpose |
|---|---|
| `qsvm.py` | Implements a Quantum Support Vector Machine using Qiskit's QSVM (Aer simulator) with a feature map based on Pauli-expansion. |
| `qsvm_grid_search.py` | Grid-search over hyper-parameters (entanglement, reps, shots). |
| `qsvm_crossrun.py` | Repeats the QSVM training with different random seeds to assess variance. |
| `gauss_qsvm_optim.py` / `no_down_qsvm.py` | Variants using Gaussian kernels or without down-sampling. |
| `gan_qsvm_optim.py` | Experiments with a GAN-generated synthetic dataset to augment the training set before QSVM. |
| `quantum_sweep.py` | Sweeps over numbers of qubits and entangling layers, logging training time and accuracy. |
| `classical_top_10.py` / `quantum_top_10.py` | Compare the top-10 classical features against a quantum model using the same features. |
| `svm.py` | A baseline scikit-learn SVM (RBF kernel) for direct comparison. |
| `best_quantum_model.pt` | Serialized PyTorch-like object (the actual quantum model weights) obtained from the best run in the sweep. |

**Typical workflow** (as seen in the scripts):

1. Load the same normalised, window-aggregated dataset used by `classical.py`.
2. Optionally apply PCA to reduce dimensionality to ≤ 4–8 qubits (state-vector simulators blow up exponentially).
3. Encode each sample via a feature map (e.g., `PauliFeatureMap` or `ZZFeatureMap`).
4. Train a Quantum Kernel SVM (or a variational classifier) using the quantum kernel matrix evaluated on the Aer simulator.
5. Evaluate on the hold-out test set, report accuracy, and store the model.

### 5.3 Other Utility Scripts

- `distro.py` / `distribution_comparison.csv` – compares statistical distributions of each feature between goodware and ransomware (KS test, Jensen-Shannon divergence).
- `feature_importance.csv` – output of `classical.py`.
- `train_data_no_down.py` – variant that trains on the raw (non-windowed) 100-row samples (used for ablation).

---

## 6. Dataset Layout

```
train_set/
├─ goodware/
│   ├─ <sample1>/
│   │   └─ features.csv
│   └─ <sample2>/
│       └─ features.csv
└─ ransomwarePoC/
    ├─ <sampleA>/
    │   └─ features.csv
    └─ ...
test_set/  (same structure)
train_set_org/  (original, non-normalized splits – used for sanity checks)
```

Each `features.csv` is the output of `parse_logs.py` → `normalize.py` (i.e., exactly 100 rows, 31 columns). This uniformity enables the sliding-window approach used in the classical and quantum models.

---

## 7. Supporting Files

| File | Purpose |
|---|---|
| `README.md` | Quick-start: `sudo ./monitor.sh setup`, run a sample, parse logs, verify `features.csv`. |
| `.gitignore` | Excludes `__pycache__`, `.ipynb_checkpoints`, `*.pt` model snapshots, large data directories (`logs_*`, `dataset_normalized_*`), OS artefacts (`.DS_Store`). |
| `monitor.sh` (see §2.1) | The only script requiring root (for strace, tcpdump, perf). |
| `eda_comparison.txt` (generated by `analysis/collect.py`) | A consolidated EDA report across all samples. |
| `best_quantum_model.pt` | The best-performing quantum model found during hyperparameter sweeps. |
| `feature_importance.csv` | Ranking of the 31 temporal features (after window-aggregation) as derived from an ExtraTreesClassifier. |

---

## 8. How It All Fits Together – End-to-End Example

1. **Setup (once)**

   ```bash
   sudo ./detection_pipeline/monitor.sh setup
   ```

2. **Run a ransomware sample** (replace paths)

   ```bash
   sudo ./detection_pipeline/monitor.sh run \
       /path/to/ransomware.bin \
       ./logs_ransomware_01 \
       ./test_corpus \
       --timeout 60 --
   ```

   Outputs: `logs_ransomware_01/straces.log`, `entropy_timeline.csv`, `proc_timeline.csv`, `meta.json`, …

3. **Parse logs → features**

   ```bash
   python3 detection_pipeline/parse_logs.py \
       --logdir logs_ransomware_01 \
       --out logs_ransomware_01/features.csv
   ```

4. **Normalise to fixed length** (if you have many samples)

   ```bash
   python3 detection_pipeline/normalize.py   # reads from dataset/, writes dataset_normalized_100r/
   ```

5. **(Optional) Generate EDA report**

   ```bash
   python3 analysis/eda_features.py logs_ransomware_01/
   # → logs_ransomware_01/eda_report.txt
   ```

6. **Collect reports for a whole campaign**

   ```bash
   python3 analysis/collect.py /path/to/root_with_many_sample_dirs
   # → /path/to/root_with_many_sample_dirs/eda_comparison.txt
   ```

7. **Train / evaluate classical model**

   ```bash
   cd models
   python3 classical.py   # reads train_set/ and test_set/, prints accuracy, saves feature_importance.csv
   ```

8. **Experiment with quantum models**

   ```bash
   python3 qsvm.py   # or any of the qsvm_*.py scripts
   ```

9. **Inspect results** – look at `feature_importance.csv`, `classification_report.txt` (printed by the scripts), and the saved model files (`best_quantum_model.pt`).

---

## Closing Remarks

This repository packs a full lifecycle for behavioural malware analysis:

- Safe execution (QEMU + strace/ltrace + resource limits).
- Rich multi-modal tracing (syscalls, library calls, network, perf, entropy, process I/O).
- Deterministic feature extraction (fixed-width windows → stationary ML-ready vectors).
- Exploratory analytics to understand what each feature family tells you about ransomware vs goodware.
- Baseline classical ML (ExtraTrees) and experimental quantum-enhanced models for research-grade comparison.

All pieces are loosely coupled via well-defined file-system contracts (directories containing `strace.log`, `entropy_timeline.csv`, `proc_timeline.csv`, `meta.json`, and ultimately `features.csv`). This makes it straightforward to replace any component (e.g., swap perf for eBPF, or replace the QT classifier with a neural net) while keeping the rest of the pipeline intact.

Feel free to explore each directory, run the scripts on your own samples, and extend the feature set as needed! Happy hunting.
