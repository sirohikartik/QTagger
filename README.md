# 🛡️ QTagger: Ransomware Detection Pipeline for ARM64 Linux

QTagger is a sophisticated behavioral analysis and detection framework designed to identify ransomware on Linux ARM64 virtual machines. It leverages a complete pipeline from secure binary execution and multi-modal data collection to advanced machine learning—incorporating both classical and quantum-enhanced models.

## 📋 Project Overview

The core goal of this project is to differentiate between **ransomware** and **goodware** by analyzing their behavioral fingerprints during execution. The pipeline automates the process of running potentially malicious binaries in a controlled environment, extracting high-fidelity temporal features, and applying ML classifiers to determine the nature of the software.

---

## 🏗️ System Architecture

The project is divided into four main stages: **Instrumentation**, **Feature Extraction**, **Data Normalization**, and **Classification**.

### 1. Behavioral Instrumentation (`detection_pipeline/`)
The `monitor.sh` script acts as the orchestration engine for sample execution. It creates a "sandbox-like" environment to safely monitor binaries.

*   **Cross-Architecture Support**: Natively supports ARM64 binaries and uses `QEMU user-mode emulation` to execute x86_64 binaries on ARM64 hosts.
*   **Multi-Modal Monitoring**:
    *   **System Calls**: Captures detailed syscall traces using `strace` or `ltrace`.
    *   **Network Traffic**: Records all network activity via `tcpdump` (PCAP).
    *   **Hardware Performance**: Tracks CPU cycles, cache misses, and page faults using `perf stat`.
    *   **Resource Consumption**: Polls `/proc/[pid]/status` and `/proc/[pid]/io` for memory (VmRSS) and I/O throughput.
    *   **File Entropy**: A custom Python sampler calculates the **Shannon Entropy** of files in a monitored directory every 250ms to detect the onset of encryption.
    *   **GUI Simulation**: Uses `Xvfb` and `xdotool` to simulate a display and interact with binaries that require a GUI or prompt user input.
*   **Environment Seeding**: Automatically seeds the environment with a variety of dummy files (`.pdf`, `.docx`, `.jpg`, etc.) to provide targets for ransomware encryption.

### 2. Feature Extraction (`detection_pipeline/parse_logs.py`)
The raw logs are processed into a structured temporal dataset. The pipeline extracts **31 behavioral features** windowed every 100ms:

| Category | Features |
| :--- | :--- |
| **I/O Activity** | Bytes read/written per sec, RW volume ratio, file deletion/rename rates, target extension velocity. |
| **Crypto Indicators** | `/dev/urandom` read rates, mean file entropy, entropy delta, encryption slope. |
| **Memory & CPU** | `mmap` bytes, `mprotect` calls, `futex` wait rates, VmRSS. |
| **Network** | Packet rate, byte volume, unique destination IPs, DNS query rate, connection failures. |
| **Process Behavior** | Child process spawn rate, `execve` calls, syscall sequence entropy, I/O burstiness. |
| **Targeting** | Time to first encryption, ransom note heuristics. |

### 3. Data Normalization (`detection_pipeline/normalize.py`)
To ensure consistency across samples of varying execution lengths, the pipeline normalizes all feature sets to a fixed length of **100 rows**.
*   **Downsampling**: Uniformly samples rows if the execution is longer than 100 windows.
*   **Interpolation**: Linear interpolation for numeric columns if the execution is shorter than 100 windows.

### 4. Detection Models (`models/`)
The project implements two distinct approaches to classification:

#### 🌲 Classical ML (`models/classical/`)
Uses an **Extra Trees Classifier** (Random Forest variant). 
*   **Window-to-Sample Aggregation**: Instead of raw sequences, it computes summary statistics (mean, std, max, min) over sliding windows.
*   **Feature Selection**: Employs an initial model to identify the **Top 10-30 most discriminative features** to reduce overfitting and improve inference speed.

#### ⚛️ Quantum ML (`models/quantum/`)
Implements a **Learnable Quantum Support Vector Machine (QSVM)** and **Quantum Classifiers** using `Pennylane` and `PyTorch`.
*   **Quantum Feature Map**: Uses `AngleEmbedding` and `StronglyEntanglingLayers` to map classical features into a high-dimensional Hilbert space.
*   **KTA Optimization**: The quantum kernel is optimized using **Kernel Target Alignment (KTA)** loss to specifically sharpen the boundary between ransomware and goodware.
*   **Adaptive Classification**: Employs both SVCs with precomputed quantum kernels and direct neural quantum classifiers for multiclass tasks.

---

## 🎯 Detection Strategies

The project implements two distinct evaluation paradigms to test the robustness and specificity of the models:

### 1. Binary Classification (Out-of-Domain Detection)
The goal is to evaluate how well models generalize from synthetic "Proof of Concept" (PoC) behavior to real-world ransomware.
*   **Training Set**: `train_set/` (Contains PoC ransomware and goodware).
*   **Test Set**: The **first feature set (`features.csv`)** of samples in `test_set/` (Contains actual ransomware binaries and real goodware).
*   **Objective**: Detect whether a binary is "Ransomware" or "Goodware," treating the transition from PoC to actual malware as an out-of-domain challenge.

### 2. Multiclass Classification (Cross-Run Evaluation)
The goal is to identify specific ransomware families or individual samples by training on different execution runs.
*   **Data Source**: Uses the three feature sets (`features.csv`, `features2.csv`, `features3.csv`) provided for each sample in `test_set/`.
*   **Methodology**: A **3-Fold Cross-Run** strategy is used to ensure that the model can recognize a sample regardless of which specific execution run it is analyzing:
    *   **Fold 1**: Train on runs 2 & 3 $\rightarrow$ Test on run 1 (`features.csv`).
    *   **Fold 2**: Train on runs 1 & 3 $\rightarrow$ Test on run 2 (`features2.csv`).
    *   **Fold 3**: Train on runs 1 & 2 $\rightarrow$ Test on run 3 (`features3.csv`).
*   **Objective**: Distinguish between multiple ransomware samples/families, using the sample names as target labels.

---

## 📂 Repository Structure

```text
.
├── analysis/               # Data analysis scripts and collection utilities
├── detection_pipeline/     # Core instrumentation and feature extraction
│   ├── monitor.sh          # The main execution and monitoring script
│   ├── parse_logs.py       # Raw log -> ML features transformer
│   └── normalize.py       # Temporal data length normalization
├── models/                 # ML Implementations
│   ├── classical/           # Random Forest models and reports
│   └── quantum/            # QSVM models and reports
├── train_set/              # Training data (Goodware & Ransomware)
└── test_set/               # Testing data (Goodware & Ransomware)
```

---

## 🚀 Usage Guide

### 1. Setup the Environment
Install the required toolchain (QEMU, Xvfb, strace, etc.):
```bash
sudo ./detection_pipeline/monitor.sh setup
```

### 2. Execute and Monitor a Binary
Run a sample binary and collect behavioral logs:
```bash
sudo ./detection_pipeline/monitor.sh run "/path/to/binary" ./logs_folder ./test_corpus
```

### 3. Extract Features
Convert the resulting logs into a CSV feature set:
```bash
python3 ./detection_pipeline/parse_logs.py --logdir ./logs_folder --out ./logs_folder/features.csv
```

### 4. Run Detection Models
Navigate to the `models` directory and run the desired classifier:
```bash
# Classical RF
python3 models/classical/rf_binary.py

# Quantum SVM
python3 models/quantum/qsvm_binary.py
```

---

## 🛠️ Technical Stack
- **OS**: Linux (ARM64)
- **Virtualization**: QEMU User-mode
- **Instrumentation**: `strace`, `tcpdump`, `perf`, `Xvfb`
- **Languages**: Python 3, Bash
- **ML Libraries**: `scikit-learn`, `pandas`, `numpy`
- **Quantum Framework**: `Pennylane`, `PyTorch`
