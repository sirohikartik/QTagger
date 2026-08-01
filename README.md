# Ransomware Dataset Curation Pipeline for arm64 linux VMs
This repository provides an automated pipeline for safely executing, monitoring, and extracting behavioral features from ransomware samples using QEMU virtualization and Xvfb.

# 🚀 Quick Start Guide

Follow these steps to set up the environment, execute a sample, and extract features for machine learning analysis.

### 1. One-time setup (installs everything including QEMU, Xvfb, etc.)
Install all required dependencies, including QEMU, Xvfb, and monitoring utilities:
```bash
sudo ./monitor.sh setup
```

### 2. Run the Binary
The framework auto-detects the architecture and executes the sample securely inside QEMU. Choose the command that fits your use case:

* **Standard Execution (ARM64 / x86_64 sample):**
  ```bash
  sudo ./monitor.sh run "ransomware_sample_pathway" ./logs_folder ./test_corpus

* **If running with custom arguments and a 60s timeout:**
  ```bash
  sudo ./monitor.sh run "ransomware_sample_pathway" ./logs_folder ./corpus --timeout 60 -- -p /tmp/targets -e
```

### 3. Parse the logs into ML Features
Process the generated execution logs into a structured CSV format ready for machine learning models:
  ```bash
  python3 parse_logs.py --logdir ./logs_folder --out ./logs_folder/features.csv
```

### 4. Verify Output
Inspect the generated dataset to ensure features were successfully extracted:
  ```bash
  head -5 ./logs_folder/features.csv
  wc -l ./logs_folder/features.csv
```

---
##### *Note: For some binaries they may require different .so files for execution so if any error occurs please carefully download the required .so files and other dependencies as the monitor.sh file only downloads the bare minimum dependencies and measurement tools.*
---

This pipeline was created using coding assistance of Qwen 3.7 Plus
