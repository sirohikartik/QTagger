# 📂 Comprehensive Dataset & Methodology Documentation

This document provides a thorough breakdown of the datasets utilized in the pipeline, including file schemas, provenance, categorization, and the isolated execution environments used for behavioral analysis.

---

## 🦠 1. ELF Binaries Ransomware (Wild-Type Samples)

### 📋 Schema & Structure
- **File Identifier:** Individual sample directories are named using their unique **SHA-256 hash codes**, while the primary master collection is categorized under `ransomware` inside the `test_set` directory.
- **File Type:** Raw executables are stored in **`.elf` format**, while structured telemetry over time is logged in **`.csv` feature tables**.
- **Class / Label:** Real-world, compiled ELF x86_64 ransomware families, including:
  - **LockBit 5.0**
  - **Gunra**
  - **Qilin**
  - **01flip**
  - **Generic ELF Ransomware**
- **Per-Class Counts:** One representative sample is used per family, except for **LockBit 5.0**, which utilizes two variant samples to capture behavioral diversity.

### 🌐 Provenance & Collection
- **Source / Origin:** Retrieved directly from **MalwareBazaar** to guarantee authenticity and wild-type behavior.

### 🛡️ Safety, Handling & Analysis Environment
- **Virtualization & Emulation Engine:** VMware running an ARM64 guest environment, leveraging **QEMU user-mode/system emulation** to execute **x86_64** binary payloads on ARM64 host hardware to bridge the architecture gap caused by ARM64 malware scarcity.
- **Guest Operating System:** Ubuntu 24.04 LTS (ARM64).
- **Analysis Toolchain:** Automated monitoring via custom Bash shell scripts paired with a Python telemetry parser tracking system resource consumption, file modifications, and encryption velocity ⚡.

---

## 🧪 2. Proof-of-Concept (PoC) Ransomware

### 📋 Schema & Structure
- **File Identifier:** Individual sample folders are labeled using their respective descriptive ransomware project/repository names (**`c_cry`**, **`gocry`**, **`jimmy`**, **`lollocker`**, **`nooled`**), while the consolidated master collection is labeled as `ransomwarePoC` inside the `train_set` directory.
- **File Type:** Executable ELF binaries (`.elf`) paired with CSV dataset files.
- **Class / Label:** Educational, research, and open-source implementation variants of Linux ransomware.
- **Per-Class Counts:** One sample/repository is utilized per family.

### 🌐 Provenance & Collection
- **Source / Origin:** Sourced from open-source repositories on **GitHub**.

### 🛡️ Safety, Handling & Analysis Environment
- **Virtualization Engine:** Canonical **Multipass** (utilizing **QEMU/KVM** as the underlying hypervisor).
- **Guest Operating System:** Ubuntu 24.04 LTS.
- **Analysis Toolchain:** Shell automation scripts and Python-based log parsers.

---

## ✅ 3. Goodware (Legitimate Baseline Binaries)

### 📋 Schema & Structure
- **File Identifier:** Individual utility names (**`curl`**, **`gpg`**, **`rsync`**, **`sqlite`**, **`stressng`**), while the master collection is labeled as `goodware` across both `train_set` and `test_set` directories.
- **File Type:** Compiled `.elf` binaries and `.csv` parsed feature logs.
- **Class / Label:** Standard Linux utility programs and command-line tools functioning as non-malicious baseline behavior.
- **Per-Class Counts:** One sample per standard tool package.

### 🌐 Provenance & Collection
- **Source / Origin:** Standard package repositories and open-source codebases via **GitHub**.

### 🛡️ Safety, Handling & Analysis Environment
- **Virtualization Engine:** Canonical **Multipass** (utilizing **QEMU** as the underlying hypervisor).
- **Guest Operating System:** Ubuntu 24.04 LTS.
- **Analysis Toolchain:** Shell code and Python parser monitoring resource utilization metrics (e.g., VMRSS, file read/write speeds, network calls) to establish a distinct behavioral contrast against ransomware execution patterns.
