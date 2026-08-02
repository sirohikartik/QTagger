# ELF Binaries Ransomware Dataset

## 📁 Schema & Structure

### File Identifier
- **Individual Samples:** Labeled using their respective **SHA-256 hash codes**.
- **Master Dataset:** Consolidated under a primary root directory (`...`).

### File Formats
- **Raw Binaries:** Executable files in **ELF (Executable and Linkable Format)**, specifically tailored for x86_64 architectures.
- **Derived Datasets:** Tabular summaries exported as **`.csv` files** containing engineered statistical features extracted via windowing and feature aggregation.

---

## 🦠 Classes & Ransomware Families
The dataset comprises targeted ELF x86_64 ransomware strains sourced from real-world threats, encompassing the following families:
- **LockBit 5.0** (Multiple variants)
- **Gunra**
- **Qilin**
- **01flip**
- **Generic ELF Ransomware**

---

## 🔍 Provenance & Collection

- **Source / Origin:** All malicious binaries were retrieved directly from **MalwareBazaar** to ensure verified, wild-type threat representation.
- **Per-Class Distribution:** 
  - 1 sample per unique family (Gunra, Qilin, 01flip, Generic).
  - 2 samples representing distinct variants for **LockBit 5.0**.

---

## 🛡️ Safety, Handling & Analysis Environment

To ensure complete containment and operational security, all malware samples were executed and dynamically analyzed under strict isolation protocols:

- **Host Platform:** VMware Workstation Pro / ESXi virtualization.
- **Operating System:** Ubuntu 24.04 LTS (x86_64) clean environment.
- **Toolchain:** Custom-built shell scripts paired with Python-based parsers to monitor behavioral telemetry, system calls, IO bursts, and file system modifications.
