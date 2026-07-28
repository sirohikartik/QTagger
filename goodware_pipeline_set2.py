#!/usr/bin/env python3

import os
import subprocess
import sys
import time

# Configuration
PIPELINE_DIR = "/home/ubuntu/ransomware_pipeline"
MONITOR_SCRIPT = os.path.join(PIPELINE_DIR, "monitor.sh")
CORPUS_BASE = "/home/ubuntu/goodware_corpus_v2"
MIN_RUNTIME = 50  # seconds

def run_command(cmd):
    print(f"[*] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def seed_corpus(folder_name, num_files=10000):
    """Creates a folder with dummy files."""
    path = os.path.join(CORPUS_BASE, folder_name)
    os.makedirs(path, exist_ok=True)
    print(f"[*] Seeding {num_files} files into {path}...")
    for i in range(num_files):
        with open(os.path.join(path, f"data_{i}.txt"), "w") as f:
            f.write(f"Sensitive data block {i}\n" * 10)

def run_goodware(name, binary, args, outdir_suffix, timeout=300):
    """Runs a goodware binary through the monitor, ensuring at least MIN_RUNTIME seconds total."""
    corpus_path = os.path.join(CORPUS_BASE, name)
    outdir = os.path.join(PIPELINE_DIR, f"logs_goodware_v2_{outdir_suffix}")

    print(f"\n[🚀] Starting Goodware Capture V2: {name}")

    cmd = [
        "sudo", "bash", MONITOR_SCRIPT, "run",
        binary,
        outdir,
        corpus_path,
        "--num-files", "0",
        "--timeout", str(timeout),
        "--no-gui",
        "--"
    ] + args

    overall_start = time.time()
    run_count = 0

    while True:
        run_count += 1
        elapsed = time.time() - overall_start
        remaining = MIN_RUNTIME - elapsed

        if run_count > 1:
            print(f"[⏱] Only {elapsed:.1f}s elapsed — re-running to meet {MIN_RUNTIME}s minimum (run #{run_count})...")

        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[!] {name} run #{run_count} failed: {e}")

        elapsed = time.time() - overall_start
        if elapsed >= MIN_RUNTIME:
            break

        # Small pause between re-runs to avoid hammering
        time.sleep(1)

    total = time.time() - overall_start
    print(f"[✅] {name} completed after {run_count} run(s), {total:.1f}s total. Logs in {outdir}")


def ensure_gpg_key(uid="test@test.com", passphrase="test123"):
    """Check if the GPG key exists locally; generate it only if missing."""
    result = subprocess.run(
        ["gpg", "--list-keys", uid],
        capture_output=True
    )
    if result.returncode == 0:
        print(f"[*] GPG key for {uid} already exists, skipping generation.")
        return

    print("[*] Generating dummy GPG key...")
    try:
        run_command(["gpg", "--batch", "--passphrase", passphrase, "--quick-generate-key", uid])
    except subprocess.CalledProcessError:
        print("[!] GPG key generation failed. Continuing, but the encrypt step may fail...")


if __name__ == "__main__":
    os.makedirs(CORPUS_BASE, exist_ok=True)

    # 1. GPG (Encryption/Decryption - Different Libs than OpenSSL)
    # Uses libgcrypt instead of openssl libs. High file I/O + Crypto.
    seed_corpus("gpg_test", num_files=100)

    ensure_gpg_key(uid="test@test.com", passphrase="test123")

    # Encrypt one of the seeded files to generate activity
    target_file = os.path.join(CORPUS_BASE, "gpg_test", "data_0.txt")
    run_goodware(
        name="gpg",
        binary="/usr/bin/gpg",
        args=[
            "--batch", "--yes", "--passphrase", "test123",
            "--auto-key-locate", "local",   # don't try WKD/network lookup
            "--trust-model", "always",      # skip trust prompts on a fresh key
            "-r", "test@test.com", "-e", target_file
        ],
        outdir_suffix="gpg",
        timeout=300
    )

    # 2. RSYNC (Network/File Sync - Different from Tar/Cp)
    # Uses its own delta-transfer algorithm. High network + file stat activity.
    seed_corpus("rsync_src", num_files=5000)
    dest_dir = "/tmp/rsync_dest"
    os.makedirs(dest_dir, exist_ok=True)

    run_goodware(
        name="rsync",
        binary="/usr/bin/rsync",
        args=["-avz", "/home/ubuntu/goodware_corpus_v2/rsync_src/", "/tmp/rsync_dest/"],
        outdir_suffix="rsync",
        timeout=300
    )

    # 3. SQLITE3 (Database I/O - Random Access Patterns)
    # Unlike sequential tar/grep, this does random reads/writes and fsyncs.
    seed_corpus("sqlite_test", num_files=1)
    db_file = "/home/ubuntu/goodware_corpus_v2/sqlite_test/test.db"

    # Wrapper to create DB and insert data for 60 seconds
    wrapper_script = "/home/ubuntu/sqlite_wrapper.sh"
    with open(wrapper_script, "w") as f:
        f.write(f"""#!/bin/bash
rm -f {db_file}
sqlite3 {db_file} "CREATE TABLE data (id INTEGER PRIMARY KEY, content TEXT);"
END_TIME=$((SECONDS + 60))
while [ $SECONDS -lt $END_TIME ]; do
    sqlite3 {db_file} "INSERT INTO data (content) VALUES ('random_data_block_$RANDOM');"
done
""")
    os.chmod(wrapper_script, 0o755)

    run_goodware(
        name="sqlite",
        binary="/bin/bash",
        args=[wrapper_script],
        outdir_suffix="sqlite",
        timeout=300
    )

    # 4. CURL (Network Download - Different from HTTP Server)
    # Active client-side network behavior. DNS + TCP + Write to disk.
    seed_corpus("curl_test", num_files=1)

    run_goodware(
        name="curl",
        binary="/usr/bin/curl",
        args=["-O", "https://releases.ubuntu.com/24.04/ubuntu-24.04.1-live-server-amd64.iso"],
        outdir_suffix="curl",
        timeout=300
    )

    # 5. STRESS-NG (CPU/Memory Pressure - Different from Ffmpeg)
    # Pure computational stress without media encoding logic.
    seed_corpus("stress_test", num_files=1)

    run_goodware(
        name="stressng",
        binary="/usr/bin/stress-ng",
        args=["--cpu", "2", "--vm", "1", "--vm-bytes", "256M", "--timeout", "60s"],
        outdir_suffix="stressng",
        timeout=300
    )

    print("\n[🏁] Heavy Goodware Suite V2 Complete!")
