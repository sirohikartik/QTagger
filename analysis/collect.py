#!/usr/bin/env python3
"""
collect_texts.py
----------------
For each immediate subfolder (super-parent) of the root directory,
walks all nested subdirectories, finds every .txt file, and concatenates
them into a single file named <super_parent_name>.txt in the root.

Structure expected:
  root/
    c_cry_mega_archive/        ← super-parent → c_cry_mega_archive.txt
      logs_cry_mega/
        static_strings.txt
        file_info.txt
        ...
    gocry_mega_archive/        ← super-parent → gocry_mega_archive.txt
      ...

Usage:
  python3 collect_texts.py                  # uses cwd as root
  python3 collect_texts.py /path/to/root    # uses given path as root
"""

import os
import sys


TARGET_FILENAME = "eda_report.txt"

DIVIDER = "=" * 72


def collect_for_superparent(superparent_path, superparent_name, root):
    """Walk superparent_path, gather all eda_report.txt files, return combined string."""
    collected = []

    for dirpath, dirnames, filenames in os.walk(superparent_path):
        dirnames.sort()
        for fname in sorted(filenames):
            if fname != TARGET_FILENAME:
                continue

            fpath = os.path.join(dirpath, fname)
            rel   = os.path.relpath(fpath, root)

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                content = f"[ERROR reading file: {e}]"

            header = (
                f"{DIVIDER}\n"
                f"FILE : {rel}\n"
                f"{DIVIDER}\n"
            )
            collected.append(header + content)

    return collected


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        sys.exit(f"[ERROR] Not a directory: {root}")

    print(f"[*] Root: {root}")

    superparents = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    if not superparents:
        sys.exit("[!] No subfolders found in root.")

    out_path = os.path.join(root, "eda_comparison.txt")
    all_blocks = []

    for name in superparents:
        sp_path = os.path.join(root, name)
        print(f"  [COLLECTING] {name}/ ...")
        blocks = collect_for_superparent(sp_path, name, root)

        if not blocks:
            print(f"  [SKIP] No eda_report.txt found under {name}/")
            continue

        # One report per super-parent — wrap it with a clear super-parent header
        header = (
            f"\n{'█' * 72}\n"
            f"  SAMPLE: {name}\n"
            f"{'█' * 72}\n"
        )
        all_blocks.append(header + "\n" + "\n\n".join(blocks))
        print(f"  [OK] {name}")

    if not all_blocks:
        sys.exit("[!] No eda_report.txt files found anywhere.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("EDA COMPARISON REPORT\n")
        f.write(f"Root    : {root}\n")
        f.write(f"Samples : {len(all_blocks)}\n")
        f.write(DIVIDER + "\n")
        f.write("\n".join(all_blocks))

    print(f"\n[✅] Combined report → {out_path}  ({len(all_blocks)} samples)")


if __name__ == "__main__":
    main()
