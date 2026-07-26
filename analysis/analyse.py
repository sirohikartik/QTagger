#!/usr/bin/env python3
"""
eda_features.py
---------------
Walks every immediate subfolder of a root directory (default: current working dir),
finds features.csv in each one, and writes a detailed EDA report to
  <subfolder>/eda_report.txt

Covers:
  - Shape & basic info
  - Nullity  (missing values per column)
  - Sparsity (zero values per column, for numeric cols)
  - Column-level distributions (numeric: percentiles, mean, std, skew, kurt)
                               (categorical: value counts top-10, cardinality)
  - Constant / near-constant columns
  - Duplicate rows

Usage:
  python3 eda_features.py                  # scans subfolders of cwd
  python3 eda_features.py /path/to/root    # scans subfolders of given path
"""

import os
import sys
import math
from collections import Counter

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("[ERROR] pandas and numpy are required: pip install pandas numpy")


# ── helpers ──────────────────────────────────────────────────────────────────

def hr(char="─", width=72):
    return char * width

def section(title):
    return f"\n{hr()}\n  {title}\n{hr()}\n"

def fmt_pct(n, total):
    if total == 0:
        return "0.00%"
    return f"{100 * n / total:.2f}%"

def percentiles(series, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    vals = series.dropna()
    if vals.empty:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(vals, q)) for q in qs}

def skewness(series):
    vals = series.dropna()
    if len(vals) < 3:
        return float("nan")
    m = vals.mean()
    s = vals.std()
    if s == 0:
        return float("nan")
    return float(((vals - m) ** 3).mean() / s ** 3)

def kurtosis(series):
    vals = series.dropna()
    if len(vals) < 4:
        return float("nan")
    m = vals.mean()
    s = vals.std()
    if s == 0:
        return float("nan")
    return float(((vals - m) ** 4).mean() / s ** 4 - 3)


# ── per-column analysis ───────────────────────────────────────────────────────

def analyze_numeric(col, series, n_rows):
    lines = []
    non_null = series.dropna()
    null_count = series.isna().sum()
    zero_count = int((non_null == 0).sum())
    neg_count  = int((non_null < 0).sum())

    lines.append(f"  dtype        : {series.dtype}")
    lines.append(f"  non-null     : {len(non_null):,}  ({fmt_pct(len(non_null), n_rows)})")
    lines.append(f"  null         : {null_count:,}  ({fmt_pct(null_count, n_rows)})")
    lines.append(f"  zeros        : {zero_count:,}  ({fmt_pct(zero_count, n_rows)})  ← sparsity")
    lines.append(f"  negatives    : {neg_count:,}  ({fmt_pct(neg_count, n_rows)})")

    if non_null.empty:
        lines.append("  [all values null — no distribution stats]")
        return lines

    lines.append(f"  mean         : {non_null.mean():.6g}")
    lines.append(f"  std          : {non_null.std():.6g}")
    lines.append(f"  skewness     : {skewness(series):.4f}  (0=symmetric, >1 right-skewed)")
    lines.append(f"  kurtosis     : {kurtosis(series):.4f}  (0=normal tails)")

    pcts = percentiles(series)
    lines.append(f"  min  (p0)    : {pcts[0]:.6g}")
    lines.append(f"  p1           : {pcts[1]:.6g}")
    lines.append(f"  p5           : {pcts[5]:.6g}")
    lines.append(f"  p25  (Q1)    : {pcts[25]:.6g}")
    lines.append(f"  p50  (med)   : {pcts[50]:.6g}")
    lines.append(f"  p75  (Q3)    : {pcts[75]:.6g}")
    lines.append(f"  p95          : {pcts[95]:.6g}")
    lines.append(f"  p99          : {pcts[99]:.6g}")
    lines.append(f"  max  (p100)  : {pcts[100]:.6g}")

    unique = non_null.nunique()
    lines.append(f"  unique vals  : {unique:,}")
    if unique == 1:
        lines.append(f"  ⚠ CONSTANT column (value = {non_null.iloc[0]})")

    return lines


def analyze_categorical(col, series, n_rows, top_n=10):
    lines = []
    null_count = series.isna().sum()
    non_null = series.dropna()
    cardinality = non_null.nunique()

    lines.append(f"  dtype        : {series.dtype}")
    lines.append(f"  non-null     : {len(non_null):,}  ({fmt_pct(len(non_null), n_rows)})")
    lines.append(f"  null         : {null_count:,}  ({fmt_pct(null_count, n_rows)})")
    lines.append(f"  cardinality  : {cardinality:,} unique values")

    if cardinality == 0:
        return lines
    if cardinality == 1:
        lines.append(f"  ⚠ CONSTANT column (value = '{non_null.iloc[0]}')")

    counts = non_null.value_counts(dropna=True)
    lines.append(f"  top {top_n} values (count | pct):")
    for val, cnt in counts.head(top_n).items():
        lines.append(f"    {str(val)[:50]:<52} {cnt:>8,}  {fmt_pct(cnt, n_rows):>7}")

    return lines


# ── main report ───────────────────────────────────────────────────────────────

def build_report(csv_path):
    lines = []
    lines.append(f"EDA Report")
    lines.append(f"File    : {csv_path}")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        lines.append(f"\n[ERROR] Could not read CSV: {e}")
        return "\n".join(lines)

    n_rows, n_cols = df.shape
    dup_rows = int(df.duplicated().sum())

    # ── overview ──
    lines.append(section("OVERVIEW"))
    lines.append(f"  Rows              : {n_rows:,}")
    lines.append(f"  Columns           : {n_cols:,}")
    lines.append(f"  Duplicate rows    : {dup_rows:,}  ({fmt_pct(dup_rows, n_rows)})")
    lines.append(f"  Total cells       : {n_rows * n_cols:,}")

    total_null  = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    lines.append(f"  Total nulls       : {total_null:,}  ({fmt_pct(total_null, total_cells)} of all cells)")

    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols  = df.select_dtypes(exclude=[np.number]).columns.tolist()
    lines.append(f"  Numeric columns   : {len(num_cols)}")
    lines.append(f"  Categorical cols  : {len(cat_cols)}")

    # ── nullity summary ──
    lines.append(section("NULLITY SUMMARY  (columns with any nulls)"))
    null_series = df.isna().sum()
    null_cols = null_series[null_series > 0].sort_values(ascending=False)
    if null_cols.empty:
        lines.append("  No null values found across all columns.")
    else:
        lines.append(f"  {'Column':<45} {'Null Count':>10}  {'Null %':>7}")
        lines.append(f"  {'─'*45} {'─'*10}  {'─'*7}")
        for col, cnt in null_cols.items():
            lines.append(f"  {str(col)[:45]:<45} {cnt:>10,}  {fmt_pct(cnt, n_rows):>7}")

    # ── sparsity summary ──
    lines.append(section("SPARSITY SUMMARY  (zero counts in numeric columns)"))
    if not num_cols:
        lines.append("  No numeric columns.")
    else:
        zero_data = []
        for col in num_cols:
            zc = int((df[col] == 0).sum())
            zero_data.append((col, zc, fmt_pct(zc, n_rows)))
        zero_data.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"  {'Column':<45} {'Zero Count':>10}  {'Zero %':>7}")
        lines.append(f"  {'─'*45} {'─'*10}  {'─'*7}")
        for col, zc, zpct in zero_data:
            flag = "  ← highly sparse" if zc / n_rows > 0.9 else ""
            lines.append(f"  {str(col)[:45]:<45} {zc:>10,}  {zpct:>7}{flag}")

    # ── constant / near-constant columns ──
    lines.append(section("CONSTANT / NEAR-CONSTANT COLUMNS"))
    flagged = []
    for col in df.columns:
        vc = df[col].dropna().nunique()
        if vc <= 1:
            flagged.append((col, vc, "CONSTANT"))
        elif vc <= 5:
            flagged.append((col, vc, f"near-constant ({vc} unique)"))
    if not flagged:
        lines.append("  None found.")
    else:
        for col, vc, label in flagged:
            lines.append(f"  {str(col)[:55]:<55}  {label}")

    # ── per-column distributions ──
    lines.append(section("PER-COLUMN DISTRIBUTIONS"))

    for col in df.columns:
        lines.append(f"\n  ┌─ [{col}]")
        if col in num_cols:
            for l in analyze_numeric(col, df[col], n_rows):
                lines.append("  │" + l)
        else:
            for l in analyze_categorical(col, df[col], n_rows):
                lines.append("  │" + l)
        lines.append("  └" + "─" * 68)

    lines.append(f"\n{hr('═')}")
    lines.append("  End of Report")
    lines.append(hr("═"))
    return "\n".join(lines)


# ── walker ────────────────────────────────────────────────────────────────────

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        sys.exit(f"[ERROR] Not a directory: {root}")

    print(f"[*] Scanning subfolders of: {root}")

    found, skipped_dirs = 0, 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()  # walk in alphabetical order
        if "features.csv" in filenames:
            csv_path = os.path.join(dirpath, "features.csv")
            rel = os.path.relpath(dirpath, root)
            print(f"  [ANALYZING] {rel}/features.csv ...")
            report = build_report(csv_path)

            out_path = os.path.join(dirpath, "eda_report.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(f"  [DONE] Report written → {out_path}")
            found += 1

    if found == 0:
        print("[!] No features.csv found anywhere under the root.")

    print(f"\n[✅] Done. Found and analyzed: {found} features.csv file(s)")


if __name__ == "__main__":
    main()
