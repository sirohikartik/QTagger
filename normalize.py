from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path("dataset")
DST_ROOT = Path("dataset_normalized_100r")

TARGET_ROWS = 100


def downsample(df, target_rows=100):
    """Uniformly sample rows across the execution."""
    idx = np.linspace(0, len(df) - 1, target_rows, dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def interpolate(df, target_rows=100):
    """Interpolate numeric columns to target_rows."""
    old_x = np.linspace(0, 1, len(df))
    new_x = np.linspace(0, 1, target_rows)

    out = pd.DataFrame()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = np.interp(new_x, old_x, df[col].values)
        else:
            # Keep non-numeric columns by nearest-neighbor lookup
            nearest = np.round(
                np.linspace(0, len(df) - 1, target_rows)
            ).astype(int)
            out[col] = df[col].iloc[nearest].values

    return out


def normalize(df):
    n = len(df)

    if n == TARGET_ROWS:
        return df.reset_index(drop=True)

    if n > TARGET_ROWS:
        return downsample(df, TARGET_ROWS)

    return interpolate(df, TARGET_ROWS)


def main():
    count = 0

    for category in SRC_ROOT.iterdir():
        if not category.is_dir():
            continue

        for sample in category.iterdir():
            csv_path = sample / "features.csv"

            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            normalized = normalize(df)

            out_dir = DST_ROOT / category.name / sample.name
            out_dir.mkdir(parents=True, exist_ok=True)

            normalized.to_csv(out_dir / "features.csv", index=False)

            print(
                f"{category.name}/{sample.name:<15} "
                f"{len(df):>4} -> {len(normalized):>3}"
            )

            count += 1

    print(f"\nDone! Normalized {count} datasets.")
    print(f"Saved to: {DST_ROOT.resolve()}")


if __name__ == "__main__":
    main()
