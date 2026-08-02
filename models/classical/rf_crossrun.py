from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

TEST_ROOT = Path("../../../test_set")
TRAIN_ROOT = TEST_ROOT

N_ROWS = 100
WINDOW_SIZE = 10
STRIDE = 5
RANDOM_STATE = 42

# Fold configuration: (train_files, test_files)
FOLDS = [
    (["features2.csv", "features3.csv"], ["features.csv"]),
    (["features.csv", "features3.csv"], ["features2.csv"]),
    (["features.csv", "features2.csv"], ["features3.csv"]),
]

# Features to drop as requested previously
FEATURES_TO_DROP = [
    "window_idx", "file_entropy_delta", "encryption_slope", "mean_file_entropy",
    "ransom_note_hits", "library_call_rate_openssl", "net_packet_rate",
    "net_byte_volume", "unique_dest_ip_rate", "dns_query_rate",
    "connection_failure_rate", "drop_rate", "vmrss_kb",
]

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_rows(df, target_rows=100):
    df = df.reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(df) == target_rows:
        out = df.copy()
    elif len(df) > target_rows:
        idx = np.linspace(0, len(df) - 1, target_rows, dtype=int)
        out = df.iloc[idx].reset_index(drop=True)
    else:
        old_idx = np.arange(len(df))
        new_idx = np.linspace(0, len(df) - 1, target_rows)
        out = pd.DataFrame()
        for col in df.columns:
            if col in numeric_cols:
                out[col] = np.interp(new_idx, old_idx, df[col].values)
            else:
                out[col] = df[col].reindex(range(target_rows)).ffill().bfill().values
    if "window_idx" in out.columns:
        out["window_idx"] = np.arange(target_rows)
    return out

# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(root, filenames=["features.csv"]):
    all_data = []
    for category in sorted(root.iterdir()):
        if not category.is_dir():
            continue
        for sample in sorted(category.iterdir()):
            label = sample.name
            for fn in filenames:
                csv = sample / fn
                if not csv.exists():
                    raise FileNotFoundError(f"Required file {fn} not found for sample {sample.name} in {category.name}")
                df = pd.read_csv(csv)
                df = normalize_rows(df, N_ROWS)
                cols = df.columns.tolist()
                data = df.values
                num_rows = data.shape[0]
                for start in range(0, num_rows - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    window_data = data[start:end]
                    window_dict = {}
                    for j in range(len(cols)):
                        col_name = cols[j]
                        feature_values = window_data[:, j]
                        window_dict[f"{col_name}_mean"] = np.mean(feature_values)
                        window_dict[f"{col_name}_std"] = np.std(feature_values)
                        window_dict[f"{col_name}_max"] = np.max(feature_values)
                        window_dict[f"{col_name}_min"] = np.min(feature_values)
                    window_dict["label"] = label
                    window_dict["sample_name"] = sample.name
                    all_data.append(window_dict)
    return pd.DataFrame(all_data)

# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

if __name__ == "__main__":
    for fold_idx, (train_files, test_files) in enumerate(FOLDS, 1):
        print("\n" + "=" * 40)
        print(f"FOLD {fold_idx}")
        print("=" * 40)

        # Load data for this fold
        train_df = load_dataset(TRAIN_ROOT, filenames=train_files)
        test_df = load_dataset(TEST_ROOT, filenames=test_files)

        # Feature dropout
        DROP_COLUMNS = ["label", "sample_name"]
        for col in train_df.columns:
            if any(col.startswith(f + "_") for f in FEATURES_TO_DROP) or col in FEATURES_TO_DROP:
                DROP_COLUMNS.append(col)

        X_train_full = train_df.drop(columns=DROP_COLUMNS)
        y_train = train_df["label"]
        X_test_full = test_df.drop(columns=DROP_COLUMNS)
        y_test = test_df["label"]
        X_test_full = X_test_full[X_train_full.columns]

        # Step 1: Feature Selection on Training set ONLY
        initial_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", ExtraTreesClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])
        initial_model.fit(X_train_full, y_train)
        
        rf_feat = initial_model.named_steps["rf"]
        importance = pd.DataFrame({
            "Feature": X_train_full.columns,
            "Importance": rf_feat.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        top_30_features = importance["Feature"].iloc[:30].tolist()
        
        # Step 2: Train RF on Top 30 features
        X_train_top30 = X_train_full[top_30_features]
        X_test_top30 = X_test_full[top_30_features]
        
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", ExtraTreesClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])
        model.fit(X_train_top30, y_train)

        # Window-level accuracy
        pred = model.predict(X_test_top30)
        acc = accuracy_score(y_test, pred)
        print(f"\nWindow Accuracy: {acc:.4f}")

        # Classification Report
        print("\nClassification Report")
        print(classification_report(y_test, pred))

        # Confusion Matrix
        print("\nConfusion Matrix")
        cm = confusion_matrix(y_test, pred)
        print(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))

        # Sample-level predictions
        print("\nPer-sample predictions")
        sample_correct = 0
        total_samples = 0
        
        for sample in sorted(test_df["sample_name"].unique()):
            subset = test_df[test_df["sample_name"] == sample]
            X = subset.drop(columns=DROP_COLUMNS)
            X = X[top_30_features]
            
            probs = model.predict(X)
            pred_label = pd.Series(probs).mode()[0]
            true_label = subset["label"].iloc[0]
            correct = pred_label == true_label
            if correct: sample_correct += 1
            total_samples += 1
            
            print(f"{sample:40s} True={true_label:11s} Pred={pred_label:11s} {'✓' if correct else '✗'}")

        sample_acc = sample_correct / total_samples if total_samples > 0 else 0
        print(f"\nSample Accuracy: {sample_acc:.4f}")
        print("\n")
