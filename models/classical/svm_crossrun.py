"""
SVM with 3-fold cross-run evaluation.
Classical equivalent of qsvm_crossrun.py using RBF kernel and hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

SEED = 42
np.random.seed(SEED)

# ============================================================
# CONFIG
# ============================================================

TEST_ROOT = Path("../test_set")
TRAIN_ROOT = TEST_ROOT

N_ROWS = 100
WINDOW_SIZE = 10
STRIDE = 5

# Fold configuration: (train_files, test_files)
FOLDS = [
    (["features2.csv", "features3.csv"], ["features.csv"]),
    (["features.csv", "features3.csv"], ["features2.csv"]),
    (["features.csv", "features2.csv"], ["features3.csv"]),
]

# Feature dropout logic
FEATURES_TO_DROP = ["window_idx", "file_entropy_delta", "encryption_slope", "mean_file_entropy", "ransom_note_hits", "library_call_rate_openssl", "net_packet_rate", "net_byte_volume", "unique_dest_ip_rate", "dns_query_rate", "connection_failure_rate", "drop_rate", "vmrss_kb"]

# ============================================================
# UTILS & DATA LOADING
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

def load_dataset(root, filenames=["features.csv"]):
    all_data = []
    for category in sorted(root.iterdir()):
        if not category.is_dir(): continue
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
                for start in range(0, data.shape[0] - WINDOW_SIZE + 1, STRIDE):
                    window_data = data[start : start + WINDOW_SIZE]
                    window_dict = {}
                    for j in range(len(cols)):
                        col = cols[j]
                        vals = window_data[:, j]
                        window_dict[f"{col}_mean"] = np.mean(vals)
                        window_dict[f"{col}_std"] = np.std(vals)
                        window_dict[f"{col}_max"] = np.max(vals)
                        window_dict[f"{col}_min"] = np.min(vals)
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

        print("Loading data...")
        train_df = load_dataset(TRAIN_ROOT, filenames=train_files)
        test_df = load_dataset(TEST_ROOT, filenames=test_files)

        test_df = test_df.reset_index(drop=True)

        cols_to_drop = ["label", "sample_name"]
        for col in train_df.columns:
            if any(col.startswith(f + "_") for f in FEATURES_TO_DROP) or col in FEATURES_TO_DROP:
                cols_to_drop.append(col)

        X_train_full = train_df.drop(columns=cols_to_drop)
        y_train_raw = train_df["label"]
        X_test_full = test_df.drop(columns=cols_to_drop)
        y_test_raw = test_df["label"]
        X_test_full = X_test_full[X_train_full.columns]

        print("Selecting Top 30 features...")
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")), 
            ("rf", ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced"))
        ])
        selector.fit(X_train_full, y_train_raw)
        feat_importances = selector.named_steps["rf"].feature_importances_
        feat_names = X_train_full.columns[np.argsort(feat_importances)[-30:]]
        
        X_train_subset = X_train_full[feat_names].values
        X_test_subset = X_test_full[feat_names].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)
        
        encoder = LabelEncoder()
        y_train_np = encoder.fit_transform(y_train_raw)
        y_test_np = encoder.transform(y_test_raw)

        print(f"Classes: {list(encoder.classes_)}")

        print("\nOptimizing SVM hyperparameters (RBF kernel)...")
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'class_weight': ['balanced']
        }
        
        grid_search = GridSearchCV(
            SVC(kernel='rbf', random_state=SEED),
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1, 
            verbose=1
        )
        grid_search.fit(X_train_scaled, y_train_np)
        
        best_clf = grid_search.best_estimator_
        print(f"Best Parameters for Fold {fold_idx}: {grid_search.best_params_}")
        
        print("\nPredicting labels...")
        preds_encoded = best_clf.predict(X_test_scaled)
        preds_labels = encoder.inverse_transform(preds_encoded)

        acc = accuracy_score(y_test_np, preds_encoded)
        print(f"\nWindow Accuracy: {acc:.4f}")

        print("\nClassification Report")
        print(classification_report(y_test_np, preds_encoded))

        print("\nConfusion Matrix")
        cm = confusion_matrix(y_test_np, preds_encoded)
        print(pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_))

        assert len(preds_labels) == len(test_df), "prediction length mismatch with test_df"
        test_df = test_df.copy()
        test_df["pred_label"] = preds_labels

        print("\nPer-sample predictions")
        sample_correct = 0
        total_samples = 0

        for sample_name, group in test_df.groupby("sample_name", sort=True):
            true_label = group["label"].iloc[0]
            assert group["label"].nunique() == 1, f"sample {sample_name} has mixed labels"

            mode_label = group["pred_label"].mode()[0]
            correct = mode_label == true_label
            if correct:
                sample_correct += 1
            total_samples += 1

            print(f"{sample_name:40s} True={true_label:11s} Pred={mode_label:11s} {'✓' if correct else '✗'}")

        sample_acc = sample_correct / total_samples if total_samples > 0 else 0
        print(f"\nSample Accuracy: {sample_acc:.4f}")
        print("\n")
