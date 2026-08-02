"""
SVM with RBF Kernel and Top 10 Features.
Finds the best hyperparameters (C, gamma) and evaluates window-level and sample-level accuracy.
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

TRAIN_ROOT = Path("../train_set")
TEST_ROOT = Path("../test_set")

N_ROWS = 100
WINDOW_SIZE = 10
STRIDE = 5

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

def load_dataset(root):
    all_data = []
    for category in sorted(root.iterdir()):
        if not category.is_dir(): continue
        label = "goodware" if category.name == "goodware" else "ransomware"
        for sample in sorted(category.iterdir()):
            csv = sample / "features.csv"
            if not csv.exists(): continue
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
                window_dict["binary_label"] = label
                window_dict["sample_name"] = sample.name
                all_data.append(window_dict)
    return pd.DataFrame(all_data)

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Loading data...")
    train_df = load_dataset(TRAIN_ROOT)
    test_df = load_dataset(TEST_ROOT)

    # Feature dropout logic
    FEATURES_TO_DROP = ["window_idx", "file_entropy_delta", "encryption_slope", "mean_file_entropy", "ransom_note_hits", "library_call_rate_openssl", "net_packet_rate", "net_byte_volume", "unique_dest_ip_rate", "dns_query_rate", "connection_failure_rate", "drop_rate", "vmrss_kb"]
    cols_to_drop = ["binary_label", "sample_name"]
    for col in train_df.columns:
        if any(col.startswith(f + "_") for f in FEATURES_TO_DROP) or col in FEATURES_TO_DROP:
            cols_to_drop.append(col)

    X_train_full = train_df.drop(columns=cols_to_drop)
    y_train_raw = train_df["binary_label"]
    X_test_full = test_df.drop(columns=cols_to_drop)
    y_test_raw = test_df["binary_label"]
    X_test_full = X_test_full[X_train_full.columns]

    # Select Top 10 features
    print("Selecting Top 10 features...")
    selector = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), 
        ("rf", ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced"))
    ])
    selector.fit(X_train_full, y_train_raw)
    feat_importances = selector.named_steps["rf"].feature_importances_
    feat_names = X_train_full.columns[np.argsort(feat_importances)[-10:]]
    
    X_train_subset = X_train_full[feat_names].values
    X_test_subset = X_test_full[feat_names].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_subset)
    X_test_scaled = scaler.transform(X_test_subset)
    
    encoder = LabelEncoder()
    y_train_np = encoder.fit_transform(y_train_raw)
    y_test_np = encoder.transform(y_test_raw)

    print("\nSearching for best SVM hyperparameters (RBF kernel)...")
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
    print(f"\nBest Parameters: {grid_search.best_params_}")
    
    preds_encoded = best_clf.predict(X_test_scaled)
    preds_labels = encoder.inverse_transform(preds_encoded)

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS (Classical SVM RBF - Top 10 Features)")
    print("=" * 80)

    acc = accuracy_score(y_test_np, preds_encoded)
    print(f"\nWindow-level Accuracy : {acc:.4f}\n")

    print("Classification Report")
    print(classification_report(y_test_np, preds_encoded))

    print("Confusion Matrix")
    cm = confusion_matrix(y_test_np, preds_encoded)
    print(pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_))

    # Sample-level evaluation (Majority Voting)
    print("\n" + "=" * 80)
    print("PER SAMPLE RESULTS (Majority Vote)")
    print("=" * 80)

    all_preds = preds_labels
    window_idx = 0
    for sample in sorted(test_df["sample_name"].unique()):
        subset = test_df[test_df["sample_name"] == sample]
        num_windows = len(subset)
        
        sample_preds = all_preds[window_idx : window_idx + num_windows]
        window_idx += num_windows
        
        mode_label = pd.Series(sample_preds).mode()[0]
        true_label = subset["binary_label"].iloc[0]
        correct = mode_label == true_label

        print(f"{sample:40s} True={true_label:11s} Pred={mode_label:11s} {'✓' if correct else '✗'}")

    print("\nFinished.")
