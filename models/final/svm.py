import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

# ============================================================
# CONFIG & UTILS
# ============================================================

TRAIN_ROOT = Path(__file__).parent.parent / "train_set"
TEST_ROOT = Path(__file__).parent.parent / "test_set"
N_ROWS = 100
WINDOW_SIZE = 10
STRIDE = 5
RANDOM_STATE = 42

FEATURES_TO_DROP = [
    "window_idx", "file_entropy_delta", "encryption_slope", "mean_file_entropy",
    "ransom_note_hits", "library_call_rate_openssl", "net_packet_rate",
    "net_byte_volume", "unique_dest_ip_rate", "dns_query_rate",
    "connection_failure_rate", "drop_rate", "vmrss_kb",
]

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

def load_dataset(root, filenames=["features.csv"], binary=True):
    all_data = []
    for category in sorted(root.iterdir()):
        if not category.is_dir(): continue
        for sample in sorted(category.iterdir()):
            if not sample.is_dir(): continue
            label = "goodware" if category.name == "goodware" else "ransomware" if binary else sample.name
            for fn in filenames:
                csv = sample / fn
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
                    label_col = "binary_label" if binary else "label"
                    window_dict[label_col] = label
                    window_dict["sample_name"] = sample.name
                    all_data.append(window_dict)
    return pd.DataFrame(all_data)

def prepare_data(df, binary=True):
    label_col = "binary_label" if binary else "label"
    drop_cols = [label_col, "sample_name"]
    for col in df.columns:
        if any(col.startswith(f + "_") for f in FEATURES_TO_DROP) or col in FEATURES_TO_DROP:
            drop_cols.append(col)
    X = df.drop(columns=drop_cols)
    y = df[label_col]
    return X, y

def run_svm_grid_search(X_train, y_train, X_test, y_test, task_name, top_n=None):
    print(f"\n{'='*20} SVM RBF Grid Search for {task_name} {'='*20}")
    
    if top_n is not None:
        print(f"Performing feature selection (Top {top_n})...")
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")),
        ])
        selector.fit(X_train, y_train)
        importances = selector.named_steps["rf"].feature_importances_
        top_features = X_train.columns[np.argsort(importances)[-top_n:]].tolist()
        X_train = X_train[top_features]
        X_test = X_test[top_features]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    
    y_pred = grid_search.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return test_acc

if __name__ == "__main__":
    # Binary Task
    print("\n>>> Starting Binary SVM Task...")
    train_df_bin = load_dataset(TRAIN_ROOT, binary=True)
    test_df_bin = load_dataset(TEST_ROOT, binary=True)
    X_train_bin, y_train_bin = prepare_data(train_df_bin, binary=True)
    X_test_bin, y_test_bin = prepare_data(test_df_bin, binary=True)
    X_test_bin = X_test_bin[X_train_bin.columns]
    run_svm_grid_search(X_train_bin, y_train_bin, X_test_bin, y_test_bin, "Binary SVM", top_n=10)

    # Multiclass Task (Cross-Run)
    print("\n>>> Starting Multiclass SVM Task...")
    FOLDS = [
        (["features2.csv", "features3.csv"], ["features.csv"]),
        (["features.csv", "features3.csv"], ["features2.csv"]),
        (["features.csv", "features2.csv"], ["features3.csv"]),
    ]
    all_fold_accs = []
    for fold_idx, (train_files, test_files) in enumerate(FOLDS, 1):
        print(f"\n--- Fold {fold_idx} ---")
        train_df = load_dataset(TEST_ROOT, filenames=train_files, binary=False)
        test_df = load_dataset(TEST_ROOT, filenames=test_files, binary=False)
        X_train, y_train = prepare_data(train_df, binary=False)
        X_test, y_test = prepare_data(test_df, binary=False)
        X_test = X_test[X_train.columns]
        acc = run_svm_grid_search(X_train, y_train, X_test, y_test, f"Multiclass SVM Fold {fold_idx}", top_n=30)
        all_fold_accs.append(acc)

    print(f"\nMean Multiclass SVM Window Accuracy: {np.mean(all_fold_accs):.4f}")
