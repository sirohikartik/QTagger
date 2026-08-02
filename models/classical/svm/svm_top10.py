from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler  # Added for SVM scaling
from sklearn.svm import SVC                       # Added SVM Classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

TRAIN_ROOT = Path("/Users/jieminphan/Desktop/ransomware_pipeline/train_set")
TEST_ROOT = Path("/Users/jieminphan/Desktop/ransomware_pipeline/test_set")

N_ROWS = 100
WINDOW_SIZE = 10
STRIDE = 5
RANDOM_STATE = 42

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
# DATA LOADING (Summary Statistics)
# ============================================================

def load_dataset(root):
    all_data = []
    for category in sorted(root.iterdir()):
        if not category.is_dir():
            continue
        label = "goodware" if category.name == "goodware" else "ransomware"
        for sample in sorted(category.iterdir()):
            csv = sample / "features.csv"
            if not csv.exists():
                continue
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
                window_dict["binary_label"] = label
                window_dict["sample_name"] = sample.name
                all_data.append(window_dict)
    return pd.DataFrame(all_data)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading training dataset...")
train_df = load_dataset(TRAIN_ROOT)
print("Loading testing dataset...")
test_df = load_dataset(TEST_ROOT)

# Features to drop as requested previously
FEATURES_TO_DROP = [
    "window_idx", "file_entropy_delta", "encryption_slope", "mean_file_entropy",
    "ransom_note_hits", "library_call_rate_openssl", "net_packet_rate",
    "net_byte_volume", "unique_dest_ip_rate", "dns_query_rate",
    "connection_failure_rate", "drop_rate", "vmrss_kb",
]

DROP_COLUMNS = ["binary_label", "sample_name"]
for col in train_df.columns:
    if any(col.startswith(f + "_") for f in FEATURES_TO_DROP) or col in FEATURES_TO_DROP:
        DROP_COLUMNS.append(col)

X_full = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["binary_label"]
X_test_full = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["binary_label"]
X_test_full = X_test_full[X_full.columns]

from sklearn.calibration import CalibratedClassifierCV  # Added to fix warning

# ============================================================
# STEP 1: FIND TOP 10 FEATURES (FIXED CORRECTION)
# ============================================================

print("\nTraining initial model to find top 10 features...")

# 1. Manually transform data for selection so importances match correctly
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_transformed = scaler.fit_transform(imputer.fit_transform(X_full))

# 2. Fit selection model on scaled data
selector_svm = SVC(
    kernel="linear",
    random_state=RANDOM_STATE,
    class_weight="balanced",
)
selector_svm.fit(X_train_transformed, y_train)

# 3. Extract correct, normalized importances
# .ravel() handles multi-dimensional arrays if present
importances = np.abs(selector_svm.coef_.ravel()) 

importance_df = pd.DataFrame({
    "Feature": X_full.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

top_10_features = importance_df["Feature"].iloc[:10].tolist()
print("Top 10 Features selected:")
print(top_10_features)

# ============================================================
# STEP 2: TRAIN TOP 10 MODEL (FIXED ACCURACY & DEPRECATION)
# ============================================================

X_train_top10 = X_full[top_10_features]
X_test_top10 = X_test_full[top_10_features]

print("\nTraining final model on Top 10 features...")

# Tuning C and gamma helps map malicious software footprints more tightly
base_svm = SVC(
    kernel="rbf", 
    C=10.0,            # Higher C focuses on correct classification over margin width
    gamma="scale",     # Adapts kernel radius according to feature variance
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

# CalibratedClassifierCV native wrapper handles predictable 'predict_proba' 
calibrated_svm = CalibratedClassifierCV(base_svm, ensemble=False, cv=5)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", calibrated_svm),
])
model.fit(X_train_top10, y_train)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL RESULTS (TOP 10 FEATURES)")
print("=" * 80)

# Window-level accuracy
pred = model.predict(X_test_top10)
acc = accuracy_score(y_test, pred)
print(f"\nWindow-level Accuracy : {acc:.4f}\n")

print("Classification Report")
print(classification_report(y_test, pred))

print("Confusion Matrix")
cm = confusion_matrix(y_test, pred, labels=["goodware", "ransomware"])
print(pd.DataFrame(cm, index=["Actual Goodware", "Actual Ransomware"], columns=["Pred Goodware", "Pred Ransomware"]))

# ============================================================
# SAMPLE-LEVEL RESULTS (Probabilities @ 0.40)
# ============================================================

print("\n" + "=" * 80)
print("PER SAMPLE RESULTS (Prob Average @ 0.40)")
print("=" * 80)

for sample in sorted(test_df["sample_name"].unique()):
    subset = test_df[test_df["sample_name"] == sample]
    
    # We must drop the same columns and then subset to top 10
    X = subset.drop(columns=DROP_COLUMNS)
    X = X[top_10_features]

    probs = model.predict_proba(X)
    idx = list(model.classes_).index("ransomware")
    score = probs[:, idx].mean()
    
    pred_label = "ransomware" if score > 0.40 else "goodware"
    true_label = subset["binary_label"].iloc[0]
    correct = pred_label == true_label

    print(f"{sample:40s} True={true_label:11s} Pred={pred_label:11s} {'✓' if correct else '✗'}")

print("\nFinished.")