"""
QSVM with 3-fold cross-run evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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

# BEST PARAMS
PARAMS = {
    "n_qubits": 8,
    "n_layers": 2,
    "epochs": 60,
    "lr": 0.03,
    "C": 10.0
}

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

def make_device(n_qubits):
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff_method = "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        diff_method = "backprop"
    return dev, diff_method

# ============================================================
# QSVM COMPONENTS
# ============================================================

class QuantumFeatureMap(nn.Module):
    def __init__(self, n_features, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        dev, diff_method = make_device(n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(x, weights):
            for l in range(n_layers):
                qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[l : l + 1], wires=range(n_qubits))
            
            obs = []
            for i in range(n_qubits):
                obs.append(qml.expval(qml.PauliX(i)))
                obs.append(qml.expval(qml.PauliY(i)))
                obs.append(qml.expval(qml.PauliZ(i)))
            
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
                    obs.append(qml.expval(qml.PauliX(i) @ qml.PauliX(j)))
            return obs

        self.circuit = circuit
        self.pre = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits),
            nn.Tanh()
        )
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3))

    def forward(self, x):
        z = self.pre(x) * np.pi / 2
        phis = []
        for sample in z:
            out = self.circuit(sample, self.weights)
            phis.append(torch.stack(out))
        return torch.stack(phis)

    def get_embedding_dim(self):
        return 3 * self.n_qubits + self.n_qubits * (self.n_qubits - 1)

class QuantumClassifier(nn.Module):
    def __init__(self, n_features, n_qubits, n_layers, n_classes=2):
        super().__init__()
        self.feature_map = QuantumFeatureMap(n_features, n_qubits, n_layers)
        self.classifier = nn.Linear(self.feature_map.get_embedding_dim(), n_classes)

    def forward(self, x):
        phi = self.feature_map(x)
        return self.classifier(phi)

def train_classifier(model, X_train, y_train, epochs, lr, weights=None, subsample=128):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    n = X_train.shape[0]
    
    model.train()
    for epoch in range(epochs):
        if subsample and subsample < n:
            idx = torch.randperm(n)[:subsample]
            xb, yb = X_train[idx], y_train[idx]
        else:
            xb, yb = X_train, y_train
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    return model

@torch.no_grad()
def predict_classifier(model, X_test):
    model.eval()
    logits = model(X_test)
    return torch.argmax(logits, dim=1).cpu().numpy()

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

        # Keep a stable, explicit row index for test_df so predictions can
        # always be re-attached to the exact row they came from, regardless
        # of any later sorting/grouping.
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
        selector.fit(X_train_full, y_train_raw)  # fit on TRAIN ONLY -- no leakage
        feat_importances = selector.named_steps["rf"].feature_importances_
        feat_names = X_train_full.columns[np.argsort(feat_importances)[-30:]]
        
        X_train_subset = X_train_full[feat_names].values
        X_test_subset = X_test_full[feat_names].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = torch.tensor(scaler.fit_transform(X_train_subset), dtype=torch.float32)  # fit on TRAIN ONLY
        X_test_scaled = torch.tensor(scaler.transform(X_test_subset), dtype=torch.float32)          # test only transformed
        
        encoder = LabelEncoder()
        y_train_np = encoder.fit_transform(y_train_raw)
        y_test_np = encoder.transform(y_test_raw)
        y_train_kta = torch.tensor(y_train_np, dtype=torch.long)

        print(f"Classes: {list(encoder.classes_)}")

        # Calculate class weights for balanced loss
        class_counts = np.bincount(y_train_np)
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / np.sum(weights) * len(class_counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        print(f"\nTraining Quantum Classifier with params: {PARAMS}")
        model = QuantumClassifier(n_features=30, n_qubits=PARAMS['n_qubits'], n_layers=PARAMS['n_layers'], n_classes=len(encoder.classes_))
        train_classifier(model, X_train_scaled, y_train_kta, epochs=PARAMS['epochs'], lr=PARAMS['lr'], weights=weights_tensor, subsample=128)
        
        print("\nPredicting labels...")
        preds_encoded = predict_classifier(model, X_test_scaled)
        preds_labels = encoder.inverse_transform(preds_encoded)

        acc = accuracy_score(y_test_np, preds_encoded)
        print(f"\nWindow Accuracy: {acc:.4f}")

        print("\nClassification Report")
        print(classification_report(y_test_np, preds_encoded))

        print("\nConfusion Matrix")
        cm = confusion_matrix(y_test_np, preds_encoded)
        print(pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_))

        # --------------------------------------------------------------
        # FIX: per-sample aggregation.
        #
        # The previous version assumed `all_preds` (in test_df row order,
        # i.e. category-then-sample load order) lined up with
        # `sorted(test_df["sample_name"].unique())` (global alphabetical
        # order across ALL categories). Those two orderings do not match
        # in general, so a running `window_idx` counter silently sliced
        # each sample's predictions out of the wrong block -> every
        # sample ends up being scored against someone else's windows.
        # That's why window accuracy (65%) and sample accuracy (0%)
        # could coexist: individual window predictions were fine, only
        # the bookkeeping that reassembled them per-sample was wrong.
        #
        # Fix: attach predictions to test_df as a column (this is safe
        # because X_test_scaled / preds_encoded / preds_labels were built
        # directly from test_df's row order, with no re-sorting in
        # between), then use groupby("sample_name"), which is correct
        # regardless of row/sort order -- no manual index math needed.
        # --------------------------------------------------------------
        assert len(preds_labels) == len(test_df), "prediction length mismatch with test_df"
        test_df = test_df.copy()
        test_df["pred_label"] = preds_labels

        print("\nPer-sample predictions")
        sample_correct = 0
        total_samples = 0

        for sample_name, group in test_df.groupby("sample_name", sort=True):
            true_label = group["label"].iloc[0]
            # sanity check: a sample should never contain more than one true label
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
