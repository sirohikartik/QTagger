"""
QSVM with Best Hyperparameters.

Params:
- n_qubits: 4
- n_layers: 2
- epochs: 60
- lr: 0.03
- C: 10.0
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

TRAIN_ROOT = Path("../train_set")
TEST_ROOT = Path("../test_set")

N_ROWS = 100
WINDOW_SIZE = 10
STRIDE = 5

# BEST PARAMS
PARAMS = {
    "n_qubits": 4,
    "n_layers": 2,
    "epochs": 60,
    "lr": 0.03,
    "C": 10.0
}

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
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + [
                qml.expval(qml.PauliX(i)) for i in range(n_qubits)
            ]

        self.circuit = circuit
        self.pre = nn.Linear(n_features, n_qubits)
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3))

    def forward(self, x):
        z = torch.tanh(self.pre(x)) * np.pi / 2
        phis = []
        for sample in z:
            out = self.circuit(sample, self.weights)
            phis.append(torch.stack(out))
        return torch.stack(phis)

class LearnableQuantumKernel(nn.Module):
    def __init__(self, n_features, n_qubits, n_layers):
        super().__init__()
        self.feature_map = QuantumFeatureMap(n_features, n_qubits, n_layers)
        self.log_gamma = nn.Parameter(torch.tensor(0.0))

    def embed(self, x):
        return self.feature_map(x)

    def kernel_from_features(self, phi_a, phi_b):
        gamma = torch.nn.functional.softplus(self.log_gamma) + 1e-4
        sq_dists = torch.cdist(phi_a, phi_b, p=2) ** 2
        return torch.exp(-gamma * sq_dists)

    def forward(self, x_a, x_b=None):
        phi_a = self.embed(x_a)
        phi_b = phi_a if x_b is None else self.embed(x_b)
        return self.kernel_from_features(phi_a, phi_b)

def kta_loss(K, y):
    y = y.float().view(-1, 1)
    T = y @ y.T
    numerator = (K * T).sum()
    denom = torch.norm(K) * torch.norm(T) + 1e-8
    return -(numerator / denom)

def train_kernel(kernel_model, X_train, y_train, epochs, lr, subsample=128):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr=lr)
    n = X_train.shape[0]
    for epoch in range(epochs):
        if subsample and subsample < n:
            idx = torch.randperm(n)[:subsample]
            xb, yb = X_train[idx], y_train[idx]
        else:
            xb, yb = X_train, y_train
        optimizer.zero_grad()
        K = kernel_model(xb)
        loss = kta_loss(K, yb)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    return kernel_model

@torch.no_grad()
def build_kernel_matrices(kernel_model, X_train, X_test):
    phi_train = kernel_model.embed(X_train)
    phi_test = kernel_model.embed(X_test)
    K_train = kernel_model.kernel_from_features(phi_train, phi_train)
    K_test = kernel_model.kernel_from_features(phi_test, phi_train)
    return K_train.cpu().numpy(), K_test.cpu().numpy()

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

    # Select Top 20 features
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
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train_subset), dtype=torch.float32)
    X_test_scaled = torch.tensor(scaler.transform(X_test_subset), dtype=torch.float32)
    
    encoder = LabelEncoder()
    y_train_np = encoder.fit_transform(y_train_raw)
    y_test_np = encoder.transform(y_test_raw)
    y_train_kta = torch.tensor(2 * y_train_np - 1, dtype=torch.float32)

    print(f"\nTraining Learnable Quantum Kernel with params: {PARAMS}")
    kernel_model = LearnableQuantumKernel(n_features=10, n_qubits=PARAMS['n_qubits'], n_layers=PARAMS['n_layers'])
    train_kernel(kernel_model, X_train_scaled, y_train_kta, epochs=PARAMS['epochs'], lr=PARAMS['lr'], subsample=128)
    
    print("\nBuilding kernel matrices and fitting SVC...")
    K_train, K_test = build_kernel_matrices(kernel_model, X_train_scaled, X_test_scaled)
    clf = SVC(kernel="precomputed", C=PARAMS['C'], class_weight="balanced", random_state=SEED)
    clf.fit(K_train, y_train_np)
    
    preds_encoded = clf.predict(K_test)
    preds_labels = encoder.inverse_transform(preds_encoded)

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS (QSVM BEST PARAMS - n_qubits: {PARAMS['n_qubits']})")
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
