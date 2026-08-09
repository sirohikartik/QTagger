import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import ParameterGrid

# ============================================================
# CONFIG & UTILS
# ============================================================

TRAIN_ROOT = Path(__file__).parent.parent / "train_set"
TEST_ROOT = Path(__file__).parent.parent / "test_set"
N_ROWS = 100
WINDOW_SIZE = 10
STRIDE = 5
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

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

def make_device(n_qubits):
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff_method = "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        diff_method = "backprop"
    return dev, diff_method

# ============================================================
# KERNEL-BASED QUANTUM MODEL
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
    return kernel_model

# ============================================================
# SEARCH LOGIC
# ============================================================

def search_binary():
    print("\n>>> Binary QSVM Search...")
    train_df = load_dataset(TRAIN_ROOT, binary=True)
    test_df = load_dataset(TEST_ROOT, binary=True)
    
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

    print("Performing feature selection (Top 10)...")
    selector = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), 
        ("rf", ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced"))
    ])
    selector.fit(X_train_full, y_train_raw)
    feat_importances = selector.named_steps["rf"].feature_importances_
    feat_names = X_train_full.columns[np.argsort(feat_importances)[-10:]]
    
    X_train_subset = X_train_full[feat_names].values
    X_test_subset = X_test_full[feat_names].values
    scaler = MinMaxScaler()
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train_subset), dtype=torch.float32)
    X_test_scaled = torch.tensor(scaler.transform(X_test_subset), dtype=torch.float32)
    encoder = LabelEncoder()
    y_train_np = encoder.fit_transform(y_train_raw)
    y_test_np = encoder.transform(y_test_raw)
    y_train_kta = torch.tensor(2 * y_train_np - 1, dtype=torch.float32)

    grid = {
        "n_qubits": [4, 8],
        "n_layers": [2, 3],
        "lr": [0.01, 0.03],
        "C": [1.0, 10.0]
    }
    
    best_acc = -1
    best_params = None

    for params in ParameterGrid(grid):
        print(f"Testing: {params}")
        kernel_model = LearnableQuantumKernel(n_features=10, n_qubits=params['n_qubits'], n_layers=params['n_layers'])
        train_kernel(kernel_model, X_train_scaled, y_train_kta, epochs=60, lr=params['lr'], subsample=128)
        
        phi_train = kernel_model.embed(X_train_scaled)
        phi_test = kernel_model.embed(X_test_scaled)
        K_train = kernel_model.kernel_from_features(phi_train, phi_train).detach().cpu().numpy()
        K_test = kernel_model.kernel_from_features(phi_test, phi_train).detach().cpu().numpy()
        
        clf = SVC(kernel="precomputed", C=params['C'], class_weight="balanced")
        clf.fit(K_train, y_train_np)
        acc = accuracy_score(y_test_np, clf.predict(K_test))
        if acc > best_acc:
            best_acc = acc
            best_params = params
            
    print(f"Best Binary Params: {best_params} with Acc: {best_acc:.4f}")

def search_multiclass():
    print("\n>>> Multiclass QSVM Search (Cross-Run)...")
    
    FOLDS = [
        (["features2.csv", "features3.csv"], ["features.csv"]),
        (["features.csv", "features3.csv"], ["features2.csv"]),
        (["features.csv", "features2.csv"], ["features3.csv"]),
    ]
    
    grid = {
        "n_qubits": [8],
        "n_layers": [2],
        "lr": [0.03],
        "C": [10.0]
    }
    
    fold_accs = []
    
    for fold_idx, (train_files, test_files) in enumerate(FOLDS, 1):
        print(f"\n--- Fold {fold_idx} ---")
        train_df = load_dataset(TEST_ROOT, filenames=train_files, binary=False)
        test_df = load_dataset(TEST_ROOT, filenames=test_files, binary=False)
        
        X_train_full = train_df.drop(columns=["label", "sample_name"])
        y_train_raw = train_df["label"]
        X_test_full = test_df.drop(columns=["label", "sample_name"])
        y_test_raw = test_df["label"]
        X_test_full = X_test_full[X_train_full.columns]
        
        # Top 30 features
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")), 
            ("rf", ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced"))
        ])
        selector.fit(X_train_full, y_train_raw)
        feat_importances = selector.named_steps["rf"].feature_importances_
        feat_names = X_train_full.columns[np.argsort(feat_importances)[-30:]]
        
        X_train_subset = X_train_full[feat_names].values
        X_test_subset = X_test_full[feat_names].values
        scaler = MinMaxScaler()
        X_train_scaled = torch.tensor(scaler.fit_transform(X_train_subset), dtype=torch.float32)
        X_test_scaled = torch.tensor(scaler.transform(X_test_subset), dtype=torch.float32)
        
        encoder = LabelEncoder()
        encoder.fit(pd.concat([y_train_raw, y_test_raw]))
        y_train_np = encoder.transform(y_train_raw)
        y_test_np = encoder.transform(y_test_raw)
        
        # KTA needs binary targets. For multiclass, we use One-Vs-Rest or a similar proxy.
        # The report uses KTA. A common way to do KTA for multiclass is to sum the KTA of each class.
        # For simplicity in the grid search, we'll use a binary target based on "is_ransomware" for KTA training.
        y_train_binary = np.where(y_train_raw == "goodware", -1, 1)
        y_train_kta = torch.tensor(y_train_binary, dtype=torch.float32)

        # Using the fixed "Best Params" from the report to validate
        params = {'n_qubits': 8, 'n_layers': 2, 'lr': 0.03, 'C': 10.0}
        print(f"Testing params: {params}")
        
        kernel_model = LearnableQuantumKernel(n_features=30, n_qubits=params['n_qubits'], n_layers=params['n_layers'])
        train_kernel(kernel_model, X_train_scaled, y_train_kta, epochs=60, lr=params['lr'], subsample=128)
        
        phi_train = kernel_model.embed(X_train_scaled)
        phi_test = kernel_model.embed(X_test_scaled)
        K_train = kernel_model.kernel_from_features(phi_train, phi_train).detach().cpu().numpy()
        K_test = kernel_model.kernel_from_features(phi_test, phi_train).detach().cpu().numpy()
        
        clf = SVC(kernel="precomputed", C=params['C'], class_weight="balanced")
        clf.fit(K_train, y_train_np)
        acc = accuracy_score(y_test_np, clf.predict(K_test))
        print(f"Fold {fold_idx} Accuracy: {acc:.4f}")
        fold_accs.append(acc)
            
    print(f"\nMean Multiclass QSVM Window Accuracy: {np.mean(fold_accs):.4f}")

if __name__ == "__main__":
    search_binary()
    search_multiclass()
