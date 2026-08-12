# SKILL: Analysis Execution

# Version: 1.0

# Reproduces: Bichindaritz et al. 2021, ML_ordCOX pipeline

## Purpose

Reproduce the full ML_ordCOX pipeline from paper Section 2:

1. lmQCM gene co-expression clustering
2. Eigengene extraction
3. biLSTM ordinal Cox model with adaptive multi-task loss
4. 10-fold cross-validation
5. C-index and log-rank test evaluation

## ISOLATION RULE

outputs/extracted_results.json must NOT be read during this skill.
It has been renamed to outputs/.verification_lock by the caller.
Do not attempt to access it under any name or path.

## Input

- data/mrna_brca.parquet
- data/methylation_brca.parquet
- data/clinical_brca.csv
- outputs/extracted_results.json IS NOT AVAILABLE -- do not look for it

## Setup

```python
import yaml
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from scipy.stats import chi2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

with open("paper_config.yaml") as f:
    config = yaml.safe_load(f)

os.makedirs("outputs", exist_ok=True)

# Load data
mrna = pd.read_parquet("data/mrna_brca.parquet")
methylation = pd.read_parquet("data/methylation_brca.parquet")
clinical = pd.read_csv("data/clinical_brca.csv", index_col=0)

print(f"mRNA: {mrna.shape}, methylation: {methylation.shape}, "
      f"clinical: {clinical.shape}")
```

## Step 1: lmQCM gene co-expression clustering

Parameters from paper Section 3.1: γ=0.30, t=1, α=1, β=0.4

```python
from biolearns.coexpression import lmQCM
from biolearns.preprocessing import expression_filter

def run_lmqcm(data, gamma=0.30, label=""):
    """
    Run lmQCM clustering and return eigengene matrix.
    data: genes x samples DataFrame
    Returns: samples x modules DataFrame (eigengenes)
    """
    print(f"Running lmQCM on {label}: {data.shape}")

    # Filter low-expression genes (paper uses mean/var filtering)
    data_filtered = expression_filter(data, meanq=0.5, varq=0.5)
    print(f"After expression filter: {data_filtered.shape}")

    lobj = lmQCM(data_filtered, gamma=gamma)
    clusters, genes, eigengene_mat = lobj.fit()

    print(f"lmQCM found {len(clusters)} modules for {label}")

    # eigengene_mat: modules x samples -- transpose to samples x modules
    eigengenes = pd.DataFrame(
        eigengene_mat.T,
        columns=[f"{label}_module_{i}" for i in range(eigengene_mat.shape[0])]
    )
    return eigengenes

# Run on mRNA and methylation separately (paper Section 2.2)
mrna_eigengenes = run_lmqcm(mrna, gamma=0.30, label="mrna")
meth_eigengenes = run_lmqcm(methylation, gamma=0.30, label="meth")

print(f"mRNA eigengenes: {mrna_eigengenes.shape}")
print(f"Methylation eigengenes: {meth_eigengenes.shape}")

# Concatenate (paper Section 2.2: 116 + 17 = 133 features)
features = pd.concat([mrna_eigengenes, meth_eigengenes], axis=1)
print(f"Combined features: {features.shape}")
```

## Step 2: Prepare survival data

```python
def prepare_survival(clinical):
    """
    Convert clinical DataFrame to structured array for sksurv.
    """
    # OS_status: 1=deceased, 0=censored
    # OS_time: months
    y = np.array(
        [(bool(row["OS_status"]), row["OS_time"])
         for _, row in clinical.iterrows()],
        dtype=[("event", bool), ("time", float)]
    )
    return y

y = prepare_survival(clinical)
X = features.values.astype(np.float32)

print(f"Features shape: {X.shape}")
print(f"Events: {y['event'].sum()} / {len(y)}")
```

## Step 3: biLSTM ordinal Cox model

Faithful reimplementation of paper Section 2.3-2.5.

```python
class OrdinalCoxLoss(nn.Module):
    """
    Multi-task loss combining Cox partial likelihood (main)
    with ordinal ranking loss (auxiliary).
    Adaptive weights via negative exponential (paper Section 2.4).
    """
    def __init__(self):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.zeros(1))  # main task weight
        self.lambda2 = nn.Parameter(torch.zeros(1))  # aux task weight

    def cox_loss(self, risk, event, time):
        """Negative partial log-likelihood (equation 5)."""
        # Sort by time descending
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        event = event[order]

        log_cumsum = torch.logcumsumexp(risk, dim=0)
        loss = -torch.mean((risk - log_cumsum) * event)
        return loss

    def ordinal_loss(self, risk, event, time):
        """Ordinal ranking loss (equation 7)."""
        n = len(risk)
        loss = 0.0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j and time[i] < time[j]:
                    rec_ij = torch.exp(risk[i] - risk[j])
                    loss += torch.clamp(1 - rec_ij, min=0)
                    count += 1
        if count > 0:
            loss = loss / count
        return loss

    def forward(self, risk, event, time):
        """
        Combined adaptive multi-task loss (equation 8).
        K_i(lambda_i) = exp(-lambda_i)
        """
        k1 = torch.exp(-self.lambda1)
        k2 = torch.exp(-self.lambda2)

        l_cox = self.cox_loss(risk, event, time)
        l_ord = self.ordinal_loss(risk, event, time)

        total = k1 * l_cox + k2 * l_ord
        return total, l_cox.item(), l_ord.item()


class BiLSTMCox(nn.Module):
    """
    Bidirectional LSTM Cox model (paper Section 2.5).
    Input: (batch, seq_len, features) -- seq_len=1 for tabular data
    Output: risk score per sample
    """
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        lstm_out, _ = self.bilstm(x)
        # Take last timestep
        out = lstm_out[:, -1, :]
        risk = self.fc(out).squeeze(-1)
        return risk
```

## Step 4: Training function

```python
def train_fold(X_train, y_train, n_epochs=1000, lr=0.001):
    """
    Train biLSTM ordinal Cox model for one fold.
    Learning rate decays by 0.5 every 100 epochs (paper Section 3.2).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)

    X_tensor = torch.FloatTensor(X_scaled)
    time_tensor = torch.FloatTensor(y_train["time"])
    event_tensor = torch.FloatTensor(y_train["event"].astype(float))

    input_dim = X_scaled.shape[1]
    model = BiLSTMCox(input_dim=input_dim)
    criterion = OrdinalCoxLoss()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        risk = model(X_tensor)
        loss, l_cox, l_ord = criterion(risk, event_tensor, time_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} "
                  f"loss={loss.item():.4f} "
                  f"cox={l_cox:.4f} ord={l_ord:.4f}")

    return model, scaler


def predict_risk(model, scaler, X_test):
    model.eval()
    X_scaled = scaler.transform(X_test).astype(np.float32)
    with torch.no_grad():
        risk = model(torch.FloatTensor(X_scaled))
    return risk.numpy()
```

## Step 5: 10-fold cross-validation

```python
def run_cross_validation(X, y, n_folds=10, n_epochs=1000):
    """
    10-fold cross-validation with C-index evaluation.
    Returns per-fold C-indices and aggregate risk scores.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_cindices = []
    all_risks = np.zeros(len(y))
    all_train_risks = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/{n_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, scaler = train_fold(X_train, y_train, n_epochs=n_epochs)
        risk_test = predict_risk(model, scaler, X_test)
        all_risks[test_idx] = risk_test

        # Compute train risk scores for stratification threshold
        risk_train = predict_risk(model, scaler, X_train)
        all_train_risks.extend(risk_train.tolist())

        # C-index for this fold
        cindex = concordance_index_censored(
            y_test["event"],
            y_test["time"],
            risk_test
        )[0]
        fold_cindices.append(cindex)
        print(f"  Fold {fold+1} C-index: {cindex:.4f}")

    return fold_cindices, all_risks, np.array(all_train_risks)

print("Starting 10-fold cross-validation...")
fold_cindices, all_risks, train_risks = run_cross_validation(
    X, y, n_folds=10, n_epochs=1000
)

mean_cindex = np.mean(fold_cindices)
std_cindex = np.std(fold_cindices)
print(f"\nFinal C-index: {mean_cindex:.4f} ± {std_cindex:.4f}")
```

## Step 6: Survival stratification

```python
def compute_logrank_p(risk_scores, y, threshold=None):
    """
    Stratify patients into high/low risk using median risk as threshold.
    Returns log-rank test p-value.
    """
    if threshold is None:
        threshold = np.median(risk_scores)

    high_risk = risk_scores >= threshold
    low_risk = ~high_risk

    # Log-rank test statistic
    from sksurv.compare import compare_survival
    group = high_risk.astype(int)
    chi2_stat, p_value = compare_survival(y, group)
    return p_value, threshold

logrank_p, threshold = compute_logrank_p(all_risks, y)
print(f"Log-rank p-value: {logrank_p:.2e}")
```

## Step 7: KM curves

```python
def plot_km_curves(risk_scores, y, threshold, output_path):
    high_risk = risk_scores >= threshold
    fig, ax = plt.subplots(figsize=(8, 6))

    for group, label, color in [
        (high_risk, "High risk", "red"),
        (~high_risk, "Low risk", "blue"),
    ]:
        t, s = kaplan_meier_estimator(y["event"][group], y["time"][group])
        ax.step(t, s, where="post", label=label, color=color)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title("ML_ordCOX -- Kaplan-Meier survival curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"KM curves saved to {output_path}")

plot_km_curves(all_risks, y, threshold, "outputs/km_curves.png")
```

## Step 8: Save reproduced results

```python
reproduced = {
    "model": "ML_ordCOX_reimplementation",
    "baselines": {
        "ML_ordCOX": {
            "cindex": round(float(mean_cindex), 4),
            "std": round(float(std_cindex), 4),
        }
    },
    "stratification": {
        "ML_ordCOX_logrank_p": float(logrank_p),
    },
    "fold_cindices": [round(float(c), 4) for c in fold_cindices],
    "lmqcm_modules": {
        "mrna": mrna_eigengenes.shape[1],
        "methylation": meth_eigengenes.shape[1],
        "total_features": features.shape[1],
    },
    "dataset": {
        "n_samples": len(y),
        "n_events": int(y["event"].sum()),
    },
}

with open("outputs/reproduced_results.json", "w") as f:
    json.dump(reproduced, f, indent=2)

print("\nReproduced results saved.")
print(json.dumps(reproduced, indent=2))
```

## Output

- `outputs/reproduced_results.json`
- `outputs/km_curves.png`

## Known limitations

The ordinal loss (equation 7) is O(n²) in the number of patients.
For 485 patients this is ~235k pair comparisons per forward pass.
If training is too slow, reduce to a sampled subset of pairs:

```python
# Fast approximation: sample max_pairs random pairs instead of all
max_pairs = 1000
indices = torch.randperm(n)[:max_pairs]
```

Document any approximation in outputs/verification_report.md.

## Generalization

To use with a different paper:

1. Replace BiLSTMCox with the paper's model architecture
2. Replace OrdinalCoxLoss with the paper's loss function
3. Cross-validation and evaluation logic is generic
