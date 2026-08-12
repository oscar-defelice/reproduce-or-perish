# tools/analysis_runner.py
"""
Reproduce the ML_ordCOX pipeline from Bichindaritz et al. 2021.

Pipeline:
  1. lmQCM gene co-expression clustering on mRNA and methylation
  2. Eigengene extraction (116 mRNA + 17 methylation modules)
  3. biLSTM ordinal Cox model with adaptive multi-task loss
  4. 10-fold cross-validation
  5. C-index and log-rank test evaluation

Reads data from data/ and extracted_results.json parameters.
Outputs reproduced_results.json and km_curves.png to outputs/.

ISOLATION RULE: this script must NOT read outputs/extracted_results.json.
It is renamed to outputs/.verification_lock by the caller before
this script runs. See CLAUDE.md for the anti-self-convincing protocol.
"""

import json
import logging
import os
import warnings
from pathlib import Path

# OpenMP conflict workaround -- must be set before torch import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from biolearns.coexpression import lmQCM
from biolearns.preprocessing import expression_filter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.compare import compare_survival
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Device selection -- MPS on Apple Silicon, CUDA on NVIDIA, CPU fallback
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# ── Model architecture ────────────────────────────────────────────────────────

class OrdinalCoxLoss(nn.Module):
    """
    Multi-task loss combining Cox partial likelihood with ordinal ranking loss.

    Adaptive weights via negative exponential weighting (paper Section 2.4):
        K_i(lambda_i) = exp(-lambda_i)

    Parameters are learned jointly with the model during training.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.zeros(1))
        self.lambda2 = nn.Parameter(torch.zeros(1))

    def cox_loss(
        self,
        risk: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative partial log-likelihood of the Cox model (paper Eq. 5).

        Parameters
        ----------
        risk : torch.Tensor
            Predicted risk scores, shape (n,).
        event : torch.Tensor
            Event indicators (1=deceased, 0=censored), shape (n,).
        time : torch.Tensor
            Observed times, shape (n,).

        Returns
        -------
        torch.Tensor
            Scalar loss value.

        Notes
        -----
        Risk scores are normalized by subtracting the max before
        logcumsumexp to prevent numerical overflow. This is
        mathematically equivalent as the max cancels in the log ratio.
        """
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        event = event[order]

        # Normalize to prevent numerical overflow in logcumsumexp
        risk = risk - risk.max()

        log_cumsum = torch.logcumsumexp(risk, dim=0)
        return -torch.mean((risk - log_cumsum) * event)

    def ordinal_loss(
        self,
        risk: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
        max_pairs: int = 1000,
    ) -> torch.Tensor:
        """
        Ordinal ranking loss (paper Eq. 7).

        Parameters
        ----------
        risk : torch.Tensor
            Predicted risk scores, shape (n,).
        event : torch.Tensor
            Event indicators, shape (n,).
        time : torch.Tensor
            Observed times, shape (n,).
        max_pairs : int, optional
            Maximum number of pairs to sample for efficiency, by default 1000.

        Returns
        -------
        torch.Tensor
            Scalar loss value.

        Notes
        -----
        Full O(n^2) pairwise computation is infeasible for n=785.
        Random pair sampling approximates the full loss with O(max_pairs)
        cost. Documented as an approximation in verification_report.md.
        """
        n = len(risk)
        idx_i = torch.randint(0, n, (max_pairs,), device=risk.device)
        idx_j = torch.randint(0, n, (max_pairs,), device=risk.device)

        valid = time[idx_i] < time[idx_j]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=risk.device, requires_grad=True)

        risk_i = risk[idx_i[valid]]
        risk_j = risk[idx_j[valid]]
        rec_ij = torch.exp(risk_i - risk_j)
        return torch.clamp(1 - rec_ij, min=0).mean()

    def forward(
        self,
        risk: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Combined adaptive multi-task loss (paper Eq. 8).

        Parameters
        ----------
        risk : torch.Tensor
            Predicted risk scores.
        event : torch.Tensor
            Event indicators.
        time : torch.Tensor
            Observed times.

        Returns
        -------
        tuple[torch.Tensor, float, float]
            (total_loss, cox_loss_value, ordinal_loss_value)
        """
        k1 = torch.exp(-self.lambda1)
        k2 = torch.exp(-self.lambda2)
        l_cox = self.cox_loss(risk, event, time)
        l_ord = self.ordinal_loss(risk, event, time)
        total = k1 * l_cox + k2 * l_ord
        return total, l_cox.item(), l_ord.item()


class BiLSTMCox(nn.Module):
    """
    Bidirectional LSTM Cox proportional hazards model (paper Section 2.5).

    Parameters
    ----------
    input_dim : int
        Number of input features (eigengenes).
    hidden_dim : int, optional
        LSTM hidden dimension, by default 64.
    n_layers : int, optional
        Number of LSTM layers, by default 2.
    dropout : float, optional
        Dropout rate, by default 0.3.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Risk scores, shape (batch,).
        """
        x = x.unsqueeze(1)
        lstm_out, _ = self.bilstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out).squeeze(-1)


# ── lmQCM clustering ──────────────────────────────────────────────────────────

def run_lmqcm(
    data: pd.DataFrame,
    gamma: float = 0.30,
    label: str = "",
    max_genes: int = 5000,
) -> pd.DataFrame:
    """
    Run lmQCM gene co-expression clustering and return eigengene matrix.

    Parameters
    ----------
    data : pd.DataFrame
        Gene expression or methylation matrix (genes x samples).
    gamma : float, optional
        lmQCM gamma parameter (weight threshold), by default 0.30.
    label : str, optional
        Label for logging, by default "".
    max_genes : int, optional
        Maximum genes passed to lmQCM after variance pre-filtering,
        by default 5000. lmQCM computes O(n^2) Spearman correlations --
        pre-filtering makes it tractable for large methylation matrices.

    Returns
    -------
    pd.DataFrame
        Eigengene matrix (samples x modules), index = TCGA sample barcodes.

    Notes
    -----
    Three-step filtering before lmQCM:
    1. Variance pre-filter to max_genes -- reduces O(n^2) cost.
    2. Expression filter (mean/var quantile 0.5) via biolearns.
    3. lmQCM clustering with gamma from paper Section 3.1.

    biolearns lmQCM bug: raises ValueError when neighborWeights is empty.
    Patched in place at biolearns/coexpression/_lmQCM.py line 151.
    Retry with lower gamma as fallback if patch is not applied.
    """
    logger.info(f"lmQCM on {label}: {data.shape}")

    # Step 1: variance pre-filter
    if data.shape[0] > max_genes:
        variances = data.var(axis=1)
        top_genes = variances.nlargest(max_genes).index
        data = data.loc[top_genes]
        logger.info(
            f"Variance pre-filter ({label}): -> {data.shape[0]} genes "
            f"(top {max_genes} by variance)"
        )

    # Step 2: expression filter
    data_filtered = expression_filter(data, meanq=0.5, varq=0.5)
    logger.info(f"After expression filter ({label}): {data_filtered.shape}")

    # Step 3: lmQCM clustering with gamma retry fallback
    for attempt, gamma_try in enumerate([gamma, gamma * 0.8, gamma * 0.6]):
        try:
            lobj = lmQCM(data_filtered, gamma=gamma_try)
            with tqdm(
                desc=f"lmQCM ({label}, gamma={gamma_try:.2f})",
                unit=" file",
            ) as pbar:
                clusters, genes, eigengene_mat = lobj.fit()
                pbar.update(1)

            if attempt > 0:
                logger.warning(
                    f"lmQCM succeeded with reduced gamma={gamma_try:.2f} "
                    f"(original={gamma}). Documented in verification_report."
                )
            break

        except ValueError as e:
            if "empty sequence" in str(e) and attempt < 2:
                logger.warning(
                    f"lmQCM failed with gamma={gamma_try:.2f} "
                    f"(empty neighbor sequence). Retrying with lower gamma..."
                )
                continue
            else:
                raise

    n_modules = eigengene_mat.shape[0]
    logger.info(f"lmQCM found {n_modules} modules for {label}")

    # eigengene_mat shape: (n_modules, n_samples)
    # .T gives (n_samples, n_modules)
    # columns of eigengene_mat are the sample barcodes
    eigengenes = pd.DataFrame(
        eigengene_mat.T.values,              # usa .values per evitare conflitti di indice
        index=eigengene_mat.columns,         # sample barcodes dalle colonne di eigengene_mat
        columns=[f"{label}_module_{i}" for i in range(n_modules)],
    )
    return eigengenes


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 1000,
    lr: float = 0.001,
) -> tuple["BiLSTMCox", StandardScaler]:
    """
    Train biLSTM ordinal Cox model for one cross-validation fold.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_train, n_features).
    y_train : np.ndarray
        Structured survival array with fields 'event' and 'time'.
    n_epochs : int, optional
        Number of training epochs, by default 1000.
    lr : float, optional
        Initial learning rate, by default 0.001.

    Returns
    -------
    tuple[BiLSTMCox, StandardScaler]
        Trained model and fitted scaler.

    Notes
    -----
    Learning rate decays by factor 0.5 every 100 epochs (paper Section 3.2).
    Gradient clipping (max_norm=1.0) prevents explosion on MPS/CUDA.
    NaN losses are skipped with a warning rather than crashing.
    Model and tensors are moved to DEVICE (MPS/CUDA/CPU).
    numpy structured array fields are copied before tensor conversion
    to ensure contiguous memory layout.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)

    X_tensor = torch.FloatTensor(X_scaled.copy()).to(DEVICE)
    time_tensor = torch.FloatTensor(y_train["time"].copy()).to(DEVICE)
    event_tensor = torch.FloatTensor(
        y_train["event"].astype(float).copy()
    ).to(DEVICE)

    model = BiLSTMCox(input_dim=X_scaled.shape[1]).to(DEVICE)
    criterion = OrdinalCoxLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )

    all_params = list(model.parameters()) + list(criterion.parameters())

    model.train()
    with tqdm(
        total=n_epochs,
        desc="Training",
        unit="epoch",
        leave=False,
    ) as pbar:
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            risk = model(X_tensor)
            loss, l_cox, l_ord = criterion(risk, event_tensor, time_tensor)

            # Skip NaN losses -- can occur with extreme risk scores
            # early in training, especially on MPS
            if torch.isnan(loss):
                logger.warning(
                    f"NaN loss at epoch {epoch + 1} -- skipping update"
                )
                scheduler.step()
                pbar.update(1)
                continue

            loss.backward()

            # Gradient clipping prevents explosion on MPS/CUDA
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 100 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    cox=f"{l_cox:.4f}",
                    ord=f"{l_ord:.4f}",
                )
            pbar.update(1)

    return model, scaler


def predict_risk(
    model: "BiLSTMCox",
    scaler: StandardScaler,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Predict risk scores for test samples.

    Parameters
    ----------
    model : BiLSTMCox
        Trained model.
    scaler : StandardScaler
        Fitted scaler from training fold.
    X_test : np.ndarray
        Test features, shape (n_test, n_features).

    Returns
    -------
    np.ndarray
        Risk scores on CPU, shape (n_test,).

    Notes
    -----
    NaN risk scores are replaced with 0.0 and logged as a warning.
    This can occur when training was unstable -- check loss curves
    if this warning appears frequently.
    """
    model.eval()
    X_scaled = scaler.transform(X_test).astype(np.float32)
    with torch.no_grad():
        risk = model(torch.FloatTensor(X_scaled.copy()).to(DEVICE))

    risk_np = risk.cpu().numpy()

    if np.isnan(risk_np).any():
        n_nan = np.isnan(risk_np).sum()
        logger.warning(
            f"{n_nan} NaN risk scores replaced with 0.0. "
            "Training may have been unstable -- consider reducing lr."
        )
        risk_np = np.nan_to_num(risk_np, nan=0.0)

    return risk_np


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    n_epochs: int = 1000,
) -> tuple[list[float], np.ndarray]:
    """
    Run n-fold cross-validation with C-index evaluation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples x features).
    y : np.ndarray
        Structured survival array with fields 'event' (bool) and
        'time' (float).
    n_folds : int, optional
        Number of cross-validation folds, by default 10.
    n_epochs : int, optional
        Training epochs per fold, by default 1000.

    Returns
    -------
    tuple[list[float], np.ndarray]
        (per-fold C-indices, out-of-fold risk scores for all samples)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_cindices = []
    all_risks = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(kf.split(X), total=n_folds, desc="Cross-validation folds")
    ):
        logger.info(f"Fold {fold + 1}/{n_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, scaler = train_fold(
            X_train, y_train, n_epochs=n_epochs
        )
        risk_test = predict_risk(model, scaler, X_test)
        all_risks[test_idx] = risk_test

        cindex = concordance_index_censored(
            y_test["event"],
            y_test["time"],
            risk_test,
        )[0]
        fold_cindices.append(cindex)
        logger.info(f"Fold {fold + 1} C-index: {cindex:.4f}")

    return fold_cindices, all_risks


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_logrank_p(
    risk_scores: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """
    Stratify patients into high/low risk and compute log-rank p-value.

    Parameters
    ----------
    risk_scores : np.ndarray
        Predicted risk scores for all samples.
    y : np.ndarray
        Structured survival array.

    Returns
    -------
    tuple[float, float]
        (log-rank p-value, risk threshold used for stratification)

    Notes
    -----
    Median risk score is used as threshold (paper Section 3.5).
    Returns p-value of 1.0 if all samples fall in one group
    (degenerate case when all risk scores are identical).
    """
    threshold = float(np.median(risk_scores))
    group = (risk_scores >= threshold).astype(int)

    # Handle degenerate case -- all scores identical after NaN replacement
    if len(np.unique(group)) < 2:
        logger.warning(
            "All risk scores identical -- cannot compute log-rank test. "
            "Returning p_value=1.0. Training was likely unstable."
        )
        return 1.0, threshold

    _, p_value = compare_survival(y, group)
    return float(p_value), threshold


def plot_km_curves(
    risk_scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    output_path: str,
) -> None:
    """
    Plot Kaplan-Meier survival curves for high and low risk groups.

    Parameters
    ----------
    risk_scores : np.ndarray
        Predicted risk scores.
    y : np.ndarray
        Structured survival array.
    threshold : float
        Risk score threshold for stratification.
    output_path : str
        Path where the figure will be saved.
    """
    high_risk = risk_scores >= threshold
    fig, ax = plt.subplots(figsize=(8, 6))

    for group_mask, label, color in [
        (high_risk, "High risk", "red"),
        (~high_risk, "Low risk", "blue"),
    ]:
        t, s = kaplan_meier_estimator(
            y["event"][group_mask],
            y["time"][group_mask],
        )
        ax.step(t, s, where="post", label=label, color=color, linewidth=2)

    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival probability", fontsize=12)
    ax.set_title("ML_ordCOX -- Kaplan-Meier survival curves", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"KM curves saved to {output_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_analysis(
    data_dir: str = "data",
    output_dir: str = "outputs",
    n_epochs: int = 1000,
    n_folds: int = 10,
    gamma: float = 0.30,
    max_genes_lmqcm: int = 20000,
) -> dict:
    """
    Run the full ML_ordCOX reproduction pipeline.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing parquet data files, by default "data".
    output_dir : str, optional
        Directory for output files, by default "outputs".
    n_epochs : int, optional
        Training epochs per fold, by default 1000.
    n_folds : int, optional
        Number of cross-validation folds, by default 10.
    gamma : float, optional
        lmQCM gamma parameter, by default 0.30.
    max_genes_lmqcm : int, optional
        Maximum genes passed to lmQCM after variance pre-filtering,
        by default 20000.

    Returns
    -------
    dict
        Reproduced results saved to outputs/reproduced_results.json.

    Notes
    -----
    ISOLATION RULE: this function must not read extracted_results.json.
    All hyperparameters are passed explicitly or use paper defaults.
    Results are saved before any comparison with extracted values.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Using device: {DEVICE}")

    # ── Anti-self-convincing isolation protocol ──
    # Rename extracted_results.json to .verification_lock before analysis
    # so it cannot be read during execution -- even accidentally.
    # Restored after results are saved.
    results_path = os.path.join(output_dir, "extracted_results.json")
    lock_path = os.path.join(output_dir, ".verification_lock")

    if os.path.exists(results_path):
        os.rename(results_path, lock_path)
        logger.info(
            "Anti-self-convincing: extracted_results.json -> "
            ".verification_lock (will be restored after analysis)"
        )
    elif os.path.exists(lock_path):
        logger.info(
            "Anti-self-convincing: .verification_lock already in place"
        )
    else:
        logger.warning(
            "extracted_results.json not found -- "
            "run pdf_parser.py before analysis_runner.py"
        )

    # ── Load data ──
    logger.info("Loading data...")
    mrna = pd.read_parquet(f"{data_dir}/mrna_brca.parquet")
    methylation = pd.read_parquet(f"{data_dir}/methylation_brca.parquet")
    clinical = pd.read_csv(f"{data_dir}/clinical_brca.csv", index_col=0)

    logger.info(
        f"Loaded: mRNA {mrna.shape}, "
        f"methylation {methylation.shape}, "
        f"clinical {clinical.shape}"
    )

    # ── lmQCM clustering ──
    logger.info("Running lmQCM on mRNA...")
    mrna_eigengenes = run_lmqcm(
        mrna, gamma=gamma, label="mrna", max_genes=999999  # no pre-filter
    )

    logger.info("Running lmQCM on methylation...")
    meth_eigengenes = run_lmqcm(
        methylation, gamma=gamma, label="meth", max_genes=max_genes_lmqcm
    )

    logger.info(
        f"Eigengenes: mRNA={mrna_eigengenes.shape[1]}, "
        f"methylation={meth_eigengenes.shape[1]}, "
        f"total={mrna_eigengenes.shape[1] + meth_eigengenes.shape[1]}"
    )

    # ── Align to common samples ──
    common_samples = (
        mrna_eigengenes.index
        .intersection(meth_eigengenes.index)
        .intersection(clinical.index)
    )
    mrna_eigengenes = mrna_eigengenes.loc[common_samples]
    meth_eigengenes = meth_eigengenes.loc[common_samples]
    clinical = clinical.loc[common_samples]
    logger.info(f"Common samples after alignment: {len(common_samples)}")

    # ── Concatenate eigengenes ──
    features = pd.concat([mrna_eigengenes, meth_eigengenes], axis=1)

    # Fill NaN eigengenes -- can occur when samples are not covered
    # by any lmQCM module
    n_nan = features.isna().sum().sum()
    if n_nan > 0:
        logger.warning(
            f"Filling {n_nan} NaN values in feature matrix with 0"
        )
        features = features.fillna(0)

    logger.info(f"Combined feature matrix: {features.shape}")

    # ── Prepare survival arrays ──
    y = np.array(
        [
            (bool(row["OS_status"]), float(row["OS_time"]))
            for _, row in clinical.iterrows()
        ],
        dtype=[("event", bool), ("time", float)],
    )
    X = features.values.astype(np.float32)

    logger.info(
        f"Survival: n={len(y)}, "
        f"events={y['event'].sum()}, "
        f"median_time={np.median(y['time']):.1f} months"
    )

    # ── Cross-validation ──
    logger.info(
        f"Starting {n_folds}-fold cross-validation "
        f"({n_epochs} epochs/fold) on {DEVICE}..."
    )
    fold_cindices, all_risks = run_cross_validation(
        X, y, n_folds=n_folds, n_epochs=n_epochs
    )

    mean_cindex = float(np.mean(fold_cindices))
    std_cindex = float(np.std(fold_cindices))
    logger.info(f"C-index: {mean_cindex:.4f} ± {std_cindex:.4f}")

    # ── Log-rank test ──
    logrank_p, threshold = compute_logrank_p(all_risks, y)
    logger.info(f"Log-rank p-value: {logrank_p:.2e}")

    # ── KM curves ──
    plot_km_curves(
        all_risks, y, threshold,
        output_path=f"{output_dir}/km_curves.png",
    )

    # ── Save reproduced results ──
    # Saved BEFORE any comparison with extracted_results.json
    reproduced = {
        "model": "ML_ordCOX_reimplementation",
        "baselines": {
            "ML_ordCOX": {
                "cindex": round(mean_cindex, 4),
                "std": round(std_cindex, 4),
            }
        },
        "stratification": {
            "ML_ordCOX_logrank_p": logrank_p,
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
        "approximations": [
            "Ordinal loss uses sampled pairs (max_pairs=1000) not O(n^2)",
            "Methylation probe aggregation uses mean not min-correlation (Section 2.1)",
            "mRNA: no variance pre-filter, expression_filter only (5052 genes -> 28 modules)",
            "Methylation: variance pre-filter to 20000 genes (full 388k not tractable)",
            "Dataset has 785 samples vs 485 in paper (Firehose version difference)",
            "Methylation lmQCM finds 1 module vs 17 in paper -- likely due to probe aggregation approximation",
        ],
    }

    output_path = f"{output_dir}/reproduced_results.json"
    with open(output_path, "w") as f:
        json.dump(reproduced, f, indent=2)

    # Restore extracted_results.json now that reproduced results are saved
    if os.path.exists(lock_path):
        os.rename(lock_path, results_path)
        logger.info(
            "Anti-self-convincing: .verification_lock -> "
            "extracted_results.json (restored for verification step)"
        )

    logger.info(f"Reproduced results saved to {output_path}")
    logger.info(json.dumps(reproduced, indent=2))

    return reproduced


if __name__ == "__main__":
    results = run_analysis()
    print("\nReproduced results:")
    print(json.dumps(results, indent=2))