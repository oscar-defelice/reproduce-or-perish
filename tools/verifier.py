# tools/verifier.py
"""
Quantitative verification of reproduced results against extracted paper values.

Compares outputs/reproduced_results.json against outputs/extracted_results.json
and produces a structured verification report.

This is the ONLY tool that reads both files simultaneously.
It must be run AFTER analysis_runner.py has completed and restored
extracted_results.json from .verification_lock.
"""

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_results(output_dir: str = "outputs") -> tuple[dict, dict]:
    """
    Load reproduced and extracted results from output directory.

    Parameters
    ----------
    output_dir : str, optional
        Directory containing result files, by default "outputs".

    Returns
    -------
    tuple[dict, dict]
        (reproduced_results, extracted_results)

    Raises
    ------
    FileNotFoundError
        If either result file is missing.
    RuntimeError
        If .verification_lock exists -- analysis is still running.
    """
    lock_path = os.path.join(output_dir, ".verification_lock")
    if os.path.exists(lock_path):
        raise RuntimeError(
            ".verification_lock exists -- analysis_runner.py may still be "
            "running. Wait for it to complete before running verifier."
        )

    reproduced_path = os.path.join(output_dir, "reproduced_results.json")
    extracted_path = os.path.join(output_dir, "extracted_results.json")

    if not os.path.exists(reproduced_path):
        raise FileNotFoundError(
            f"reproduced_results.json not found at {reproduced_path}. "
            "Run analysis_runner.py first."
        )

    if not os.path.exists(extracted_path):
        raise FileNotFoundError(
            f"extracted_results.json not found at {extracted_path}. "
            "Run pdf_parser.py first."
        )

    with open(reproduced_path) as f:
        reproduced = json.load(f)

    with open(extracted_path) as f:
        extracted = json.load(f)

    return reproduced, extracted


# ── Tolerances ────────────────────────────────────────────────────────────────

TOLERANCES = {
    "cindex":           0.02,   # ±0.02 absolute
    "cindex_std":       0.015,  # ±0.015 absolute
    "logrank_p_orders": 2,      # within 2 orders of magnitude
    "n_samples_pct":    0.10,   # ±10% sample count
    "n_events_pct":     0.10,   # ±10% event count
    "n_modules_pct":    0.20,   # ±20% module count
}


# ── Comparison functions ──────────────────────────────────────────────────────

def compare_cindex(
    label: str,
    reproduced_val: float,
    expected_val: float,
    tolerance: float,
    report: list[str],
) -> bool:
    """
    Compare two C-index values within an absolute tolerance.

    Parameters
    ----------
    label : str
        Human-readable label for this comparison.
    reproduced_val : float
        Value from reproduced_results.json.
    expected_val : float
        Value from extracted_results.json.
    tolerance : float
        Maximum acceptable absolute difference.
    report : list[str]
        Report lines to append to.

    Returns
    -------
    bool
        True if within tolerance, False otherwise.
    """
    delta = abs(reproduced_val - expected_val)
    passed = delta <= tolerance
    status = "PASS" if passed else "FAIL"
    report.append(f"\n[{status}] {label}")
    report.append(f"  Expected:   {expected_val:.4f}")
    report.append(f"  Reproduced: {reproduced_val:.4f}")
    report.append(f"  Delta:      {delta:.4f} (tolerance: ±{tolerance})")
    return passed


def compare_logrank(
    label: str,
    reproduced_p: float,
    expected_p: float,
    max_orders: float,
    report: list[str],
) -> bool:
    """
    Compare two log-rank p-values within a log10 order-of-magnitude tolerance.

    Parameters
    ----------
    label : str
        Human-readable label for this comparison.
    reproduced_p : float
        Reproduced p-value.
    expected_p : float
        Expected p-value from paper.
    max_orders : float
        Maximum acceptable difference in log10 orders of magnitude.
    report : list[str]
        Report lines to append to.

    Returns
    -------
    bool or None
        True if within tolerance, False if not, None if comparison skipped.
    """
    if expected_p <= 0 or reproduced_p <= 0:
        report.append(f"\n[SKIP] {label} -- p-value zero or negative")
        return None

    orders = abs(math.log10(reproduced_p) - math.log10(expected_p))
    passed = orders <= max_orders
    status = "PASS" if passed else "FAIL"
    report.append(f"\n[{status}] {label}")
    report.append(f"  Expected:   {expected_p:.2e}")
    report.append(f"  Reproduced: {reproduced_p:.2e}")
    report.append(
        f"  Log10 distance: {orders:.1f} orders "
        f"(tolerance: {max_orders} orders)"
    )
    return passed


def compare_count(
    label: str,
    reproduced_val: int,
    expected_val: int,
    tolerance_pct: float,
    report: list[str],
) -> bool:
    """
    Compare two integer counts within a percentage tolerance.

    Parameters
    ----------
    label : str
        Human-readable label for this comparison.
    reproduced_val : int
        Reproduced count.
    expected_val : int
        Expected count from paper.
    tolerance_pct : float
        Maximum acceptable fractional difference (e.g. 0.05 = 5%).
    report : list[str]
        Report lines to append to.

    Returns
    -------
    bool
        True if within tolerance, False otherwise.
    """
    if expected_val == 0:
        report.append(f"\n[SKIP] {label} -- expected value is zero")
        return None

    delta_pct = abs(reproduced_val - expected_val) / expected_val
    passed = delta_pct <= tolerance_pct
    status = "PASS" if passed else "FAIL"
    report.append(f"\n[{status}] {label}")
    report.append(f"  Expected:   {expected_val}")
    report.append(f"  Reproduced: {reproduced_val}")
    report.append(
        f"  Delta:      {delta_pct*100:.1f}% "
        f"(tolerance: {tolerance_pct*100:.0f}%)"
    )
    return passed


# ── Main verification ─────────────────────────────────────────────────────────

def run_verification(
    output_dir: str = "outputs",
) -> dict:
    """
    Run quantitative verification of reproduced results.

    Parameters
    ----------
    output_dir : str, optional
        Directory containing result files, by default "outputs".

    Returns
    -------
    dict
        Verification summary with overall verdict and per-check results.

    Notes
    -----
    Tolerances are defined at the top of this module before any comparison.
    They are never adjusted post-hoc to make results pass.

    Saves outputs/verification_report.md and outputs/verification_summary.json.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    reproduced, extracted = load_results(output_dir)

    report = []
    results = {}

    report.append("=" * 60)
    report.append("VERIFICATION REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append(f"Paper DOI: {extracted['paper_doi']}")
    report.append("=" * 60)

    report.append("\nTolerance thresholds (defined pre-comparison):")
    for k, v in TOLERANCES.items():
        report.append(f"  {k}: {v}")

    # ── Section 1: Primary C-index ──
    report.append("\n" + "=" * 60)
    report.append("SECTION 1: Primary result -- ML_ordCOX C-index")
    report.append("=" * 60)

    results["primary_cindex"] = compare_cindex(
        label="ML_ordCOX C-index (Table 5)",
        reproduced_val=reproduced["baselines"]["ML_ordCOX"]["cindex"],
        expected_val=extracted["baselines"]["ML_ordCOX"]["cindex"],
        tolerance=TOLERANCES["cindex"],
        report=report,
    )

    results["primary_cindex_std"] = compare_cindex(
        label="ML_ordCOX C-index std (Table 5)",
        reproduced_val=reproduced["baselines"]["ML_ordCOX"]["std"],
        expected_val=extracted["baselines"]["ML_ordCOX"]["std"],
        tolerance=TOLERANCES["cindex_std"],
        report=report,
    )

    # ── Section 2: Log-rank p-value ──
    report.append("\n" + "=" * 60)
    report.append("SECTION 2: Survival stratification")
    report.append("=" * 60)

    results["logrank_p"] = compare_logrank(
        label="ML_ordCOX log-rank p-value (Figure 5)",
        reproduced_p=reproduced["stratification"]["ML_ordCOX_logrank_p"],
        expected_p=extracted["stratification"]["ML_ordCOX_logrank_p"],
        max_orders=TOLERANCES["logrank_p_orders"],
        report=report,
    )

    # ── Section 3: Dataset characteristics ──
    report.append("\n" + "=" * 60)
    report.append("SECTION 3: Dataset characteristics")
    report.append("=" * 60)

    results["n_samples"] = compare_count(
        label="Number of samples",
        reproduced_val=reproduced["dataset"]["n_samples"],
        expected_val=int(extracted["dataset"]["n_samples"]),
        tolerance_pct=TOLERANCES["n_samples_pct"],
        report=report,
    )

    results["n_events"] = compare_count(
        label="Number of events (deceased)",
        reproduced_val=reproduced["dataset"]["n_events"],
        expected_val=int(extracted["dataset"]["n_deceased"]),
        tolerance_pct=TOLERANCES["n_events_pct"],
        report=report,
    )

    # ── Section 4: lmQCM modules ──
    report.append("\n" + "=" * 60)
    report.append("SECTION 4: lmQCM modules")
    report.append("=" * 60)

    results["mrna_modules"] = compare_count(
        label="mRNA modules (expected ~116)",
        reproduced_val=reproduced["lmqcm_modules"]["mrna"],
        expected_val=int(extracted["lmqcm_params"]["n_mrna_modules"]),
        tolerance_pct=TOLERANCES["n_modules_pct"],
        report=report,
    )

    results["meth_modules"] = compare_count(
        label="Methylation modules (expected ~17)",
        reproduced_val=reproduced["lmqcm_modules"]["methylation"],
        expected_val=int(extracted["lmqcm_params"]["n_methylation_modules"]),
        tolerance_pct=TOLERANCES["n_modules_pct"],
        report=report,
    )

    # ── Summary ──
    report.append("\n" + "=" * 60)
    report.append("SUMMARY")
    report.append("=" * 60)

    passed = [k for k, v in results.items() if v is True]
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]

    report.append(f"PASSED:  {len(passed)} / {len(results)}")
    report.append(f"FAILED:  {len(failed)}")
    report.append(f"SKIPPED: {len(skipped)}")

    if failed:
        report.append("\nFailed checks:")
        for f in failed:
            report.append(f"  - {f}")

    overall = len(failed) == 0
    verdict = "REPRODUCED" if overall else "PARTIAL"
    report.append(f"\nOverall verdict: {verdict}")

    # ── Known limitations ──
    report.append("\n" + "=" * 60)
    report.append("KNOWN LIMITATIONS AND DOCUMENTED DEVIATIONS")
    report.append("=" * 60)
    report.append("""
1. METHYLATION PROBE SELECTION (primary source of C-index gap)
   Paper: selects probe with minimal correlation to mRNA expression
   per gene (Section 2.1). This requires aligned mRNA-methylation
   data and must be done within each CV fold to avoid leakage.
   Implementation: mean aggregation across all probes per gene.
   Impact: lmQCM finds 1 methylation module vs 17 in paper.
   This is the main reason C-index does not reproduce.

2. DATASET SIZE
   Paper: 485 samples (Firehose 2016 batch).
   Implementation: 785 samples (current Firehose release).
   Event rate is consistent (13.1% vs 13.0%) confirming data quality.
   The original 485-sample list is not publicly available.

3. ORDINAL LOSS APPROXIMATION
   Paper: full O(n^2) pairwise ordinal loss.
   Implementation: sampled pairs (max_pairs=1000).
   This reduces computational cost at the expense of gradient quality.

4. lmQCM VARIANCE PRE-FILTER
   mRNA: no pre-filter, expression_filter only (5052 genes -> 28 modules
   vs 116 in paper).
   Methylation: pre-filter to 20000 genes (full 388k not tractable).
   The paper runs lmQCM on all expression-filtered genes.

5. biLSTM HYPERPARAMETERS
   Paper does not specify hidden_dim, n_layers, dropout.
   Implementation uses hidden_dim=64, n_layers=2, dropout=0.3.

6. BIOLEARNS lmQCM BUG
   Fixed in place: ValueError on empty neighborWeights sequence.
   Patch applied to biolearns/coexpression/_lmQCM.py line 151.
""")

    # ── Approximations from analysis_runner ──
    if "approximations" in reproduced:
        report.append("APPROXIMATIONS DOCUMENTED BY ANALYSIS RUNNER")
        report.append("=" * 60)
        for a in reproduced["approximations"]:
            report.append(f"  - {a}")

    # ── Save report ──
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "verification_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Verification report saved to {report_path}")

    # ── Save machine-readable summary ──
    summary = {
        "timestamp": datetime.now().isoformat(),
        "overall": verdict,
        "checks": {k: str(v) for k, v in results.items()},
        "n_passed": len(passed),
        "n_failed": len(failed),
        "n_skipped": len(skipped),
        "reproduced_cindex": reproduced["baselines"]["ML_ordCOX"]["cindex"],
        "expected_cindex": extracted["baselines"]["ML_ordCOX"]["cindex"],
        "cindex_gap": round(
            extracted["baselines"]["ML_ordCOX"]["cindex"] -
            reproduced["baselines"]["ML_ordCOX"]["cindex"],
            4
        ),
    }

    summary_path = os.path.join(output_dir, "verification_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Verification summary saved to {summary_path}")

    print(report_text)
    return summary


if __name__ == "__main__":
    summary = run_verification()
    print(f"\nOverall: {summary['overall']}")
    print(
        f"C-index gap: {summary['cindex_gap']:.4f} "
        f"({summary['reproduced_cindex']:.4f} reproduced vs "
        f"{summary['expected_cindex']:.4f} expected)"
    )