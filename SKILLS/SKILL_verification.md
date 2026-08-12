# SKILL: Verification

# Version: 1.0

# Generalized: compares reproduced results against extracted ground truth

## Purpose

Quantitatively compare outputs/reproduced_results.json against
outputs/extracted_results.json and produce a structured verification
report. This is the ONLY skill that reads extracted_results.json.

## CRITICAL

This skill must be run AFTER analysis_execution completes.
The .verification_lock file is restored to extracted_results.json
before this skill runs. Do not read extracted_results.json earlier.

## Input

- outputs/reproduced_results.json
- outputs/extracted_results.json
- outputs/km_curves.png

## Setup

```python
import json
import os
from datetime import datetime

with open("outputs/reproduced_results.json") as f:
    reproduced = json.load(f)

with open("outputs/extracted_results.json") as f:
    expected = json.load(f)

report_lines = []

def log(line=""):
    print(line)
    report_lines.append(line)

log("=" * 60)
log("VERIFICATION REPORT")
log(f"Generated: {datetime.now().isoformat()}")
log(f"Paper DOI: {expected['paper_doi']}")
log("=" * 60)
```

## Tolerance thresholds

Defined here, before any comparison.
Never adjusted post-hoc to make results pass.

```python
TOLERANCES = {
    "cindex": 0.02,          # ±0.02 absolute
    "cindex_std": 0.015,     # ±0.015 absolute
    "logrank_p_order": 2,    # within 2 orders of magnitude
}

log("\nTolerance thresholds (defined pre-comparison):")
for k, v in TOLERANCES.items():
    log(f"  {k}: {v}")
```

## Comparison functions

```python
def compare_cindex(label, reproduced_val, expected_val, tolerance):
    delta = abs(reproduced_val - expected_val)
    passed = delta <= tolerance
    status = "PASS" if passed else "FAIL"
    log(f"\n[{status}] {label}")
    log(f"  Expected:   {expected_val:.4f}")
    log(f"  Reproduced: {reproduced_val:.4f}")
    log(f"  Delta:      {delta:.4f} (tolerance: {tolerance})")
    return passed

def compare_logrank(label, reproduced_p, expected_p, max_orders):
    import math
    if expected_p <= 0 or reproduced_p <= 0:
        log(f"\n[SKIP] {label} -- p-value is zero or negative")
        return None
    orders = abs(math.log10(reproduced_p) - math.log10(expected_p))
    passed = orders <= max_orders
    status = "PASS" if passed else "FAIL"
    log(f"\n[{status}] {label}")
    log(f"  Expected:   {expected_p:.2e}")
    log(f"  Reproduced: {reproduced_p:.2e}")
    log(f"  Orders of magnitude apart: {orders:.1f} (tolerance: {max_orders})")
    return passed

def compare_integer(label, reproduced_val, expected_val, tolerance_pct=0.05):
    delta_pct = abs(reproduced_val - expected_val) / expected_val
    passed = delta_pct <= tolerance_pct
    status = "PASS" if passed else "FAIL"
    log(f"\n[{status}] {label}")
    log(f"  Expected:   {expected_val}")
    log(f"  Reproduced: {reproduced_val}")
    log(f"  Delta:      {delta_pct*100:.1f}% (tolerance: {tolerance_pct*100:.0f}%)")
    return passed
```

## Run comparisons

```python
results = {}

log("\n" + "=" * 60)
log("SECTION 1: Primary result -- ML_ordCOX C-index")
log("=" * 60)

results["primary_cindex"] = compare_cindex(
    label="ML_ordCOX C-index (Table 5)",
    reproduced_val=reproduced["baselines"]["ML_ordCOX"]["cindex"],
    expected_val=expected["baselines"]["ML_ordCOX"]["cindex"],
    tolerance=TOLERANCES["cindex"],
)

results["primary_cindex_std"] = compare_cindex(
    label="ML_ordCOX C-index std (Table 5)",
    reproduced_val=reproduced["baselines"]["ML_ordCOX"]["std"],
    expected_val=expected["baselines"]["ML_ordCOX"]["std"],
    tolerance=TOLERANCES["cindex_std"],
)

log("\n" + "=" * 60)
log("SECTION 2: Survival stratification")
log("=" * 60)

results["logrank_p"] = compare_logrank(
    label="ML_ordCOX log-rank p-value (Figure 5)",
    reproduced_p=reproduced["stratification"]["ML_ordCOX_logrank_p"],
    expected_p=expected["stratification"]["ML_ordCOX_logrank_p"],
    max_orders=TOLERANCES["logrank_p_order"],
)

log("\n" + "=" * 60)
log("SECTION 3: Dataset characteristics")
log("=" * 60)

results["n_samples"] = compare_integer(
    label="Number of samples",
    reproduced_val=reproduced["dataset"]["n_samples"],
    expected_val=expected["dataset"]["n_samples"],
)

results["n_events"] = compare_integer(
    label="Number of events (deceased)",
    reproduced_val=reproduced["dataset"]["n_events"],
    expected_val=expected["dataset"]["n_deceased"],
)

log("\n" + "=" * 60)
log("SECTION 4: lmQCM modules")
log("=" * 60)

results["mrna_modules"] = compare_integer(
    label="mRNA modules (expected ~116)",
    reproduced_val=reproduced["lmqcm_modules"]["mrna"],
    expected_val=expected["lmqcm_params"]["n_mrna_modules"],
    tolerance_pct=0.20,  # lmQCM is stochastic -- wider tolerance
)

results["meth_modules"] = compare_integer(
    label="Methylation modules (expected ~17)",
    reproduced_val=reproduced["lmqcm_modules"]["methylation"],
    expected_val=expected["lmqcm_params"]["n_methylation_modules"],
    tolerance_pct=0.20,
)
```

## Summary

```python
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)

passed = [k for k, v in results.items() if v is True]
failed = [k for k, v in results.items() if v is False]
skipped = [k for k, v in results.items() if v is None]

log(f"PASSED:  {len(passed)} / {len(results)}")
log(f"FAILED:  {len(failed)}")
log(f"SKIPPED: {len(skipped)}")

if failed:
    log("\nFailed checks:")
    for f in failed:
        log(f"  - {f}")

if skipped:
    log("\nSkipped checks:")
    for s in skipped:
        log(f"  - {s}")

overall = len(failed) == 0
log(f"\nOverall verdict: {'REPRODUCED' if overall else 'PARTIAL'}")

log("\n" + "=" * 60)
log("KNOWN LIMITATIONS AND DISCREPANCIES")
log("=" * 60)
log("""
1. TCGA data version: paper uses Broad Firehose 2016 release.
   Current GDC data may differ slightly in sample count and
   gene coverage. Discrepancies up to 5% are expected.

2. lmQCM stochasticity: module count depends on data version
   and random seed. Exact replication of 116+17 modules is
   not guaranteed. We use ±20% tolerance.

3. Ordinal loss approximation: the O(n²) pairwise ordinal loss
   may be approximated with sampled pairs if training is slow.
   This is documented in SKILL_analysis_execution.md.

4. C-index variance: 10-fold CV with random splits will produce
   different folds than the original paper. C-index within ±0.02
   of the reported value is considered a successful reproduction.

5. Baseline comparisons (RSF, LASSO, DeepSurv, MLP, MTLSA):
   not reproduced in this proof of concept. Only ML_ordCOX
   is reimplemented. Full baseline comparison is left as
   future work.
""")
```

## Save report

```python
report_text = "\n".join(report_lines)
with open("outputs/verification_report.md", "w") as f:
    f.write(report_text)

print(f"\nVerification report saved to outputs/verification_report.md")

# Save machine-readable summary
summary = {
    "timestamp": datetime.now().isoformat(),
    "overall": "REPRODUCED" if overall else "PARTIAL",
    "checks": {k: str(v) for k, v in results.items()},
    "n_passed": len(passed),
    "n_failed": len(failed),
    "n_skipped": len(skipped),
}

with open("outputs/verification_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
```

## Output

- `outputs/verification_report.md` -- human-readable report
- `outputs/verification_summary.json` -- machine-readable summary

## Generalization

To use with a different paper:

1. Add comparison blocks for the paper's specific metrics
2. Adjust TOLERANCES based on the paper's methodology
   (stochastic methods need wider tolerances than deterministic ones)
3. Update KNOWN LIMITATIONS section with paper-specific caveats
The comparison functions (compare_cindex, compare_logrank,
compare_integer) are fully generic and reusable.
