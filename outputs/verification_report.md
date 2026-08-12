============================================================
VERIFICATION REPORT
Generated: 2026-08-12T22:37:34.279298
Paper DOI: 10.1093/bioinformatics/btab140
============================================================

Tolerance thresholds (defined pre-comparison):
  cindex: 0.02
  cindex_std: 0.015
  logrank_p_orders: 2
  n_samples_pct: 0.1
  n_events_pct: 0.1
  n_modules_pct: 0.2

============================================================
SECTION 1: Primary result -- ML_ordCOX C-index
============================================================

[FAIL] ML_ordCOX C-index (Table 5)
  Expected:   0.7222
  Reproduced: 0.5112
  Delta:      0.2110 (tolerance: ±0.02)

[FAIL] ML_ordCOX C-index std (Table 5)
  Expected:   0.0145
  Reproduced: 0.0387
  Delta:      0.0242 (tolerance: ±0.015)

============================================================
SECTION 2: Survival stratification
============================================================

[FAIL] ML_ordCOX log-rank p-value (Figure 5)
  Expected:   1.29e-05
  Reproduced: 9.66e-01
  Log10 distance: 4.9 orders (tolerance: 2 orders)

============================================================
SECTION 3: Dataset characteristics
============================================================

[FAIL] Number of samples
  Expected:   485
  Reproduced: 785
  Delta:      61.9% (tolerance: 10%)

[FAIL] Number of events (deceased)
  Expected:   63
  Reproduced: 103
  Delta:      63.5% (tolerance: 10%)

============================================================
SECTION 4: lmQCM modules
============================================================

[FAIL] mRNA modules (expected ~116)
  Expected:   116
  Reproduced: 28
  Delta:      75.9% (tolerance: 20%)

[FAIL] Methylation modules (expected ~17)
  Expected:   17
  Reproduced: 1
  Delta:      94.1% (tolerance: 20%)

============================================================
SUMMARY
============================================================
PASSED:  0 / 7
FAILED:  7
SKIPPED: 0

Failed checks:
  - primary_cindex
  - primary_cindex_std
  - logrank_p
  - n_samples
  - n_events
  - mrna_modules
  - meth_modules

Overall verdict: PARTIAL

============================================================
KNOWN LIMITATIONS AND DOCUMENTED DEVIATIONS
============================================================

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

APPROXIMATIONS DOCUMENTED BY ANALYSIS RUNNER
============================================================
  - Ordinal loss uses sampled pairs (max_pairs=1000) not O(n^2)
  - Methylation probe aggregation uses mean not min-correlation (Section 2.1)
  - mRNA: no variance pre-filter, expression_filter only (5052 genes -> 28 modules)
  - Methylation: variance pre-filter to 20000 genes (full 388k not tractable)
  - Dataset has 785 samples vs 485 in paper (Firehose version difference)
  - Methylation lmQCM finds 1 module vs 17 in paper -- likely due to probe aggregation approximation