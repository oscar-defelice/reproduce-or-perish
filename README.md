# reproduce-or-perish

> *An agentic AI system that actually reads the Methods section.*

A proof-of-concept AI Scientist that autonomously downloads a computational
biology paper, extracts its methods and results, reproduces the analysis,
and verifies whether the results hold -- without ever reading the target
values during execution.

---

## What it does

Given a DOI, the system:

1. **Downloads** the paper PDF via a four-strategy fallback chain
   (Unpaywall, EuropePMC, Semantic Scholar, local fallback)
2. **Extracts** numerical results and methods using pymupdf + targeted regex
3. **Downloads** the exact dataset used in the paper (TCGA Firehose 2016)
4. **Reproduces** the full analysis pipeline (lmQCM + biLSTM ordinal Cox)
5. **Verifies** reproduced results against extracted values quantitatively

The system is designed around an **anti-self-convincing protocol**: extracted
results are renamed to `.verification_lock` before analysis begins and restored
only after results are saved. The git timestamp of the extraction checkpoint
proves that target values were fixed before any analysis was run.

---

## Paper reproduced

**Integrative survival analysis of breast cancer with gene expression and
DNA methylation data**
Bichindaritz et al., *Bioinformatics*, 2021
DOI: `10.1093/bioinformatics/btab140`

---

## Architecture

```bash
reproduce-or-perish/
├── CLAUDE.md # Claude Code orchestration instructions
├── paper_config.yaml # Single configuration entry point
├── requirements.txt # Verified on Python 3.10 macOS ARM
│
├── SKILLS/
│ ├── SKILL_paper_retrieval.md
│ ├── SKILL_results_extraction.md
│ ├── SKILL_data_fetching.md
│ ├── SKILL_analysis_execution.md
│ └── SKILL_verification.md
│
└── tools/
├── apply_patch.py # Patches biolearns lmQCM bug (run first)
├── paper_downloader.py # PDF retrieval, four-strategy fallback
├── pdf_parser.py # Result + method extraction (pymupdf + regex)
├── data_fetcher.py # Firehose 2016 download (Polars methylation parser)
├── analysis_runner.py # lmQCM + biLSTM Cox, MPS-accelerated
└── verifier.py # Quantitative comparison with tolerances
```

The system is designed to be orchestrated by **Claude Code**, which reads
`CLAUDE.md` and the SKILL files to execute the full pipeline autonomously.

---

## How to run

### Setup

```bash
git clone https://github.com/oscar-defelice/reproduce-or-perish
cd reproduce-or-perish

conda create -n pop python=3.10
conda activate pop
pip install -r requirements.txt
```

### Apply required patch

biolearns has a known bug in lmQCM (ValueError on empty neighborWeights
sequence). Apply the patch before running:

```bash
python tools/apply_patch.py
```

Safe to run multiple times -- detects if already applied.

### Run the full pipeline

```bash
# Step 1: download paper
python tools/paper_downloader.py

# Step 2: extract results and commit checkpoint
python tools/pdf_parser.py
git add outputs/extracted_results.json outputs/methods.md
git commit -m "checkpoint: extracted results before reproduction"

# Step 3: download data (~3GB, ~5 minutes)
python tools/data_fetcher.py

# Step 4: reproduce analysis (~4 minutes on Apple M-series)
python tools/analysis_runner.py

# Step 5: verify
python tools/verifier.py
```

### Or via Claude Code

```bash
claude
```

Claude Code reads `CLAUDE.md` and executes the full pipeline autonomously,
following the anti-self-convincing protocol automatically.

### Running with Claude Code -- full instructions

Claude Code orchestrates the entire pipeline autonomously by reading
`CLAUDE.md` and the SKILL files.

#### Prerequisites

```bash
cd reproduce-or-perish
conda activate pop
```

#### Launch Claude Code

```bash
claude
```

#### Initial prompt

Once Claude Code opens, paste this prompt exactly:

```bash
Follow the instructions in CLAUDE.md to reproduce the paper with DOI
10.1093/bioinformatics/btab140. Start from Step 1. The conda environment
"pop" is already active and all dependencies are installed. The biolearns
patch has already been applied. Execute each step in order and stop after
each git commit to confirm before proceeding.
````

#### What Claude Code will do

1. Run `tools/paper_downloader.py` -- downloads the PDF
2. Run `tools/pdf_parser.py` -- extracts results and methods
3. **Pause and ask you to confirm the git checkpoint commit**
   (this is mandatory for the anti-self-convincing protocol)
4. Run `tools/data_fetcher.py` -- downloads TCGA data (~5 minutes)
5. Run `tools/analysis_runner.py` -- reproduces the analysis (~4 minutes)
6. Run `tools/verifier.py` -- generates the verification report

#### The checkpoint commit

At Step 3, Claude Code will ask you to run:

```bash
git add outputs/extracted_results.json outputs/methods.md
git commit -m "checkpoint: extracted results before reproduction"
```

This commit is not optional -- it is the structural proof that target
values were fixed before any analysis was run. The git timestamp is
the anti-self-convincing guarantee.

#### Notes

- Claude Code cannot make git commits autonomously -- you approve each one
- The full pipeline takes ~10 minutes on Apple M-series
- If any step fails, Claude Code will document the failure in
  `outputs/verification_report.md` and continue with the next step
- Re-running is safe -- data fetching skips existing files automatically

---

## Run a different paper

Edit `paper_config.yaml`:

```yaml
paper:
  doi: "your-doi-here"
  title: "Paper title"
  email_unpaywall: "your@email.com"
```

The retrieval, parsing, and verification SKILLs are fully generic.
Only `SKILL_analysis_execution.md` is paper-specific -- add a new one
for each paper category you want to support.

---

## Verification results

```bash
Overall verdict: PARTIAL
C-index gap: 0.2110 (0.5112 reproduced vs 0.7222 expected)
```

| Check | Status | Note |
| --- | --- | --- |
| ML_ordCOX C-index | FAIL | 0.51 vs 0.72 -- see limitations |
| C-index std | FAIL | 0.039 vs 0.015 |
| Log-rank p-value | FAIL | 0.97 vs 1.29e-05 |
| Sample count | FAIL | 785 vs 485 -- Firehose version |
| Event count | FAIL | 103 vs 63 -- proportional (13.1% vs 13.0%) |
| mRNA modules | FAIL | 28 vs 116 |
| Methylation modules | FAIL | 1 vs 17 -- root cause |

All failures are documented with root causes in
`outputs/verification_report.md`.

---

## Known limitations and root causes

### Primary gap: methylation probe selection

The paper selects one probe per gene by **minimal correlation to mRNA
expression** (Section 2.1). This requires:

- Aligned mRNA-methylation data before clustering
- Probe selection within each CV fold to avoid leakage

This implementation uses **mean aggregation** across all probes per gene --
a simpler approximation that loses the regulatory signal making methylation
informative for survival. As a direct consequence, lmQCM finds 1 methylation
module instead of 17, leaving the model with 29 features instead of 133.

### Dataset version

The paper uses the Firehose 2016-01-28 release with 485 samples. The current
Firehose release contains 785 samples with matched mRNA and methylation. The
event rate is consistent (13.1% vs 13.0%), confirming data quality. The
original 485-sample list is not publicly available.

### What would be needed for full reproduction

| Fix | Estimated effort |
| --- | --- |
| Correct probe selection (min-corr to mRNA, per CV fold) | 2-3h |
| Original 485-sample list | difficult, possibly impossible |
| biLSTM hyperparameter search | 2-4h |
| Full O(n²) ordinal loss | 1h |

---

## Anti-self-convincing architecture

Three structural guarantees -- not prompt-level instructions:

**1. Filesystem isolation**
`extracted_results.json` is renamed to `.verification_lock` during analysis
execution. The agent cannot read it even if it tries.

**2. Session isolation**
Extraction and execution are designed to run in separate Claude Code sessions
with no shared context window.

**3. Git checkpoint**
Extracted results are committed before analysis begins. The commit timestamp
proves independence between extraction and reproduction.

---

## Design decisions

**Why Firehose instead of GDC API?**
The paper explicitly uses Firehose 2016-01-28. Current GDC data uses
probe-level SESAME methylation (not gene-level) and contains more samples.
Firehose is still accessible and provides the closest match to the paper.

**Why Polars for methylation parsing?**
The Firehose HM450 file has 3541 columns (1180 samples × 3 columns each).
Polars multi-threaded CSV reader handles this in ~30 seconds vs pandas
which would take hours due to the wide format.

**Why MPS acceleration?**
PyTorch MPS backend on Apple Silicon provides ~3× speedup over CPU for
the biLSTM training loop. The full 10-fold CV completes in ~90 seconds.

**Why not use the original ML_ordCOX repo?**
The original repository (bhioswego/ML_ordCOX) has three reproducibility
blockers: datasets on Baidu Pan (inaccessible from Europe), Python 3.6
with unmaintained Keras dependencies, and R/Python hybrid runtime.
This implementation reimplements the pipeline from the paper methods section
using maintained libraries only.

---

## Generalisation

The infrastructure layer is fully generic:

- `SKILL_paper_retrieval.md` -- works for any open-access paper
- `SKILL_results_extraction.md` -- parameterised by paper structure
- `SKILL_data_fetching.md` -- generalises to any TCGA cohort via `paper_config.yaml`
- `SKILL_verification.md` -- tolerances are configurable per paper

To add a new paper: write a new `SKILL_analysis_execution.md` following
the same input/output contract. The verification and infrastructure SKILLs
never change.

---

## Requirements

- Python 3.10
- macOS (Apple Silicon recommended for MPS acceleration)
- ~5GB disk space for raw data
- Internet access for Firehose and paper download
- Claude Code (optional, for autonomous orchestration)
