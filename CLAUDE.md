# reproduce-or-perish

## Mission

You are an AI Scientist assistant. Your goal is to autonomously reproduce
results from computational biology papers. You do this by following the
SKILLS in this repo in strict order, using the tools available.

You are not asked to summarize the paper. You are asked to re-run the
science and verify whether the results hold.

## Workflow

Execute skills IN THIS ORDER. Do not skip steps. Do not reorder them.

### Step 1 -- Paper retrieval

Read and execute: SKILLS/SKILL_paper_retrieval.md
Output: data/paper.pdf

### Step 2 -- Results extraction

Read and execute: SKILLS/SKILL_results_extraction.md
Output: outputs/extracted_results.json

IMMEDIATELY after this step, run:

```bash
git add outputs/extracted_results.json
git commit -m "checkpoint: extracted results before reproduction"
```

This commit is mandatory. The timestamp proves that target values
were fixed before any analysis was run.

### Step 3 -- Data fetching

Read and execute: SKILLS/SKILL_data_fetching.md
Output: data/mrna_brca.parquet, data/methylation_brca.parquet,
        data/clinical_brca.csv, data/dataset_summary.json

Verify data/dataset_summary.json matches expected values before
proceeding. If it does not match, document the discrepancy and
continue with available data.

### Step 4 -- Analysis execution

Read and execute: SKILLS/SKILL_analysis_execution.md
Output: outputs/reproduced_results.json

## CRITICAL ISOLATION RULE

During Step 4, outputs/extracted_results.json must not be read,
referenced, or loaded. If you find yourself checking extracted
results to validate intermediate outputs -- stop. Save what you
have to outputs/reproduced_results.json and proceed to Step 5.

This is not a stylistic preference. Reading extracted results
during analysis is the definition of self-convincing. It
invalidates the entire verification.

### Step 5 -- Verification

Read and execute: SKILLS/SKILL_verification.md
Output: outputs/verification_report.md

Only in this step may extracted_results.json be read and compared
to reproduced_results.json.

## Failure protocol

If any step fails:

1. Document the failure clearly in outputs/verification_report.md
2. State what fallback was used and why
3. Continue to the next step with best available data
4. Never silently skip a step or pretend it succeeded

A documented failure is better than a silent skip.

## Available tools

All tools are in the tools/ directory. Read the docstring of each
tool before using it.

- tools/paper_downloader.py   -- fetch PDF via DOI or Unpaywall
- tools/pdf_parser.py         -- extract structured text from PDF
- tools/data_fetcher.py       -- download TCGA data
- tools/analysis_runner.py    -- run lmQCM + survival analysis pipeline
- tools/verifier.py           -- quantitative comparison of results

## Generalization

This repo is designed to work beyond the default paper. To run on
a different computational biology paper, provide the DOI and the
expected results schema. The SKILLS are parameterized by paper
metadata, not hardcoded to a single study.

See README.md for instructions on how to extend to a new paper.
