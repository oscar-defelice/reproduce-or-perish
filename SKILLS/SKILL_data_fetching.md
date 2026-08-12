# SKILL: Data Fetching

# Version: 1.0

# Generalized: reads dataset config from paper_config.yaml

## Purpose

Download TCGA-BRCA mRNA expression, DNA methylation, and clinical
data programmatically. Merge and filter to match paper Section 2.1.
No manual downloads, no external cloud storage.

## Input

- paper_config.yaml -- contains data.tcga_project
- outputs/extracted_results.json -- contains expected sample counts

## Setup

```python
import yaml
import json
import os
import pandas as pd
import numpy as np

with open("paper_config.yaml") as f:
    config = yaml.safe_load(f)

with open("outputs/extracted_results.json") as f:
    expected = json.load(f)

project = config["data"]["tcga_project"]  # "TCGA-BRCA"
os.makedirs("data", exist_ok=True)
```

## Strategy 1: biolearns (preferred)

biolearns has TCGA data access built in.
Fast, no authentication required.

```python
def fetch_via_biolearns(project):
    from biolearns.dataset import TCGA

    cohort_name = project.replace("TCGA-", "")  # BRCA
    brca = TCGA(cohort_name)

    mrna = brca.mRNAseq          # DataFrame: genes x samples
    clinical = brca.clinical      # DataFrame: samples x features

    # Check if methylation is available
    methylation = None
    if hasattr(brca, "methylation"):
        methylation = brca.methylation
    elif hasattr(brca, "DNAmethylation"):
        methylation = brca.DNAmethylation

    return mrna, methylation, clinical
```

## Strategy 2: GDC API (fallback)

Used when biolearns does not have methylation data.
Fully programmatic, no account required.

```python
import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA  = "https://api.gdc.cancer.gov/data"

def build_gdc_filter(project, data_type):
    return {
        "op": "and",
        "content": [
            {"op": "=", "content": {
                "field": "cases.project.project_id",
                "value": project}},
            {"op": "=", "content": {
                "field": "data_type",
                "value": data_type}},
            {"op": "=", "content": {
                "field": "access",
                "value": "open"}},
        ]
    }

def get_gdc_files(project, data_type, max_files=1200):
    params = {
        "filters": json.dumps(build_gdc_filter(project, data_type)),
        "fields": "file_id,file_name,cases.submitter_id",
        "format": "json",
        "size": max_files,
    }
    response = requests.get(GDC_FILES, params=params, timeout=30)
    return response.json()["data"]["hits"]

def download_gdc_files(file_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    payload = {"ids": file_ids}
    response = requests.post(
        GDC_DATA,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
        stream=True,
    )
    # GDC returns a tar archive for multiple files
    import tarfile, io
    with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
        tar.extractall(output_dir)
    print(f"Downloaded {len(file_ids)} files to {output_dir}")
```

## Execution -- try biolearns first

```python
mrna, methylation, clinical = None, None, None

print("Trying biolearns...")
try:
    mrna, methylation, clinical = fetch_via_biolearns(project)
    print(f"biolearns: mRNA shape {mrna.shape}")
    if methylation is not None:
        print(f"biolearns: methylation shape {methylation.shape}")
    else:
        print("biolearns: methylation not available, will use GDC fallback")
except Exception as e:
    print(f"biolearns failed: {e}")

# Fallback to GDC for methylation if missing
if methylation is None:
    print("Fetching methylation via GDC API...")
    try:
        files = get_gdc_files(project, "Methylation Beta Value")
        file_ids = [f["file_id"] for f in files[:50]]  # subset for speed
        download_gdc_files(file_ids, "data/methylation_raw")
        # parse downloaded files into DataFrame
        # implementation depends on GDC file format (TSV)
        methylation = parse_gdc_methylation("data/methylation_raw")
    except Exception as e:
        print(f"GDC methylation fetch failed: {e}")
        raise RuntimeError(
            "Cannot proceed without methylation data. "
            "Document failure in outputs/verification_report.md."
        )
```

## Merging and filtering (paper Section 2.1)

```python
def merge_and_filter(mrna, methylation, clinical):
    # Standardize sample IDs to first 12 characters (TCGA barcode)
    mrna.columns = [c[:12] for c in mrna.columns]
    methylation.columns = [c[:12] for c in methylation.columns]

    # Find common samples across all three modalities
    common = (
        set(mrna.columns)
        .intersection(set(methylation.columns))
        .intersection(set(clinical.index))
    )
    print(f"Common samples before filtering: {len(common)}")

    mrna = mrna[list(common)]
    methylation = methylation[list(common)]
    clinical = clinical.loc[list(common)]

    # Remove genes with all-zero expression (mRNA only)
    mrna = mrna.loc[(mrna != 0).any(axis=1)]
    print(f"mRNA genes after zero filter: {mrna.shape[0]}")

    # Extract survival columns
    # TCGA clinical uses different column names -- try common variants
    time_col = next(
        (c for c in clinical.columns
         if "days_to_death" in c.lower() or "os_time" in c.lower()
         or "overall_survival" in c.lower()),
        None
    )
    event_col = next(
        (c for c in clinical.columns
         if "vital_status" in c.lower() or "os_status" in c.lower()
         or "deceased" in c.lower()),
        None
    )

    if time_col is None or event_col is None:
        raise ValueError(
            f"Could not find survival columns. "
            f"Available: {list(clinical.columns)}"
        )

    clinical = clinical[[time_col, event_col]].copy()
    clinical.columns = ["OS_time", "OS_status"]

    # Convert to numeric, remove missing and negative
    clinical["OS_time"] = pd.to_numeric(clinical["OS_time"], errors="coerce")
    clinical["OS_status"] = pd.to_numeric(clinical["OS_status"], errors="coerce")
    clinical = clinical.dropna()
    clinical = clinical[clinical["OS_time"] > 0]

    # Align final samples
    final_samples = list(
        set(mrna.columns)
        .intersection(set(methylation.columns))
        .intersection(set(clinical.index))
    )

    return (
        mrna[final_samples],
        methylation[final_samples],
        clinical.loc[final_samples],
    )

mrna, methylation, clinical = merge_and_filter(mrna, methylation, clinical)
print(f"Final dataset: {len(clinical)} samples")
```

## Save

```python
mrna.to_parquet("data/mrna_brca.parquet")
methylation.to_parquet("data/methylation_brca.parquet")
clinical.to_csv("data/clinical_brca.csv")

summary = {
    "n_samples": len(clinical),
    "n_mrna_genes": mrna.shape[0],
    "n_methylation_genes": methylation.shape[0],
    "os_events": int(clinical["OS_status"].sum()),
    "follow_up_min_months": float(clinical["OS_time"].min()),
    "follow_up_max_months": float(clinical["OS_time"].max()),
    "source": "biolearns" if methylation is not None else "gdc_api",
}

with open("data/dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
```

## Verification against paper Table 1

```python
n = summary["n_samples"]
expected_n = expected["dataset"]["n_samples"]
tolerance = 0.05  # 5% tolerance -- TCGA data versioning may differ

assert abs(n - expected_n) / expected_n <= tolerance, (
    f"Sample count mismatch: got {n}, expected {expected_n}. "
    f"Check TCGA data version."
)

print(f"Sample count check: {n} vs expected {expected_n} -- OK")
print("Data fetching complete. Proceeding to analysis.")
```

## Output

- `data/mrna_brca.parquet`
- `data/methylation_brca.parquet`
- `data/clinical_brca.csv`
- `data/dataset_summary.json`

## Generalization

To use with a different TCGA cohort:

1. Update data.tcga_project in paper_config.yaml
2. Update expected dataset stats in extracted_results.json
3. The merge/filter logic is generic for any TCGA project
