# tools/data_fetcher.py
"""
Download TCGA-BRCA mRNA, DNA methylation, and clinical data from Firehose.

Primary strategy: Broad Institute GDAC Firehose (2016-01-28 release).
This is the exact data version used in the paper (Section 2.1).

Reads configuration from paper_config.yaml.
Outputs parquet files and dataset_summary.json to data/.
"""

import io
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIREHOSE_BASE = (
    "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/BRCA/20160128/"
)

FIREHOSE_FILES = {
    "mrna": (
        "gdac.broadinstitute.org_BRCA.Merge_rnaseqv2__illuminahiseq_rnaseqv2"
        "__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz"
    ),
    "methylation": (
        "gdac.broadinstitute.org_BRCA.Merge_methylation__humanmethylation450"
        "__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data"
        ".Level_3.2016012800.0.0.tar.gz"
    ),
    "clinical": (
        "gdac.broadinstitute.org_BRCA.Clinical_Pick_Tier1"
        ".Level_4.2016012800.0.0.tar.gz"
    ),
}


def load_config(config_path: str = "paper_config.yaml") -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file, by default "paper_config.yaml".

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_firehose_tar(
    filename: str,
    output_dir: str,
    timeout: int = 300,
) -> list[str]:
    """
    Download and extract a Firehose tar.gz archive with progress bar.

    Parameters
    ----------
    filename : str
        Firehose archive filename (basename only).
    output_dir : str
        Directory where extracted files will be saved.
    timeout : int, optional
        Request timeout in seconds, by default 300.

    Returns
    -------
    list[str]
        List of extracted file paths.

    Notes
    -----
    Firehose archives contain a single data file plus a MANIFEST.txt.
    Both are extracted to output_dir.
    Progress is shown in bytes downloaded.
    """
    url = FIREHOSE_BASE + filename
    logger.info(f"Downloading {filename[:60]}...")

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    content = b""

    with tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=filename[:50],
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            content += chunk
            pbar.update(len(chunk))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    extracted_paths = []

    with tarfile.open(fileobj=io.BytesIO(content)) as tar:
        for member in tar.getmembers():
            tar.extract(member, output_dir)
            extracted_paths.append(os.path.join(output_dir, member.name))
            logger.info(f"Extracted: {member.name}")

    return extracted_paths


def parse_mrna(raw_dir: str) -> pd.DataFrame:
    """
    Parse Firehose RSEM genes normalized mRNA data.

    Parameters
    ----------
    raw_dir : str
        Directory containing extracted Firehose files.

    Returns
    -------
    pd.DataFrame
        mRNA expression DataFrame (genes x samples).

    Notes
    -----
    Firehose RSEM normalized files have a two-row header.
    First row: gene symbols in SYMBOL|ENTREZID format.
    Second row: "RPKM" labels -- skipped.
    Columns are TCGA barcodes.
    """
    data_file = None
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if "RSEM_genes_normalized" in fname and fname.endswith(".txt"):
                data_file = os.path.join(root, fname)
                break

    if data_file is None:
        raise FileNotFoundError(
            f"Could not find RSEM normalized data file in {raw_dir}"
        )

    logger.info(f"Parsing mRNA from {os.path.basename(data_file)}...")

    df = pd.read_csv(
        data_file,
        sep="\t",
        index_col=0,
        header=0,
        skiprows=[1],
    )

    # Gene names are in format "SYMBOL|ENTREZID" -- keep symbol only
    df.index = [str(idx).split("|")[0] for idx in df.index]

    # Remove genes with symbol "?"
    df = df[df.index != "?"]

    logger.info(f"mRNA parsed: {df.shape} (genes x samples)")
    return df


def parse_methylation(raw_dir: str) -> pd.DataFrame:
    """
    Parse Firehose HumanMethylation450 data using Polars for speed.

    Parameters
    ----------
    raw_dir : str
        Directory containing extracted Firehose files.

    Returns
    -------
    pd.DataFrame
        Methylation DataFrame (genes x samples), returned as pandas
        for compatibility with downstream pipeline.

    Notes
    -----
    Firehose HM450 format: for each sample, three columns are present:
    Beta_value, Gene_Symbol, Chromosome. Total columns = n_samples * 3 + 1.

    Polars is used for parsing due to its multi-threaded CSV reader,
    which is significantly faster than pandas for wide files (3541 cols).

    Duplicate sample barcodes are detected and removed -- only the first
    occurrence of each barcode is kept.

    APPROXIMATION: paper uses probe with minimal correlation to gene
    expression (Section 2.1). Mean aggregation per gene is used instead.
    This is expected to contribute to C-index discrepancy vs paper.
    Documented in outputs/verification_report.md.
    """
    import polars as pl

    data_file = None
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if "within_bioassay" in fname and fname.endswith(".txt"):
                data_file = os.path.join(root, fname)
                break

    if data_file is None:
        raise FileNotFoundError(
            f"Could not find methylation data file in {raw_dir}"
        )

    logger.info(f"Parsing methylation from {os.path.basename(data_file)}...")

    # Read first header row to get sample names and column structure
    header = pd.read_csv(data_file, sep="\t", nrows=1, header=None)
    n_cols = header.shape[1]
    n_samples = (n_cols - 1) // 3
    logger.info(f"Methylation: {n_samples} samples, {n_cols} total columns")

    # Sample names from header row -- some barcodes may be duplicated
    # Col structure per sample: Beta_value (1+i*3), Gene_Symbol (2+i*3), Chromosome (3+i*3)
    sample_names_raw = [
        str(header.iloc[0, 1 + i * 3]) for i in range(n_samples)
    ]

    # Deduplicate while preserving first occurrence
    # Duplicates get a _dup suffix so Polars accepts the column names
    seen = {}
    sample_names = []
    for name in sample_names_raw:
        if name not in seen:
            seen[name] = 0
            sample_names.append(name)
        else:
            seen[name] += 1
            sample_names.append(f"{name}_dup{seen[name]}")

    n_dupes = len(sample_names_raw) - len(set(sample_names_raw))
    logger.info(
        f"Samples: {len(set(sample_names_raw))} unique, "
        f"{n_dupes} duplicates marked for removal"
    )

    # Column indices to read: probe ID (0), Gene_Symbol (2), all Beta_values
    beta_col_indices = [0, 2] + [1 + i * 3 for i in range(n_samples)]

    logger.info(
        f"Reading {len(beta_col_indices)} of {n_cols} columns with Polars..."
    )

    # Polars multi-threaded CSV read -- significantly faster than pandas
    with tqdm(desc="Reading methylation (Polars)", unit=" file") as pbar:
        df_pl = pl.read_csv(
            data_file,
            separator="\t",
            skip_rows=2,
            has_header=False,
            columns=beta_col_indices,
            null_values=["NA", "nan", "NaN", ""],
            infer_schema_length=1000,
        )
        pbar.update(1)

    # Assign column names: probe_id, gene_symbol, then one per sample
    df_pl.columns = ["probe_id", "gene_symbol"] + sample_names
    logger.info(f"Raw methylation loaded: {df_pl.shape}")

    # Force gene_symbol to String type -- Polars may infer it as numeric
    # if the first rows contain values that look like numbers
    df_pl = df_pl.with_columns(
        pl.col("gene_symbol").cast(pl.Utf8, strict=False)
    )

    # Remove probes without gene annotation
    df_pl = df_pl.filter(
        pl.col("gene_symbol").is_not_null() &
        (pl.col("gene_symbol") != "")
    )

    # Cast all sample columns to Float32 for memory efficiency
    sample_cols_all = sample_names
    df_pl = df_pl.with_columns([
        pl.col(c).cast(pl.Float32, strict=False) for c in sample_cols_all
    ])

    # Aggregate probes to gene level using mean
    logger.info(
        f"Aggregating {df_pl.shape[0]} probes to gene level "
        f"(Polars groupby)..."
    )
    with tqdm(desc="Aggregating probes", unit=" file") as pbar:
        df_agg = df_pl.group_by("gene_symbol").agg([
            pl.col(c).mean() for c in sample_cols_all
        ])
        pbar.update(1)

    logger.info(f"Methylation aggregated: {df_agg.shape}")

    # Drop duplicate columns -- keep only first occurrence of each barcode
    clean_sample_names = [c for c in sample_names if "_dup" not in c]
    df_agg = df_agg.select(["gene_symbol"] + clean_sample_names)

    logger.info(
        f"After deduplication: {len(clean_sample_names)} samples retained"
    )

    # Convert to pandas with gene_symbol as index
    df_pd = df_agg.to_pandas().set_index("gene_symbol")
    df_pd = df_pd.dropna(how="all")

    logger.info(f"Methylation parsed: {df_pd.shape} (genes x samples)")
    return df_pd


def parse_clinical(raw_dir: str) -> pd.DataFrame:
    """
    Parse Firehose clinical data.

    Parameters
    ----------
    raw_dir : str
        Directory containing extracted Firehose files.

    Returns
    -------
    pd.DataFrame
        Clinical DataFrame.

    Notes
    -----
    Uses BRCA.clin.merged.picked.txt which contains curated
    clinical variables selected by Firehose Tier1 picking.
    Samples are columns, variables are rows -- transposed on load.
    """
    # Prefer the picked clinical file
    data_file = None
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if "merged.picked" in fname and fname.endswith(".txt"):
                data_file = os.path.join(root, fname)
                break
        if data_file is None:
            for fname in files:
                if fname.endswith(".txt") and "MANIFEST" not in fname \
                        and "All_CDEs" not in fname \
                        and "params" not in fname:
                    data_file = os.path.join(root, fname)

    if data_file is None:
        raise FileNotFoundError(
            f"Could not find clinical data file in {raw_dir}"
        )

    logger.info(f"Parsing clinical from {os.path.basename(data_file)}...")

    df = pd.read_csv(data_file, sep="\t", index_col=0, header=0)

    # Firehose clinical: variables as rows, samples as columns -- transpose
    if df.shape[0] < df.shape[1]:
        df = df.T

    df.index = [str(idx).upper() for idx in df.index]
    logger.info(f"Clinical shape: {df.shape}")
    logger.info(f"Clinical columns (first 10): {list(df.columns[:10])}")

    return df


def extract_survival_columns(clinical: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and standardize survival columns from TCGA clinical data.

    Parameters
    ----------
    clinical : pd.DataFrame
        Raw TCGA clinical DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns OS_time (months) and OS_status (0/1).

    Raises
    ------
    ValueError
        If survival columns cannot be identified.

    Notes
    -----
    TCGA clinical files use inconsistent column naming across versions.
    OS_time in Firehose is in days -- converted to months by /30.44.
    """
    cols_lower = {c.lower(): c for c in clinical.columns}

    # Find time columns
    death_col = next(
        (cols_lower[a.lower()] for a in ["days_to_death"]
         if a.lower() in cols_lower), None
    )
    followup_col = next(
        (cols_lower[a.lower()] for a in
         ["days_to_last_followup", "days_to_last_follow_up"]
         if a.lower() in cols_lower), None
    )
    event_col = next(
        (cols_lower[a.lower()] for a in
         ["vital_status", "OS", "os_status"]
         if a.lower() in cols_lower), None
    )

    if event_col is None:
        raise ValueError(
            f"Could not identify event column.\n"
            f"Available: {list(clinical.columns[:20])}"
        )

    result = pd.DataFrame(index=clinical.index)

    # OS_time: days_to_death for deceased, days_to_last_followup for living
    death = pd.to_numeric(clinical[death_col], errors="coerce") \
        if death_col else pd.Series(float("nan"), index=clinical.index)
    followup = pd.to_numeric(clinical[followup_col], errors="coerce") \
        if followup_col else pd.Series(float("nan"), index=clinical.index)

    result["OS_time"] = death.fillna(followup)

    # OS_status: 1=deceased, 0=living
    result["OS_status"] = clinical[event_col].map(
        lambda x: 1 if str(x).lower() in
        ["dead", "deceased", "1", "true", "yes"] else 0
    )

    result["OS_time"] = pd.to_numeric(result["OS_time"], errors="coerce")

    # Convert days to months
    if result["OS_time"].median() > 100:
        logger.info("Converting OS_time from days to months (/30.44)")
        result["OS_time"] = result["OS_time"] / 30.44

    result["OS_status"] = pd.to_numeric(result["OS_status"], errors="coerce")

    logger.info(
        f"Survival columns: n={len(result)}, "
        f"events={int(result['OS_status'].sum())}, "
        f"median_time={result['OS_time'].median():.1f} months"
    )
    return result


def standardize_sample_ids(
    df: pd.DataFrame,
    n_chars: int = 12,
) -> pd.DataFrame:
    """
    Standardize TCGA sample IDs to first n characters.
    Duplicate barcodes after truncation are deduplicated -- only the
    first occurrence is kept.
    """
    df.columns = [str(c)[:n_chars].upper() for c in df.columns]

    # Drop duplicate columns keeping first occurrence
    duplicated = df.columns.duplicated(keep="first")
    n_dupes = duplicated.sum()
    if n_dupes > 0:
        logger.info(f"Dropping {n_dupes} duplicate sample columns")
        df = df.loc[:, ~duplicated]

    return df


def merge_and_filter(
    mrna: pd.DataFrame,
    methylation: pd.DataFrame,
    clinical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge and filter TCGA modalities to matching samples.

    Parameters
    ----------
    mrna : pd.DataFrame
        mRNA expression DataFrame (genes x samples).
    methylation : pd.DataFrame
        DNA methylation DataFrame (genes x samples).
    clinical : pd.DataFrame
        Clinical DataFrame with OS_time and OS_status columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (mrna, methylation, clinical) filtered to common samples.

    Notes
    -----
    Applies filtering steps from paper Section 2.1:
    1. Standardize sample IDs to 12 characters
    2. Find common samples across all three modalities
    3. Remove genes with all-zero mRNA expression
    4. Remove samples with missing or negative OS_time
    """
    mrna = standardize_sample_ids(mrna)
    methylation = standardize_sample_ids(methylation)
    clinical.index = [str(i)[:12].upper() for i in clinical.index]

    common = (
        set(mrna.columns)
        .intersection(set(methylation.columns))
        .intersection(set(clinical.index))
    )
    logger.info(f"Common samples before filtering: {len(common)}")

    if len(common) == 0:
        raise ValueError(
            "No common samples found across modalities.\n"
            f"mRNA sample example: {list(mrna.columns[:3])}\n"
            f"Methylation sample example: {list(methylation.columns[:3])}\n"
            f"Clinical index example: {list(clinical.index[:3])}"
        )

    common = sorted(common)
    mrna = mrna[common]
    methylation = methylation[common]
    clinical = clinical.loc[common]

    # Remove all-zero mRNA genes (paper Section 2.1)
    mrna = mrna.loc[(mrna != 0).any(axis=1)]
    logger.info(f"mRNA genes after zero filter: {mrna.shape[0]}")

    # Remove samples with missing or negative OS_time
    valid = clinical["OS_time"].notna() & (clinical["OS_time"] > 0)
    n_removed = (~valid).sum()
    if n_removed > 0:
        logger.info(f"Removing {n_removed} samples with invalid OS_time")
    clinical = clinical[valid]

    final_samples = list(clinical.index)
    mrna = mrna[final_samples]
    methylation = methylation[final_samples]

    logger.info(f"Final dataset: {len(final_samples)} samples")
    logger.info(f"mRNA shape: {mrna.shape}")
    logger.info(f"Methylation shape: {methylation.shape}")
    logger.info(f"Events: {int(clinical['OS_status'].sum())}")

    return mrna, methylation, clinical


def validate_dataset(
    mrna: pd.DataFrame,
    methylation: pd.DataFrame,
    clinical: pd.DataFrame,
    expected: dict,
    tolerance: float = 0.05,
) -> None:
    """
    Validate dataset against expected values from extracted_results.json.

    Parameters
    ----------
    mrna : pd.DataFrame
        mRNA expression DataFrame.
    methylation : pd.DataFrame
        Methylation DataFrame.
    clinical : pd.DataFrame
        Clinical DataFrame.
    expected : dict
        Expected dataset characteristics from extracted_results.json.
    tolerance : float, optional
        Fractional tolerance for sample count comparison, by default 0.05.

    Notes
    -----
    Uses 5% tolerance on sample count because the current Firehose
    release may include quality-control updates vs the paper's version.
    """
    n_samples = len(clinical)
    expected_n = int(expected["dataset"]["n_samples"])
    delta_pct = abs(n_samples - expected_n) / expected_n

    if delta_pct > tolerance:
        logger.warning(
            f"Sample count mismatch: got {n_samples}, "
            f"expected {expected_n} "
            f"({delta_pct*100:.1f}% > {tolerance*100:.0f}% tolerance)"
        )
    else:
        logger.info(
            f"Sample count OK: {n_samples} "
            f"(expected {expected_n}, delta {delta_pct*100:.1f}%)"
        )

    logger.info(f"mRNA genes: {mrna.shape[0]}")
    logger.info(f"Methylation genes: {methylation.shape[0]}")
    logger.info(f"Events: {int(clinical['OS_status'].sum())}")


def fetch_data(
    config_path: str = "paper_config.yaml",
    output_dir: str = "data",
) -> dict:
    """
    Fetch TCGA-BRCA data from Firehose 2016 release.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file, by default "paper_config.yaml".
    output_dir : str, optional
        Directory where output files will be saved, by default "data".

    Returns
    -------
    dict
        Dataset summary saved to data/dataset_summary.json.

    Notes
    -----
    Downloads the exact Firehose 2016-01-28 release used in the paper.
    Skips download if raw files already exist on disk.

    Saves four files:
    - data/mrna_brca.parquet
    - data/clinical_brca.csv
    - data/methylation_brca.parquet
    - data/dataset_summary.json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load expected values for validation
    expected = {}
    if os.path.exists("outputs/extracted_results.json"):
        with open("outputs/extracted_results.json") as f:
            expected = json.load(f)

    mrna_raw_dir = f"{output_dir}/mrna_raw"
    meth_raw_dir = f"{output_dir}/methylation_raw_firehose"
    clin_raw_dir = f"{output_dir}/clinical_raw"

    # Skip download if raw files already exist
    def needs_download(raw_dir: str, keyword: str) -> bool:
        for root, _, files in os.walk(raw_dir):
            for fname in files:
                if keyword in fname and fname.endswith(".txt"):
                    logger.info(f"Found existing file in {raw_dir} -- skipping download")
                    return False
        return True

    if needs_download(mrna_raw_dir, "RSEM_genes_normalized"):
        download_firehose_tar(FIREHOSE_FILES["mrna"], mrna_raw_dir)

    if needs_download(meth_raw_dir, "within_bioassay"):
        download_firehose_tar(FIREHOSE_FILES["methylation"], meth_raw_dir)

    if needs_download(clin_raw_dir, "merged.picked"):
        download_firehose_tar(FIREHOSE_FILES["clinical"], clin_raw_dir)

    # Parse
    mrna = parse_mrna(mrna_raw_dir)
    methylation = parse_methylation(meth_raw_dir)
    clinical_raw = parse_clinical(clin_raw_dir)
    clinical = extract_survival_columns(clinical_raw)

    # Merge and filter
    mrna, methylation, clinical = merge_and_filter(mrna, methylation, clinical)

    # Validate
    if expected:
        validate_dataset(mrna, methylation, clinical, expected)

    # Save
    mrna_path = f"{output_dir}/mrna_brca.parquet"
    meth_path = f"{output_dir}/methylation_brca.parquet"
    clin_path = f"{output_dir}/clinical_brca.csv"

    logger.info("Saving parquet files...")
    mrna.to_parquet(mrna_path)
    methylation.to_parquet(meth_path)
    clinical.to_csv(clin_path)

    logger.info(f"Saved mRNA to {mrna_path}")
    logger.info(f"Saved methylation to {meth_path}")
    logger.info(f"Saved clinical to {clin_path}")

    summary = {
        "n_samples": len(clinical),
        "n_mrna_genes": mrna.shape[0],
        "n_methylation_genes": methylation.shape[0],
        "os_events": int(clinical["OS_status"].sum()),
        "follow_up_min_months": float(clinical["OS_time"].min()),
        "follow_up_max_months": float(clinical["OS_time"].max()),
        "source": "firehose_2016_01_28",
    }

    summary_path = f"{output_dir}/dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset summary:\n{json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    summary = fetch_data()
    print("\nDataset summary:")
    print(json.dumps(summary, indent=2))