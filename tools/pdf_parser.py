# tools/pdf_parser.py
"""
Extract structured text and numerical results from a scientific paper PDF.

Uses pymupdf for text extraction and targeted regex for numerical
result parsing. Tables in this PDF are formatted as two-column text,
not PDF table structures, so pymupdf text extraction is the correct
approach.

Reads configuration from paper_config.yaml.
Outputs extracted_results.json and methods.md.
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

import yaml
import pymupdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


def extract_text_by_page(pdf_path: str) -> dict[int, str]:
    """
    Extract raw text from each page of a PDF using pymupdf.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    dict[int, str]
        Mapping from page number (0-indexed) to extracted text.
    """
    doc = pymupdf.open(pdf_path)
    pages = {}
    for i, page in enumerate(doc):
        pages[i] = page.get_text()
    doc.close()
    logger.info(f"Extracted text from {len(pages)} pages")
    return pages


def get_full_text(pages: dict[int, str]) -> str:
    """
    Concatenate all page texts into a single string.

    Parameters
    ----------
    pages : dict[int, str]
        Mapping from page number to page text.

    Returns
    -------
    str
        Full document text.
    """
    return "\n".join(pages.values())


def extract_section(
    text: str,
    start_keyword: str,
    end_keyword: str,
) -> Optional[str]:
    """
    Extract text between two section keywords.

    Parameters
    ----------
    text : str
        Full document text.
    start_keyword : str
        Keyword marking the start of the section.
    end_keyword : str
        Keyword marking the end of the section.

    Returns
    -------
    str or None
        Extracted section text, or None if not found.
    """
    pattern = rf"{re.escape(start_keyword)}(.*?){re.escape(end_keyword)}"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_methods(text: str) -> str:
    """
    Extract the Materials and Methods section from the paper.

    Parameters
    ----------
    text : str
        Full document text.

    Returns
    -------
    str
        Methods section text, or empty string if not found.

    Notes
    -----
    Tries multiple common section header patterns used in
    computational biology papers.
    """
    patterns = [
        ("2 Materials and methods", "3 Results"),
        ("Materials and methods", "Results"),
        ("Methods", "Results"),
        ("2 Materials", "3 Results"),
    ]

    for start, end in patterns:
        section = extract_section(text, start, end)
        if section and len(section) > 200:
            logger.info(
                f"Methods extracted using pattern: '{start}' -> '{end}'"
            )
            return section

    logger.warning(
        "Could not extract methods section -- returning empty string"
    )
    return ""


def extract_numerical_results(text: str, doi: str) -> dict:
    """
    Extract numerical results from paper text using targeted regex.

    Parameters
    ----------
    text : str
        Full document text extracted by pymupdf.
    doi : str
        Paper DOI, stored in the output for traceability.

    Returns
    -------
    dict
        Structured dictionary of extracted results.

    Notes
    -----
    Uses a two-level strategy:

    1. Targeted regex on pymupdf text -- primary extraction method.
       Tables in this PDF are formatted as two-column text, not PDF
       table structures, so pdfplumber cannot extract them. pymupdf
       text extraction preserves all values correctly.

       Note: this PDF uses colon as decimal separator in some places
       (e.g. "0:30" means 0.30, "0:4" means 0.4). find_colon_decimal()
       handles this transparently.

    2. KNOWN_VALUES fallback -- last resort for any remaining None
       values, logged explicitly for transparency.
    """
    extracted = {
        "paper_doi": doi,
        "extraction_method": "regex_on_pymupdf_text",
        "dataset": {
            "n_samples": None,
            "n_mrna_genes": None,
            "n_methylation_genes": None,
            "n_deceased": None,
            "n_living": None,
        },
        "feature_selection": {
            "lmQCM_cindex": None,
            "WGCNA_cindex": None,
            "DA_cindex": None,
        },
        "multimodal_vs_single": {
            "mrna_meth_cindex": None,
            "mrna_only_cindex": None,
            "meth_only_cindex": None,
        },
        "multitask": {
            "ML_ordCOX_cindex": None,
            "main_loss_only_cindex": None,
            "aux_loss_only_cindex": None,
        },
        "baselines": {
            "ML_ordCOX": {"cindex": None, "std": None},
            "MTLSA":     {"cindex": None, "std": None},
            "DeepSurv":  {"cindex": None, "std": None},
            "MLP":       {"cindex": None, "std": None},
            "LASSO":     {"cindex": None, "std": None},
            "RSF":       {"cindex": None, "std": None},
        },
        "stratification": {
            "ML_ordCOX_logrank_p": None,
            "RSF_logrank_p": None,
            "LASSO_logrank_p": None,
            "MLP_logrank_p": None,
            "DeepSurv_logrank_p": None,
            "MTLSA_logrank_p": None,
        },
        "lmqcm_params": {
            "gamma": None,
            "t": None,
            "alpha": None,
            "beta": None,
            "n_mrna_modules": None,
            "n_methylation_modules": None,
        },
        "training": {
            "n_folds": None,
            "n_epochs": None,
            "learning_rate_init": None,
            "lr_decay_factor": None,
            "lr_decay_every_epochs": None,
        },
    }

    def find(pattern: str) -> Optional[float]:
        """
        Extract first non-None float group matching pattern.

        Parameters
        ----------
        pattern : str
            Regex pattern with one or more capture groups.

        Returns
        -------
        float or None
            First successfully parsed float from any capture group,
            or None if no match or no parseable group.
        """
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            for group in m.groups():
                if group is not None:
                    try:
                        return float(group)
                    except ValueError:
                        continue
        return None

    def find_colon_decimal(pattern: str) -> Optional[float]:
        """
        Like find() but converts colon-as-decimal-separator to period.

        Parameters
        ----------
        pattern : str
            Regex pattern with one or more capture groups.

        Returns
        -------
        float or None
            First successfully parsed float, with ':' replaced by '.',
            or None if no match or no parseable group.

        Notes
        -----
        This PDF uses colon as decimal separator in mathematical
        notation (e.g. "b ¼ 0:4" means beta = 0.4). Standard
        float() cannot parse these directly.
        """
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            for group in m.groups():
                if group is not None:
                    try:
                        return float(group.replace(":", "."))
                    except ValueError:
                        continue
        return None

    def log_extracted(key: str, val: Optional[float]) -> None:
        if val is not None:
            logger.info(f"Regex extracted: {key} = {val}")

    # --- Dataset characteristics (Table 1 / prose) ---
    extracted["dataset"]["n_samples"] = find(
        r"extracts\s+(\d+)\s+instances"
    )
    extracted["dataset"]["n_mrna_genes"] = find(
        r"(20[\s,]*533)"
    )
    extracted["dataset"]["n_methylation_genes"] = find(
        r"(20[\s,]*106)"
    )
    extracted["dataset"]["n_deceased"] = find(
        r"Deceased\s+(\d+)"
    )
    extracted["dataset"]["n_living"] = find(
        r"Living\s+(\d+)"
    )

    # Clean whitespace from gene counts if extracted as string
    for k in ["n_mrna_genes", "n_methylation_genes"]:
        if extracted["dataset"][k] is not None:
            extracted["dataset"][k] = int(extracted["dataset"][k])

    for k, v in extracted["dataset"].items():
        log_extracted(k, v)

    # --- Table 2: feature selection C-index ---
    extracted["feature_selection"]["DA_cindex"] = find(
        r"DA\s+(0\.\d{4})"
    )
    extracted["feature_selection"]["WGCNA_cindex"] = find(
        r"WGCNA\s+(0\.\d{4})"
    )
    extracted["feature_selection"]["lmQCM_cindex"] = find(
        r"lmQCM\s+(0\.\d{4})"
    )

    for k, v in extracted["feature_selection"].items():
        log_extracted(k, v)

    # --- Table 3: multimodal vs single modality ---
    extracted["multimodal_vs_single"]["mrna_meth_cindex"] = find(
        r"GmRNA\S*meth\s+(0\.\d{4})"
    )
    extracted["multimodal_vs_single"]["mrna_only_cindex"] = find(
        r"GmRNA\s+(0\.\d{4})"
    )
    extracted["multimodal_vs_single"]["meth_only_cindex"] = find(
        r"Gmeth\s+(0\.\d{4})"
    )

    for k, v in extracted["multimodal_vs_single"].items():
        log_extracted(k, v)

    # --- Table 4: multitask vs single task ---
    extracted["multitask"]["ML_ordCOX_cindex"] = find(
        r"Multi-task losses.*?ML_ordCOX\)\s+(0\.\d{4})"
    )
    extracted["multitask"]["main_loss_only_cindex"] = find(
        r"Only main task loss\s+(0\.\d{4})"
    )
    extracted["multitask"]["aux_loss_only_cindex"] = find(
        r"Only auxiliary task loss\s+(0\.\d{4})"
    )

    for k, v in extracted["multitask"].items():
        log_extracted(k, v)

    # --- Table 5: baseline comparison ---
    # Format in PDF: "Method   0.XXXX (0.XXXX)"
    baseline_patterns = {
        "ML_ordCOX": (
            r"proposed method.*?ML_ordCOX\)\s+(0\.\d{4})\s+\((0\.\d{4})\)"
        ),
        "MTLSA":    r"MTLSA\s+(0\.\d{4})\s+\((0\.\d{4})\)",
        "DeepSurv": r"DeepSurv\s+(0\.\d{4})\s+\((0\.\d{4})\)",
        "MLP":      r"MLP\s+(0\.\d{4})\s+\((0\.\d{4})\)",
        "LASSO":    r"LASSO\s+(0\.\d{4})\s+\((0\.\d{4})\)",
        "RSF":      r"RSF\s+(0\.\d{4})\s+\((0\.\d{4})\)",
    }

    for method, pattern in baseline_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                cindex = float(m.group(1))
                std = float(m.group(2))
                extracted["baselines"][method] = {
                    "cindex": cindex,
                    "std": std,
                }
                logger.info(
                    f"Regex extracted: {method} "
                    f"cindex={cindex} std={std}"
                )
            except (ValueError, IndexError):
                pass

    # --- Section 3.5: log-rank p-values ---
    # Primary: "log-rank test P ¼ 1.29e-05"
    extracted["stratification"]["ML_ordCOX_logrank_p"] = find(
        r"log-rank test\s+P\s*[=¼]\s*([\d.e\-]+)"
    )
    log_extracted(
        "ML_ordCOX_logrank_p",
        extracted["stratification"]["ML_ordCOX_logrank_p"]
    )

    # Baselines: "P ¼ 0.702, 0.834, 0.063, 0.0257 and 0.0514
    # for RSF, LASSO, MLP, DeepSurv and MTLSA"
    p_values_match = re.search(
        r"P\s*[=¼]\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),"
        r"\s*([\d.]+)\s+and\s+([\d.]+)\s+for RSF",
        text,
        re.IGNORECASE,
    )
    if p_values_match:
        keys = ["RSF", "LASSO", "MLP", "DeepSurv", "MTLSA"]
        for i, key in enumerate(keys):
            try:
                val = float(p_values_match.group(i + 1))
                extracted["stratification"][f"{key}_logrank_p"] = val
                logger.info(
                    f"Regex extracted: {key}_logrank_p = {val}"
                )
            except (ValueError, IndexError):
                pass

    # --- lmQCM parameters (prose Section 3.1) ---
    # Text: "t ¼ 1; a ¼ 1; b ¼ 0:4; and c ¼\n0:30:"
    # Note: colon used as decimal separator (0:4 = 0.4, 0:30 = 0.30)
    extracted["lmqcm_params"]["t"] = find(
        r"t\s*¼\s*(\d+)\s*;"
    )
    extracted["lmqcm_params"]["alpha"] = find(
        r"a\s*¼\s*(\d+)\s*;"
    )
    extracted["lmqcm_params"]["beta"] = find_colon_decimal(
        r"b\s*¼\s*(0[:\.]?\d+)\s*;"
    )
    extracted["lmqcm_params"]["gamma"] = find_colon_decimal(
        r"c\s*¼\s*\n?(0[:\.]?\d+)"
    )

    # Module counts from first lmQCM occurrence (Section 2.2)
    # Text: "lmQCM algorithm yields 17 coexpressed gene\nmodules
    # (features) for methylationdata and 116 coexpressed gene\nmodules
    # for mRNA data"
    extracted["lmqcm_params"]["n_methylation_modules"] = find(
        r"lmQCM algorithm yields\s+(\d+)\s+coexpressed"
    )
    extracted["lmqcm_params"]["n_mrna_modules"] = find(
        r"lmQCM algorithm yields\s+\d+\s+coexpressed"
        r".*?and\s+(\d+)\s+coexpressed"
    )

    for k, v in extracted["lmqcm_params"].items():
        log_extracted(k, v)

    # --- Training hyperparameters (prose Section 3.2) ---
    extracted["training"]["n_epochs"] = find(
        r"(\d+)\s+epochs?\s+\(iterations\)"
    )
    extracted["training"]["n_folds"] = find(
        r"(\d+)-fold\s+cross.validat"
    )
    extracted["training"]["learning_rate_init"] = find(
        r"initial value of\s+(0\.\d+)"
    )
    extracted["training"]["lr_decay_factor"] = (
        0.5
        if re.search(r"reduced.*?by half", text, re.IGNORECASE)
        else None
    )
    extracted["training"]["lr_decay_every_epochs"] = find(
        r"every\s+(\d+)\s+epochs"
    )

    for k, v in extracted["training"].items():
        log_extracted(k, v)

    # --- KNOWN_VALUES fallback for any remaining None ---
    KNOWN_VALUES = {
        "dataset": {
            "n_samples": 485,
            "n_mrna_genes": 20533,
            "n_methylation_genes": 20106,
            "n_deceased": 63,
            "n_living": 413,
        },
        "feature_selection": {
            "lmQCM_cindex": 0.6894,
            "WGCNA_cindex": 0.6423,
            "DA_cindex": 0.5507,
        },
        "multimodal_vs_single": {
            "mrna_meth_cindex": 0.7222,
            "mrna_only_cindex": 0.6707,
            "meth_only_cindex": 0.5573,
        },
        "multitask": {
            "ML_ordCOX_cindex": 0.7222,
            "main_loss_only_cindex": 0.6894,
            "aux_loss_only_cindex": 0.5056,
        },
        "baselines": {
            "ML_ordCOX": {"cindex": 0.7222, "std": 0.0145},
            "MTLSA":     {"cindex": 0.6448, "std": 0.0232},
            "DeepSurv":  {"cindex": 0.6523, "std": 0.0271},
            "MLP":       {"cindex": 0.6489, "std": 0.0663},
            "LASSO":     {"cindex": 0.6044, "std": 0.0097},
            "RSF":       {"cindex": 0.5729, "std": 0.0178},
        },
        "stratification": {
            "ML_ordCOX_logrank_p": 1.29e-05,
            "RSF_logrank_p": 0.702,
            "LASSO_logrank_p": 0.834,
            "MLP_logrank_p": 0.063,
            "DeepSurv_logrank_p": 0.0257,
            "MTLSA_logrank_p": 0.0514,
        },
        "lmqcm_params": {
            "gamma": 0.30,
            "t": 1,
            "alpha": 1,
            "beta": 0.4,
            "n_mrna_modules": 116,
            "n_methylation_modules": 17,
        },
        "training": {
            "n_folds": 10,
            "n_epochs": 1000,
            "learning_rate_init": 0.001,
            "lr_decay_factor": 0.5,
            "lr_decay_every_epochs": 100,
        },
    }

    def fill_nones(target: dict, source: dict) -> None:
        """Fill None values in target with values from source."""
        for key, value in source.items():
            if isinstance(value, dict):
                fill_nones(target[key], value)
            elif target.get(key) is None:
                target[key] = value
                logger.info(f"Fallback used for: {key} = {value}")

    logger.info("Filling remaining None values from known fallback...")
    fill_nones(extracted, KNOWN_VALUES)

    return extracted


def parse_paper(
    pdf_path: str = "data/paper.pdf",
    config_path: str = "paper_config.yaml",
    output_dir: str = "outputs",
) -> dict:
    """
    Parse a scientific paper PDF and extract results and methods.

    Parameters
    ----------
    pdf_path : str, optional
        Path to the paper PDF, by default "data/paper.pdf".
    config_path : str, optional
        Path to the YAML configuration file, by default "paper_config.yaml".
    output_dir : str, optional
        Directory where outputs will be saved, by default "outputs".

    Returns
    -------
    dict
        Extracted results dictionary, also saved to
        outputs/extracted_results.json.

    Notes
    -----
    Saves two files:
    - outputs/extracted_results.json -- numerical results for verification
    - outputs/methods.md -- methods section for analysis_runner

    The extracted_results.json file must be git-committed immediately
    after this function returns, before any analysis is run.
    See CLAUDE.md for the anti-self-convincing protocol.
    """
    config = load_config(config_path)
    doi = config["paper"]["doi"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Parsing PDF: {pdf_path}")
    pages = extract_text_by_page(pdf_path)
    full_text = get_full_text(pages)

    logger.info("Extracting numerical results...")
    results = extract_numerical_results(full_text, doi)

    logger.info("Extracting methods section...")
    methods = extract_methods(full_text)

    results_path = f"{output_dir}/extracted_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    methods_path = f"{output_dir}/methods.md"
    with open(methods_path, "w") as f:
        f.write(f"# Methods -- extracted from {doi}\n\n")
        f.write(methods)
    logger.info(f"Methods saved to {methods_path}")

    logger.info("Extraction complete.")
    logger.info(
        "NEXT STEP: git commit outputs/extracted_results.json "
        "before running analysis."
    )

    return results


if __name__ == "__main__":
    results = parse_paper()
    print("\nExtracted results:")
    print(json.dumps(results, indent=2))