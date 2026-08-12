# tools/paper_downloader.py
"""
Download scientific paper PDFs given a DOI.

Four-strategy fallback chain:
  1. Unpaywall -- open access PDF via Unpaywall API
  2. EuropePMC -- PDF via PubMed Central
  3. SemanticScholar -- open access PDF via API + direct S2 URL
  4. LocalFallback -- manually placed PDF at output_path

Reads configuration from paper_config.yaml.

Notes
-----
Some publishers (e.g. Oxford Academic) block automated PDF downloads
even for open-access papers via cookie-based authentication. The
SemanticScholar strategy handles this by falling back to the direct
pdfs.semanticscholar.org URL, which does not require authentication.
If all automated strategies fail, LocalFallback detects a manually
placed PDF and documents the limitation explicitly.
"""

import os
import logging
from pathlib import Path
from typing import Callable

import yaml
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://academic.oup.com/",
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


def verify_pdf(path: str, min_size_bytes: int = 50_000) -> tuple[bool, str]:
    """
    Verify that a file is a valid PDF.

    Parameters
    ----------
    path : str
        Path to the file to verify.
    min_size_bytes : int, optional
        Minimum acceptable file size in bytes, by default 50_000.

    Returns
    -------
    tuple[bool, str]
        (True, message) if valid, (False, reason) if not.

    Notes
    -----
    Checks both file size and PDF magic bytes (%PDF header).
    This catches cases where a server returns an HTML error page
    with a 200 status code instead of a real PDF.
    """
    if not os.path.exists(path):
        return False, "File does not exist"

    size = os.path.getsize(path)
    if size < min_size_bytes:
        return False, f"File too small: {size} bytes (min {min_size_bytes})"

    with open(path, "rb") as f:
        header = f.read(4)
    if header != b"%PDF":
        return False, f"Not a valid PDF (header: {header})"

    return True, f"OK ({size / 1024:.0f} KB)"


def fetch_via_unpaywall(
    doi: str,
    email: str,
    output_path: str,
    timeout: int = 30,
) -> bool:
    """
    Fetch open-access PDF via Unpaywall API.

    Parameters
    ----------
    doi : str
        Digital Object Identifier of the paper.
    email : str
        Email address required by Unpaywall API terms of service.
    output_path : str
        Local path where the PDF will be saved.
    timeout : int, optional
        Request timeout in seconds, by default 30.

    Returns
    -------
    bool
        True if download succeeded, False otherwise.

    References
    ----------
    https://unpaywall.org/data-format
    """
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("is_oa"):
            logger.info("Unpaywall: paper is not open access")
            return False

        best = data.get("best_oa_location", {})
        if not best:
            logger.info("Unpaywall: no OA location found")
            return False

        pdf_url = best.get("url_for_pdf")
        if not pdf_url:
            logger.info("Unpaywall: no PDF URL in best OA location")
            return False

        logger.info(f"Unpaywall: downloading from {pdf_url}")
        pdf = requests.get(pdf_url, timeout=timeout, headers=BROWSER_HEADERS)
        pdf.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(pdf.content)
        return True

    except Exception as e:
        logger.warning(f"Unpaywall failed: {e}")
        return False


def fetch_via_europepmc(
    doi: str,
    output_path: str,
    timeout: int = 30,
) -> bool:
    """
    Fetch PDF via Europe PubMed Central.

    Parameters
    ----------
    doi : str
        Digital Object Identifier of the paper.
    output_path : str
        Local path where the PDF will be saved.
    timeout : int, optional
        Request timeout in seconds, by default 30.

    Returns
    -------
    bool
        True if download succeeded, False otherwise.

    Notes
    -----
    Uses the ?pdf=render endpoint which returns application/pdf directly,
    unlike the ptpmcrender backend which closes the connection.
    Only works for papers with a PMCID.
    """
    search_url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query=DOI:{doi}&format=json"
    )
    try:
        response = requests.get(
            search_url, timeout=10, headers=BROWSER_HEADERS
        )
        response.raise_for_status()
        data = response.json()

        if data.get("hitCount", 0) == 0:
            logger.info("EuropePMC: no results found")
            return False

        result = data["resultList"]["result"][0]
        pmcid = result.get("pmcid")

        if not pmcid:
            logger.info("EuropePMC: no PMCID available")
            return False

        # Use ?pdf=render endpoint -- returns application/pdf directly
        pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
        logger.info(f"EuropePMC: downloading {pmcid} from {pdf_url}")
        pdf = requests.get(pdf_url, timeout=timeout, headers=BROWSER_HEADERS)
        pdf.raise_for_status()

        if "application/pdf" not in pdf.headers.get("content-type", ""):
            logger.warning(
                f"EuropePMC: unexpected content-type: "
                f"{pdf.headers.get('content-type')}"
            )
            return False

        with open(output_path, "wb") as f:
            f.write(pdf.content)
        return True

    except Exception as e:
        logger.warning(f"EuropePMC failed: {e}")
        return False

def fetch_via_semantic_scholar(
    doi: str,
    output_path: str,
    timeout: int = 30,
) -> bool:
    """
    Fetch open-access PDF via Semantic Scholar API with direct URL fallback.

    Parameters
    ----------
    doi : str
        Digital Object Identifier of the paper.
    output_path : str
        Local path where the PDF will be saved.
    timeout : int, optional
        Request timeout in seconds, by default 30.

    Returns
    -------
    bool
        True if download succeeded, False otherwise.

    Notes
    -----
    Uses two sub-strategies in order:

    A) openAccessPdf URL from the Semantic Scholar Graph API.
       This URL may point to the publisher (e.g. Oxford Academic)
       which can return 403 even for open-access papers.

    B) Direct URL on pdfs.semanticscholar.org constructed from the
       paper's Semantic Scholar ID. This endpoint does not require
       authentication and works for most indexed papers.

    References
    ----------
    https://api.semanticscholar.org/graph/v1
    """
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
    try:
        response = requests.get(
            api_url,
            params={"fields": "openAccessPdf,title,paperId"},
            timeout=10,
            headers=BROWSER_HEADERS,
        )
        response.raise_for_status()
        data = response.json()

        # Sub-strategy A: openAccessPdf URL from API
        pdf_info = data.get("openAccessPdf")
        if pdf_info:
            pdf_url = pdf_info.get("url")
            if pdf_url:
                logger.info(f"SemanticScholar: trying API URL {pdf_url}")
                pdf = requests.get(
                    pdf_url, timeout=timeout, headers=BROWSER_HEADERS
                )
                if pdf.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(pdf.content)
                    return True
                logger.warning(
                    f"SemanticScholar: API URL returned {pdf.status_code}, "
                    "trying direct S2 URL"
                )

        # Sub-strategy B: direct pdfs.semanticscholar.org URL
        # Format: /XXXX/XXXXXXXX...pdf where first 4 chars are prefix
        paper_id = data.get("paperId", "")
        if len(paper_id) > 4:
            direct_url = (
                f"https://pdfs.semanticscholar.org/"
                f"{paper_id[:4]}/{paper_id[4:]}.pdf"
            )
            logger.info(
                f"SemanticScholar: trying direct URL {direct_url}"
            )
            pdf = requests.get(
                direct_url, timeout=timeout, headers=BROWSER_HEADERS
            )
            if pdf.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(pdf.content)
                return True
            logger.warning(
                f"SemanticScholar: direct URL returned {pdf.status_code}"
            )

        logger.info("SemanticScholar: no accessible PDF found")
        return False

    except Exception as e:
        logger.warning(f"SemanticScholar failed: {e}")
        return False


def fetch_local_fallback(output_path: str) -> bool:
    """
    Detect a manually placed PDF at the expected output path.

    Parameters
    ----------
    output_path : str
        Expected path of the manually placed PDF.

    Returns
    -------
    bool
        True if a valid PDF exists at output_path, False otherwise.

    Notes
    -----
    Some publishers (e.g. Oxford Academic) block automated PDF downloads
    for open-access papers via cookie-based authentication. Unpaywall
    correctly identifies these papers as open access and returns a PDF
    URL, but the download itself returns 403 Forbidden regardless of
    User-Agent or request headers.

    In these cases, manual download via browser is the only reliable
    option. This strategy detects that case explicitly and documents it
    rather than raising an error, allowing the pipeline to continue.

    To use this fallback:
      1. Download the PDF manually via browser
      2. Save it to the path specified in output_path
      3. Re-run -- this strategy will detect it automatically

    A production system would handle this via institutional API
    credentials or a headless browser with session management.
    """
    ok, msg = verify_pdf(output_path)
    if ok:
        logger.info(f"LocalFallback: found manually placed PDF -- {msg}")
        logger.warning(
            "LocalFallback activated. Automated download was not possible.\n"
            "Likely cause: publisher blocks automated access via "
            "cookie-based authentication (e.g. Oxford Academic).\n"
            "This limitation is documented in outputs/verification_report.md."
        )
        return True
    logger.info("LocalFallback: no valid PDF found at expected path")
    return False


def download_paper(
    config_path: str = "paper_config.yaml",
    output_path: str = "data/paper.pdf",
) -> str:
    """
    Download a scientific paper PDF using a four-strategy fallback chain.

    Tries Unpaywall, EuropePMC, SemanticScholar, and LocalFallback
    in order. Verifies the downloaded file is a valid PDF before
    returning.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file, by default "paper_config.yaml".
    output_path : str, optional
        Local path where the PDF will be saved, by default "data/paper.pdf".

    Returns
    -------
    str
        Path to the downloaded PDF file.

    Raises
    ------
    RuntimeError
        If all four strategies fail or no valid PDF is found.

    Notes
    -----
    BROWSER_HEADERS are used to mimic a real browser request and avoid
    403 errors from servers that check the User-Agent. This is not
    sufficient for publishers that use cookie-based authentication,
    in which case SemanticScholar direct URL or LocalFallback handle
    the case explicitly.
    """
    config = load_config(config_path)
    doi = config["paper"]["doi"]
    email = config["paper"]["email_unpaywall"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    strategies: list[tuple[str, Callable[[], bool]]] = [
        ("Unpaywall", lambda: fetch_via_unpaywall(doi, email, output_path)),
        ("EuropePMC", lambda: fetch_via_europepmc(doi, output_path)),
        ("SemanticScholar", lambda: fetch_via_semantic_scholar(doi, output_path)),
        ("LocalFallback", lambda: fetch_local_fallback(output_path)),
    ]

    for name, strategy in strategies:
        logger.info(f"Trying {name}...")
        if strategy():
            ok, msg = verify_pdf(output_path)
            if ok:
                logger.info(f"Success via {name}: {msg}")
                return output_path
            else:
                logger.warning(
                    f"{name} downloaded but verification failed: {msg}"
                )
                if os.path.exists(output_path):
                    os.remove(output_path)

    raise RuntimeError(
        f"Could not retrieve PDF for DOI: {doi}\n"
        "All four strategies failed.\n"
        "To resolve:\n"
        f"  1. Download the PDF manually via browser\n"
        f"  2. Save it to: {output_path}\n"
        "  3. Re-run -- LocalFallback will detect it automatically\n"
        "  4. Document the failure in outputs/verification_report.md"
    )


if __name__ == "__main__":
    path = download_paper()
    print(f"Paper ready: {path}")