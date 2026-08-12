# SKILL: Paper Retrieval

# Version: 1.0

# Generalised: reads DOI from paper_config.yaml

## Purpose

Given a DOI from paper_config.yaml, download the full PDF of a
scientific paper programmatically using a three-strategy fallback chain.

## Input

paper_config.yaml -- must contain paper.doi and paper.email_unpaywall

## Setup

```python
import yaml
import requests
import os

with open("paper_config.yaml") as f:
    config = yaml.safe_load(f)

doi = config["paper"]["doi"]
email = config["paper"]["email_unpaywall"]
os.makedirs("data", exist_ok=True)
output_path = "data/paper.pdf"
```

## Strategy 1: Unpaywall (preferred)

Finds open-access versions of papers legally and for free.
Works for any open-access paper regardless of publisher.

```python
def fetch_via_unpaywall(doi, email, output_path):
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return False
    data = response.json()
    if not data.get("is_oa"):
        return False
    best = data.get("best_oa_location", {})
    pdf_url = best.get("url_for_pdf")
    if not pdf_url:
        return False
    pdf = requests.get(pdf_url, timeout=30)
    if pdf.status_code != 200:
        return False
    with open(output_path, "wb") as f:
        f.write(pdf.content)
    return True
```

## Strategy 2: Europe PMC (fallback)

Works for papers indexed in PubMed Central.
Fully generic -- no publisher-specific logic.

```python
def fetch_via_europepmc(doi, output_path):
    search_url = (
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query=DOI:{doi}&format=json"
    )
    response = requests.get(search_url, timeout=10)
    if response.status_code != 200:
        return False
    results = response.json()
    if results.get("hitCount", 0) == 0:
        return False
    pmcid = results["resultList"]["result"][0].get("pmcid")
    if not pmcid:
        return False
    pdf_url = (
        f"https://europepmc.org/backend/ptpmcrender.fcgi"
        f"?accid={pmcid}&blobtype=pdf"
    )
    pdf = requests.get(pdf_url, timeout=30)
    if pdf.status_code != 200:
        return False
    with open(output_path, "wb") as f:
        f.write(pdf.content)
    return True
```

## Strategy 3: Semantic Scholar (fallback)

Works for papers indexed in Semantic Scholar.
Returns open-access PDF URL if available.

```python
def fetch_via_semantic_scholar(doi, output_path):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
    params = {"fields": "openAccessPdf"}
    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        return False
    data = response.json()
    pdf_info = data.get("openAccessPdf")
    if not pdf_info:
        return False
    pdf_url = pdf_info.get("url")
    if not pdf_url:
        return False
    pdf = requests.get(pdf_url, timeout=30)
    if pdf.status_code != 200:
        return False
    with open(output_path, "wb") as f:
        f.write(pdf.content)
    return True
```

## Execution order

```python
strategies = [
    ("Unpaywall", lambda: fetch_via_unpaywall(doi, email, output_path)),
    ("EuropePMC", lambda: fetch_via_europepmc(doi, output_path)),
    ("SemanticScholar", lambda: fetch_via_semantic_scholar(doi, output_path)),
]

success = False
for name, strategy in strategies:
    print(f"Trying {name}...")
    try:
        if strategy():
            print(f"Success via {name}")
            success = True
            break
    except Exception as e:
        print(f"{name} failed: {e}")
        continue

if not success:
    raise RuntimeError(
        f"Could not retrieve PDF for DOI: {doi}\n"
        "All three strategies failed.\n"
        "Document this failure in outputs/verification_report.md\n"
        "and obtain the PDF manually before proceeding."
    )
```

## Verification

```python
assert os.path.exists(output_path), "PDF not downloaded"
assert os.path.getsize(output_path) > 50_000, (
    f"PDF suspiciously small: {os.path.getsize(output_path)} bytes"
)
print(f"PDF ready: {os.path.getsize(output_path) / 1024:.0f} KB")
```

## Output

- `data/paper.pdf` -- full paper PDF

## Generalization

To use with a different paper:

1. Update paper.doi in paper_config.yaml
2. Run this skill -- no other changes needed
The three-strategy chain works for any open-access paper.
