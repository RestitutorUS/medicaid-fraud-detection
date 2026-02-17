# Medicaid Anomaly Detection Pipeline

An unsupervised machine learning pipeline that identifies anomalous billing patterns in Medicaid provider spending data published by the U.S. Department of Health and Human Services. The pipeline ingests 227 million claim-level records ($843B in total payments, 2018-2024), enriches them with the CMS National Provider Registry (NPPES) and the OIG exclusion list (LEIE), and surfaces providers whose billing behavior deviates significantly from their peers — across three complementary analysis layers.

> **Disclaimer:** The anomaly flags produced by this pipeline are statistical outliers, not accusations of fraud. Many flagged providers may have legitimate explanations for unusual billing patterns (specialized patient populations, geographic factors, data reporting differences, etc.). This project is published for research and educational purposes. It does not represent the views of CMS, HHS, or any government agency. Any findings should be independently verified before drawing conclusions about specific providers.

## Data Sources

| Source | Size | Description | Link |
|---|---|---|---|
| **Medicaid Provider Spending** | ~11 GB (227M rows) | Provider-level Medicaid spending aggregated from T-MSIS outpatient/professional claims. Each row = one provider x HCPCS code x month. Covers Jan 2018 - Dec 2024, all states/territories. | [HHS Open Data](https://data.medicaid.gov/dataset/fb9e42e0-520c-4a2c-aa0e-8270f75e5753/) |
| **NPPES NPI Registry** | ~11 GB (9.4M rows) | CMS National Plan and Provider Enumeration System. Demographics, taxonomy codes, authorized officials, and deactivation status for every NPI in the US. | [CMS NPPES Downloads](https://download.cms.gov/nppes/NPI_Files.html) |
| **NPPES Deactivated NPI Report** | 5 MB (334K rows) | Supplemental list of deactivated NPIs with deactivation dates. | Included in NPPES download |
| **OIG LEIE Exclusion List** | 15 MB (83K rows) | List of Excluded Individuals/Entities — providers barred from participating in federal healthcare programs. Downloaded automatically by the pipeline. | [OIG Exclusions](https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv) |

All data is publicly available. The spending CSV and NPPES files must be downloaded manually due to their size. The LEIE is downloaded automatically at runtime.

## Pipeline Overview

The pipeline has three analysis layers, each building on the previous:

### Layer 1: General Anomaly Detection (California pilot)

**Scripts:** `explore_data.py` → `feature_engineering.py` → `train_model.py` → `analyze_results.py`

Computes 24 provider-level features from billing behavior (volume, intensity, concentration, temporal patterns), peer comparison (z-scores against providers billing the same HCPCS codes), and NPI enrichment (entity type, age, deactivation status, specialty). Runs two unsupervised models: a statistical baseline (count of features with |z-score| > 3) and an Isolation Forest. Produces ranked anomaly scores for all providers.

### Layer 2: Home Health Deep-Dive

**Script:** `analyze_home_health.py`

Focuses on home health agencies (taxonomy codes 251E, 253Z, 251J) — a sector with historically elevated fraud risk. Extracts HCPCS-level billing detail for all HH providers nationally, computes HH-specific features (skilled nursing vs aide ratio, procedure concentration, pricing outliers), and produces a 13-section report with state-by-state breakdowns and peer comparisons.

### Layer 3: NPI Laundering Detection

**Script:** `npi_laundering_analysis.py`

Detects NPI laundering patterns by analyzing the network of authorized officials who control multiple organizational NPIs. Cross-references against the OIG LEIE exclusion list (with tiered confidence scoring and name frequency penalties) and deactivated NPI list. Surfaces burn-and-churn cycling, address clustering, and multi-state billing fragmentation.

## Key Findings

**Scale:**
- 227 million claim rows, 617,503 distinct billing providers, $843B total Medicaid spending through multi-NPI networks
- 1.89 million organizational NPIs analyzed, 242,247 authorized officials controlling 2+ NPIs

**General anomaly detection (CA pilot, 52,782 providers):**
- Isolation Forest and statistical baseline agree on ~60% of top-200 anomalies
- Top anomaly: a provider with $32M in spending billing 8x the peer average for skilled nursing visits

**Home health nationally (15,697 HH providers, $164B spending):**
- New York accounts for 45% of national HH spending ($73B) from only 945 providers
- Oregon and Maryland have the highest concentration of nationally anomalous HH providers (9.1% and 7.4% of their HH providers in the national top-100)
- Only 1 of 30 California top-ranked providers remains in the national top-30, demonstrating the value of national peer comparison over state-level analysis

**NPI laundering detection:**
- 123 organizational NPIs exactly match an excluded organization by business name + state — the highest-confidence findings
- 2 authorized officials matched via direct NPI link to LEIE exclusion entries (Tier 1, highest confidence)
- 1,011 officials matched at Tier 2 (name + state + city) — the actionable review set
- 1,142 officials control both active and deactivated org NPIs ("burn and churn" pattern)
- 16,970 officials have 3+ NPIs registered at the same physical address

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/) (recommended) or pip

```bash
# Clone the repo
git clone <repo-url> && cd hhsOpenData

# Create virtual environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install pandas numpy duckdb scikit-learn openpyxl
```

**Data setup:**

1. Download `medicaid-provider-spending.csv` from [HHS Open Data](https://data.medicaid.gov/dataset/fb9e42e0-520c-4a2c-aa0e-8270f75e5753/) (~11 GB, or ~3.6 GB zipped)
2. Download the [NPPES Data Dissemination](https://download.cms.gov/nppes/NPI_Files.html) and extract to `nppes/`
3. Download the [NPPES Deactivated NPI Report](https://download.cms.gov/nppes/NPI_Files.html) and place in `nppes_deactivated/`
4. The LEIE exclusion list is downloaded automatically. If the download fails, a local fallback at `leie_exclusions.csv` is used.

## Running the Pipeline

Each script reads from the raw data or the previous step's output. Run in order:

```bash
# Layer 1: General anomaly detection (California)
python explore_data.py                # EDA — sampling, distributions, format checks
python feature_engineering.py         # Feature engineering → provider_features_ca.parquet
python train_model.py                 # Isolation Forest + statistical baseline → provider_scores_ca.parquet
python analyze_results.py             # Detailed anomaly report (console output)

# Layer 2: Home health deep-dive (national)
python analyze_home_health.py         # HH-specific analysis → hh_features_national.parquet

# Layer 3: NPI laundering detection (national)
python npi_laundering_analysis.py     # AO network analysis → ao_networks.parquet, laundering_flags.parquet
```

**Runtime:** Each DuckDB scan of the 11 GB spending or NPPES files takes ~2 minutes. The NPI laundering script does 4 full scans (3 NPPES + 1 spending) and takes ~10 minutes total. All scripts stream data through DuckDB and stay within ~4 GB RAM.

## Scoping to a Different State

The general anomaly detection layer (Layer 1) is currently scoped to California. To run it for a different state, change the `STATE_FILTER` variable in `feature_engineering.py`:

```python
STATE_FILTER = "TX"  # Change to any two-letter state code
```

The home health and NPI laundering analyses already run nationally. To filter those to a single state, add a state filter to the DuckDB extraction queries in `analyze_home_health.py` or `npi_laundering_analysis.py`.

## Output Files

| File | Rows | Description |
|---|---|---|
| `provider_features_ca.parquet` | 52,782 | CA providers with 24 behavioral features |
| `provider_scores_ca.parquet` | 52,782 | Same + anomaly scores and ranks |
| `hh_claims_national.parquet` | 1,651,339 | National HH HCPCS-level claims |
| `hh_features_national.parquet` | 15,697 | National HH provider features and anomaly flags |
| `ao_networks.parquet` | 242,247 | Multi-NPI authorized official networks |
| `laundering_flags.parquet` | 5,584 | Officials flagged for laundering risk |

## License

This project uses publicly available government data. The code is provided as-is for research purposes.
