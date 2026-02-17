# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build an ML anomaly detection pipeline to identify likely Medicaid fraud from HHS provider spending data. This is a learning project — the user is learning as they go.

## Working Style

- **Explain before coding.** Always explain the reasoning and approach before writing any code.
- **Plan before implementing.** Use plan mode for non-trivial work. Discuss tradeoffs and get approval before building.
- **Maintain devlog.md.** After each significant step (new feature, key decision, experiment result), append an entry to `devlog.md` with the date, what was done, and why.
- **Use Python.** All code should be Python (pandas, scikit-learn, duckdb, etc. as appropriate).

## Data

See `data-dictionary.md` for the full official documentation (use cases, T-MSIS background, etc.).

### Source Files

- `medicaid-provider-spending.csv` (~11 GB) — Raw Medicaid provider spending records
- `medicaid-provider-spending.csv.zip` (~3.6 GB) — Compressed version

### NPPES Provider Registry

The `nppes/` directory contains the CMS National Plan and Provider Enumeration System (NPPES) data dissemination files (as of 2026-02-08). These provide demographic and classification data for every NPI in the US.

#### Files

- `npidata_pfile_20050523-20260208.csv` (~11 GB, 9.4M rows, 330 columns) — Full NPI registry
- `pl_pfile_20050523-20260208.csv` (105 MB) — Secondary practice locations
- `endpoint_pfile_20050523-20260208.csv` (117 MB) — Electronic endpoints (DIRECT addresses, etc.)
- `othername_pfile_20050523-20260208.csv` (31 MB) — Alternate names / DBAs
- `NPPES_Data_Dissemination_Readme_v.2.pdf` — Official documentation
- `NPPES_Data_Dissemination_CodeValues.pdf` — Code value reference

#### Key Columns (from main NPI file)

| Column | Description |
|---|---|
| NPI | National Provider Identifier (join key to spending data) |
| Entity Type Code | 1 = Individual, 2 = Organization |
| Provider Business Practice Location Address State Name | Practice state |
| Provider Business Practice Location Address Postal Code | Practice zip |
| Provider Enumeration Date | Date NPI was assigned |
| NPI Deactivation Reason Code / Date | Deactivation info |
| NPI Reactivation Date | If reactivated after deactivation |
| Healthcare Provider Taxonomy Code_1 through _15 | Specialty classification codes |
| Is Sole Proprietor | Y/N/X |
| Is Organization Subpart | Y/N |

#### Entity Type Distribution

- Individual providers: 7,139,831 (76%)
- Organizations: 1,894,688 (20%)
- Unknown/deactivated: 333,563 (4%)

### NPPES Deactivated NPI List

- `nppes_deactivated/NPPES Deactivated NPI Report 20260209.xlsx` (5 MB, 333,566 rows)
- Two columns: `NPI` and `NPPES Deactivation Date`
- Overlaps with deactivation fields in the main NPI file but may be more current

### Dataset Description

Provider-level Medicaid spending aggregated from outpatient and professional claims with valid HCPCS codes, sourced from CMS's Transformed Medicaid Statistical Information System (T-MSIS). Each row is one provider x HCPCS code x month combination. Covers January 2018 – December 2024, all states/territories, fee-for-service, managed care, and CHIP.

### CSV Schema

| Column | Type | Description |
|---|---|---|
| BILLING_PROVIDER_NPI_NUM | STRING | National Provider Identifier of the billing provider |
| SERVICING_PROVIDER_NPI_NUM | STRING | National Provider Identifier of the servicing provider |
| HCPCS_CODE | STRING | Healthcare Common Procedure Coding System code for the service |
| CLAIM_FROM_MONTH | DATE | Month for which claims are aggregated (YYYY-MM-01 format) |
| TOTAL_UNIQUE_BENEFICIARIES | INTEGER | Count of unique beneficiaries for this provider/procedure/month |
| TOTAL_CLAIMS | INTEGER | Total number of claims for this provider/procedure/month |
| TOTAL_PAID | FLOAT | Total amount paid by Medicaid (in USD) |

### Cell Suppression

Rows with fewer than 12 total claims are dropped entirely to protect beneficiary privacy. This means the dataset excludes low-volume provider-procedure combinations — the model will never see rare/low-volume billing patterns.

### Data Accuracy

Derived from T-MSIS state submissions and only as accurate as what each state reports. Quality varies by state and data element — see CMS's DQ Atlas for known issues. State Medicaid agencies are the authoritative source.

### Working with the Data

The CSV is ~11 GB. Avoid reading the entire file into memory. Use chunked reading (pandas `chunksize`), DuckDB, or sampling for exploration.

The NPPES main NPI file is also ~11 GB. For the pipeline, we only need ~6 columns from it — use DuckDB with column selection to avoid reading all 330 columns. The deactivated NPI Excel file requires `openpyxl` to read with pandas, but can also be handled by converting to CSV or using DuckDB's spatial extension.
