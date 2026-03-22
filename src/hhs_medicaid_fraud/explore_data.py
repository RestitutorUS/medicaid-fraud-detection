"""
Exploratory Data Analysis for Medicaid Provider Spending CSV.

Uses DuckDB to scan the full ~11 GB file efficiently: gets an exact
row count and a true uniform random sample for statistical summaries.
"""

import duckdb
import pandas as pd

cfg = load_config()
CSV_PATH = cfg["data"]["spending"]
SAMPLE_SIZE = 10_000
RANDOM_SEED = cfg["model"]["random_seed"]


from .utils import load_sample_duckdb


from .utils import print_section


def main():
    # --- Head (first 10 rows) ---
    print_section("HEAD (first 10 rows, file is sorted by TOTAL_PAID desc)")
    head = pd.read_csv(CSV_PATH, nrows=10)
    print(head.to_string(index=False))

    # --- True random sample from entire file via DuckDB ---
    print_section(f"RANDOM SAMPLE ({SAMPLE_SIZE:,} from full file, seed={RANDOM_SEED})")
    total_rows, df = load_sample_duckdb(CSV_PATH, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Sample loaded: {len(df):,} rows x {len(df.columns)} columns")

    # --- Dtypes ---
    print_section("DATA TYPES (pandas inferred)")
    for col in df.columns:
        print(f"  {col:40s} {df[col].dtype}")

    # --- Missing values ---
    print_section("MISSING VALUES")
    for col in df.columns:
        null_count = df[col].isna().sum()
        pct = 100.0 * null_count / len(df)
        print(f"  {col:40s} {null_count:>6,} ({pct:.2f}%)")

    # --- Format checks ---
    print_section("FORMAT CHECKS")

    npi_cols = ["BILLING_PROVIDER_NPI_NUM", "SERVICING_PROVIDER_NPI_NUM"]
    for col in npi_cols:
        lengths = df[col].astype(str).str.len().value_counts().sort_index()
        print(f"  {col} string lengths: {dict(lengths)}")

    hcpcs_lengths = df["HCPCS_CODE"].astype(str).str.len().value_counts().sort_index()
    print(f"  HCPCS_CODE string lengths: {dict(hcpcs_lengths)}")

    dates = pd.to_datetime(df["CLAIM_FROM_MONTH"], errors="coerce")
    print(f"  CLAIM_FROM_MONTH range: {dates.min()} to {dates.max()}")
    print(f"  CLAIM_FROM_MONTH unparseable: {dates.isna().sum()}")

    # --- Cardinality ---
    print_section("CARDINALITY (unique values)")
    for col in df.columns:
        print(f"  {col:40s} {df[col].nunique():>10,}")

    # --- Numeric distributions ---
    print_section("NUMERIC DISTRIBUTIONS (from random sample)")
    numeric_cols = ["TOTAL_UNIQUE_BENEFICIARIES", "TOTAL_CLAIMS", "TOTAL_PAID"]
    percentiles = [0.25, 0.50, 0.75, 0.95, 0.99]
    for col in numeric_cols:
        s = df[col]
        print(f"\n  {col}:")
        print(f"    min    = {s.min():>20,.2f}")
        print(f"    max    = {s.max():>20,.2f}")
        print(f"    mean   = {s.mean():>20,.2f}")
        print(f"    median = {s.median():>20,.2f}")
        print(f"    std    = {s.std():>20,.2f}")
        for p in percentiles:
            print(f"    p{int(p*100):02d}    = {s.quantile(p):>20,.2f}")

    # --- Billing vs Servicing NPI match ---
    print_section("BILLING vs SERVICING NPI")
    match_count = (df["BILLING_PROVIDER_NPI_NUM"] == df["SERVICING_PROVIDER_NPI_NUM"]).sum()
    match_pct = 100.0 * match_count / len(df)
    print(f"  Same NPI: {match_count:,} / {len(df):,} ({match_pct:.1f}%)")

    # --- Top HCPCS codes ---
    print_section("TOP 20 HCPCS CODES (by frequency in sample)")
    top_hcpcs = df["HCPCS_CODE"].value_counts().head(20)
    for code, count in top_hcpcs.items():
        pct = 100.0 * count / len(df)
        print(f"  {code:10s} {count:>6,} ({pct:.1f}%)")

    # --- Top billing providers ---
    print_section("TOP 10 BILLING PROVIDERS (by frequency in sample)")
    top_npi = df["BILLING_PROVIDER_NPI_NUM"].value_counts().head(10)
    for npi, count in top_npi.items():
        pct = 100.0 * count / len(df)
        print(f"  {npi}  {count:>6,} ({pct:.1f}%)")

    # --- Round-number check on TOTAL_PAID ---
    print_section("ROUND-NUMBER CHECK (TOTAL_PAID)")
    round_dollar = (df["TOTAL_PAID"] == df["TOTAL_PAID"].round(0)).sum()
    round_hundred = (df["TOTAL_PAID"] % 100 == 0).sum()
    print(f"  Exact whole dollars: {round_dollar:,} / {len(df):,} ({100*round_dollar/len(df):.1f}%)")
    print(f"  Exact multiples of $100: {round_hundred:,} / {len(df):,} ({100*round_hundred/len(df):.1f}%)")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
