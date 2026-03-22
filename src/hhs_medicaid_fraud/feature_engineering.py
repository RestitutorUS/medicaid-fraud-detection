"""
Feature engineering for Medicaid fraud anomaly detection.

Reads the full spending CSV and NPPES provider registry via DuckDB,
filters to California providers, and computes provider-level features:
  A. Billing behavior (volume, intensity, patterns)
  B. Peer comparison (z-scores vs HCPCS code peers)
  C. NPI enrichment (entity type, age, deactivation, specialty)

Outputs: provider_features_ca.parquet
"""

import duckdb

cfg = load_config()
SPENDING_CSV = cfg["data"]["spending"]
NPPES_CSV = cfg["data"]["nppes"]
DEACTIVATED_XLSX = cfg["data"]["deactivated_xlsx"]
OUTPUT_PARQUET = "data/processed/provider_features_ca.parquet"
STATE_FILTER = cfg["state"]


def build_features():
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")

    # --- Step 1: Count total spending rows for drop-rate reporting ---
    print("Counting distinct billing NPIs in spending file...")
    total_npis = conn.sql(f"""
        SELECT COUNT(DISTINCT BILLING_PROVIDER_NPI_NUM)
        FROM '{SPENDING_CSV}'
    """).fetchone()[0]
    print(f"  Total distinct billing NPIs (national): {total_npis:,}")

    # --- Step 2: Build the feature query ---
    print(f"Building CA provider features (this scans both 11 GB files)...")

    query = f"""
    WITH
    -- Extract CA NPIs from NPPES (only columns we need from 330)
    ca_npis AS (
        SELECT
            CAST("NPI" AS VARCHAR) AS npi,
            "Entity Type Code" AS entity_type,
            "Provider Enumeration Date" AS enumeration_date,
            "NPI Deactivation Date" AS deactivation_date,
            "NPI Reactivation Date" AS reactivation_date,
            "Is Sole Proprietor" AS sole_proprietor,
            "Is Organization Subpart" AS org_subpart,
            "Healthcare Provider Taxonomy Code_1" AS primary_taxonomy,
            (CASE WHEN "Healthcare Provider Taxonomy Code_1"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_2"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_3"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_4"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_5"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_6"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_7"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_8"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_9"  IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_10" IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_11" IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_12" IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_13" IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_14" IS NOT NULL THEN 1 ELSE 0 END
           + CASE WHEN "Healthcare Provider Taxonomy Code_15" IS NOT NULL THEN 1 ELSE 0 END
            ) AS num_taxonomy_codes
        FROM read_csv('{NPPES_CSV}', header=true, quote='"')
        WHERE "Provider Business Practice Location Address State Name" = '{STATE_FILTER}'
    ),

    -- Deactivated NPI list (skip title + header rows in xlsx)
    deactivated AS (
        SELECT DISTINCT CAST("Field1" AS VARCHAR) AS npi
        FROM st_read('{DEACTIVATED_XLSX}')
        WHERE TRY_CAST("Field1" AS BIGINT) IS NOT NULL
    ),

    -- Inner join spending to CA NPIs: filters to CA + enriches
    ca_spending AS (
        SELECT
            s.BILLING_PROVIDER_NPI_NUM AS billing_npi,
            s.SERVICING_PROVIDER_NPI_NUM AS servicing_npi,
            s.HCPCS_CODE,
            s.CLAIM_FROM_MONTH,
            s.TOTAL_UNIQUE_BENEFICIARIES,
            s.TOTAL_CLAIMS,
            s.TOTAL_PAID,
            n.entity_type,
            n.enumeration_date,
            n.deactivation_date,
            n.reactivation_date,
            n.sole_proprietor,
            n.org_subpart,
            n.primary_taxonomy,
            n.num_taxonomy_codes
        FROM read_csv('{SPENDING_CSV}', header=true, auto_detect=true) s
        INNER JOIN ca_npis n ON s.BILLING_PROVIDER_NPI_NUM = n.npi
    ),

    -- Per provider-code aggregation
    provider_code AS (
        SELECT
            billing_npi,
            HCPCS_CODE,
            SUM(TOTAL_PAID) AS code_paid,
            SUM(TOTAL_CLAIMS) AS code_claims
        FROM ca_spending
        GROUP BY billing_npi, HCPCS_CODE
    ),

    -- HCPCS peer stats (CA providers only)
    hcpcs_stats AS (
        SELECT
            HCPCS_CODE,
            AVG(code_paid / code_claims) AS avg_paid_per_claim,
            STDDEV_POP(code_paid / code_claims) AS std_paid_per_claim,
            COUNT(*) AS num_providers
        FROM provider_code
        GROUP BY HCPCS_CODE
        HAVING COUNT(*) >= 5  -- need enough peers for meaningful z-score
    ),

    -- Per provider-code z-scores
    provider_code_zscore AS (
        SELECT
            pc.billing_npi,
            pc.HCPCS_CODE,
            pc.code_paid,
            pc.code_claims,
            CASE WHEN hs.std_paid_per_claim > 0
                 THEN ((pc.code_paid / pc.code_claims) - hs.avg_paid_per_claim)
                      / hs.std_paid_per_claim
                 ELSE 0
            END AS peer_zscore
        FROM provider_code pc
        LEFT JOIN hcpcs_stats hs ON pc.HCPCS_CODE = hs.HCPCS_CODE
    ),

    -- Peer z-score rollup to provider level
    provider_peer AS (
        SELECT
            billing_npi,
            MAX(peer_zscore) AS max_peer_zscore,
            SUM(peer_zscore * code_paid) / NULLIF(SUM(code_paid), 0)
                AS mean_peer_zscore
        FROM provider_code_zscore
        GROUP BY billing_npi
    ),

    -- Monthly aggregation for volatility
    provider_monthly AS (
        SELECT
            billing_npi,
            CLAIM_FROM_MONTH,
            SUM(TOTAL_PAID) AS monthly_paid
        FROM ca_spending
        GROUP BY billing_npi, CLAIM_FROM_MONTH
    ),
    provider_monthly_stats AS (
        SELECT
            billing_npi,
            CASE WHEN AVG(monthly_paid) > 0
                 THEN STDDEV_POP(monthly_paid) / AVG(monthly_paid)
                 ELSE 0
            END AS monthly_paid_cv,
            MAX(monthly_paid) AS max_monthly_paid
        FROM provider_monthly
        GROUP BY billing_npi
    ),

    -- Revenue concentration (top code share)
    provider_top_code AS (
        SELECT
            billing_npi,
            MAX(code_paid) AS top_code_paid
        FROM provider_code
        GROUP BY billing_npi
    ),

    -- Main provider-level aggregation
    provider_agg AS (
        SELECT
            billing_npi,
            -- Volume
            SUM(TOTAL_PAID)                    AS total_paid,
            SUM(TOTAL_CLAIMS)                  AS total_claims,
            SUM(TOTAL_UNIQUE_BENEFICIARIES)    AS total_beneficiaries,
            COUNT(DISTINCT HCPCS_CODE)         AS num_hcpcs_codes,
            COUNT(DISTINCT CLAIM_FROM_MONTH)   AS num_active_months,
            COUNT(*)                           AS num_rows,
            -- Intensity
            SUM(TOTAL_PAID) / SUM(TOTAL_CLAIMS)
                AS paid_per_claim,
            SUM(TOTAL_CLAIMS)::DOUBLE / SUM(TOTAL_UNIQUE_BENEFICIARIES)
                AS claims_per_beneficiary,
            SUM(TOTAL_PAID) / SUM(TOTAL_UNIQUE_BENEFICIARIES)
                AS paid_per_beneficiary,
            -- Billing patterns
            SUM(CASE WHEN TOTAL_PAID = ROUND(TOTAL_PAID, 0) THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*) AS pct_round_dollar,
            SUM(CASE WHEN TOTAL_PAID % 100 = 0 THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*) AS pct_round_hundred,
            SUM(CASE WHEN billing_npi != servicing_npi THEN 1 ELSE 0 END)::DOUBLE
                / COUNT(*) AS pct_npi_mismatch,
            -- NPI attributes (same for all rows of a provider)
            ANY_VALUE(entity_type)             AS entity_type,
            ANY_VALUE(enumeration_date)        AS enumeration_date,
            ANY_VALUE(deactivation_date)       AS deactivation_date,
            ANY_VALUE(reactivation_date)       AS reactivation_date,
            ANY_VALUE(sole_proprietor)         AS sole_proprietor,
            ANY_VALUE(org_subpart)             AS org_subpart,
            ANY_VALUE(primary_taxonomy)        AS primary_taxonomy,
            ANY_VALUE(num_taxonomy_codes)      AS num_taxonomy_codes
        FROM ca_spending
        GROUP BY billing_npi
    )

    -- Final assembly
    SELECT
        a.billing_npi,
        -- Billing behavior
        a.total_paid,
        a.total_claims,
        a.total_beneficiaries,
        a.num_hcpcs_codes,
        a.num_active_months,
        a.num_rows,
        a.paid_per_claim,
        a.claims_per_beneficiary,
        a.paid_per_beneficiary,
        a.pct_round_dollar,
        a.pct_round_hundred,
        a.pct_npi_mismatch,
        -- Revenue concentration
        tc.top_code_paid / a.total_paid AS revenue_concentration,
        -- Volatility
        ms.monthly_paid_cv,
        ms.max_monthly_paid,
        -- Peer comparison
        COALESCE(pp.max_peer_zscore, 0) AS max_peer_zscore,
        COALESCE(pp.mean_peer_zscore, 0) AS mean_peer_zscore,
        -- NPI enrichment
        a.entity_type,
        DATE_DIFF('day', a.enumeration_date, CURRENT_DATE) / 365.25
            AS provider_age_years,
        CASE WHEN a.deactivation_date IS NOT NULL OR d.npi IS NOT NULL
             THEN 1 ELSE 0 END AS is_deactivated,
        CASE WHEN a.reactivation_date IS NOT NULL
             THEN 1 ELSE 0 END AS was_reactivated,
        CASE WHEN a.sole_proprietor = 'Y' THEN 1 ELSE 0 END
            AS is_sole_proprietor,
        CASE WHEN a.org_subpart = 'Y' THEN 1 ELSE 0 END
            AS is_org_subpart,
        a.num_taxonomy_codes,
        a.primary_taxonomy
    FROM provider_agg a
    LEFT JOIN provider_top_code tc ON a.billing_npi = tc.billing_npi
    LEFT JOIN provider_monthly_stats ms ON a.billing_npi = ms.billing_npi
    LEFT JOIN provider_peer pp ON a.billing_npi = pp.billing_npi
    LEFT JOIN deactivated d ON a.billing_npi = d.npi
    """

    print("Executing query...")
    result = conn.sql(query)

    print(f"Writing to {OUTPUT_PARQUET}...")
    result.write_parquet(OUTPUT_PARQUET)

    # --- Step 3: Summary stats ---
    print(f"\nReading back {OUTPUT_PARQUET} for summary...")
    df = conn.sql(f"SELECT * FROM '{OUTPUT_PARQUET}'").df()

    ca_npis_count = len(df)
    dropped = total_npis - ca_npis_count
    print(f"\n{'='*60}")
    print(f"  FEATURE ENGINEERING SUMMARY (CA only)")
    print(f"{'='*60}")
    print(f"  National billing NPIs:    {total_npis:>10,}")
    print(f"  CA providers (output):    {ca_npis_count:>10,}")
    print(f"  Dropped (non-CA / no NPPES): {dropped:>10,}")
    print(f"  Drop rate:                {100*dropped/total_npis:>9.1f}%")
    print()

    # Deactivated count
    deactivated_count = (df["is_deactivated"] == 1).sum()
    reactivated_count = (df["was_reactivated"] == 1).sum()
    print(f"  Deactivated providers:    {deactivated_count:>10,}")
    print(f"  Reactivated providers:    {reactivated_count:>10,}")
    print()

    # Entity type
    print("  Entity types:")
    for et, count in df["entity_type"].value_counts().items():
        print(f"    {et}: {count:,}")
    print()

    # Numeric feature distributions
    numeric_cols = [
        "total_paid", "total_claims", "total_beneficiaries",
        "num_hcpcs_codes", "num_active_months",
        "paid_per_claim", "claims_per_beneficiary", "paid_per_beneficiary",
        "pct_round_dollar", "pct_round_hundred", "pct_npi_mismatch",
        "revenue_concentration", "monthly_paid_cv", "max_monthly_paid",
        "max_peer_zscore", "mean_peer_zscore",
        "provider_age_years", "num_taxonomy_codes",
    ]
    print("  Feature distributions (p50 / mean / max):")
    for col in numeric_cols:
        s = df[col]
        med = s.median()
        avg = s.mean()
        mx = s.max()
        print(f"    {col:30s}  {med:>14,.2f}  {avg:>14,.2f}  {mx:>14,.2f}")

    print(f"\n  Output: {OUTPUT_PARQUET} ({ca_npis_count:,} rows)")
    print(f"{'='*60}")

    conn.close()
    return df


if __name__ == "__main__":
    build_features()
