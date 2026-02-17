"""
National Home Health Fraud Deep-Dive for Medicaid Providers.

Extracts HCPCS-level billing detail for home health providers across all
states/territories, computes HH-specific features and within-cohort peer
comparisons, and produces a detailed report on the most anomalous home
health providers nationally.

Home health taxonomy codes (from NUCC):
  251E00000X — Home Health Agency (primary)
  253Z00000X — In Home Supportive Care Agency
  251J00000X — Nursing Care Agency

Input:  medicaid-provider-spending.csv, NPPES registry
Output: hh_claims_national.parquet, hh_features_national.parquet, console report
"""

import os

import numpy as np
import pandas as pd
import duckdb

SPENDING_CSV = "medicaid-provider-spending.csv"
NPPES_CSV = "nppes/npidata_pfile_20050523-20260208.csv"
SCORES_PARQUET = "provider_scores_ca.parquet"
HH_CLAIMS_PARQUET = "hh_claims_national.parquet"
HH_CLAIMS_CA_PARQUET = "hh_claims_ca.parquet"
HH_FEATURES_PARQUET = "hh_features_national.parquet"

# Home health taxonomy prefixes
HH_TAXONOMY_PREFIXES = ["251E", "253Z", "251J"]

# HCPCS code categorization for home health
HH_CODE_CATEGORIES = {
    # Skilled Nursing
    "G0299": "skilled_nursing", "G0300": "skilled_nursing",
    "T1030": "skilled_nursing", "T1031": "skilled_nursing",
    "S9123": "skilled_nursing", "S9124": "skilled_nursing",
    # Home Health Aide
    "G0156": "aide", "T1020": "aide", "T1021": "aide", "S9122": "aide",
    # Personal Care
    "T1019": "personal_care",
    # Therapy
    "G0151": "therapy", "G0152": "therapy", "G0153": "therapy",
    "G0157": "therapy", "G0159": "therapy", "G0160": "therapy",
    # Assessment / Management
    "G0155": "assessment", "G0162": "assessment", "G0493": "assessment",
}

HH_CATEGORIES = [
    "skilled_nursing", "aide", "personal_care", "therapy", "assessment",
]


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def extract_hh_claims():
    """Extract HCPCS-level claims for HH providers nationally via DuckDB."""
    conn = duckdb.connect()

    taxonomy_filter = " OR ".join(
        f"\"Healthcare Provider Taxonomy Code_1\" LIKE '{p}%'"
        for p in HH_TAXONOMY_PREFIXES
    )

    query = f"""
    WITH hh_npis AS (
        SELECT
            CAST("NPI" AS VARCHAR) AS npi,
            "Healthcare Provider Taxonomy Code_1" AS primary_taxonomy,
            "Provider Business Practice Location Address State Name" AS provider_state,
            CASE WHEN CAST("NPI Deactivation Date" AS VARCHAR) IS NOT NULL
                      AND CAST("NPI Deactivation Date" AS VARCHAR) != ''
                 THEN 1 ELSE 0 END AS is_deactivated
        FROM read_csv('{NPPES_CSV}', header=true, quote='"')
        WHERE ({taxonomy_filter})
    )
    SELECT
        s.BILLING_PROVIDER_NPI_NUM AS billing_npi,
        s.SERVICING_PROVIDER_NPI_NUM AS servicing_npi,
        s.HCPCS_CODE,
        s.CLAIM_FROM_MONTH,
        s.TOTAL_UNIQUE_BENEFICIARIES,
        s.TOTAL_CLAIMS,
        s.TOTAL_PAID,
        h.primary_taxonomy,
        h.provider_state,
        h.is_deactivated
    FROM read_csv('{SPENDING_CSV}', header=true, auto_detect=true) s
    INNER JOIN hh_npis h ON s.BILLING_PROVIDER_NPI_NUM = h.npi
    """

    print("Extracting national HH claims from spending CSV (full 11 GB scan)...")
    result = conn.sql(query)
    result.write_parquet(HH_CLAIMS_PARQUET)

    df = pd.read_parquet(HH_CLAIMS_PARQUET)
    print(f"  Extracted {len(df):,} claim rows for {df['billing_npi'].nunique():,} "
          f"HH providers across {df['provider_state'].nunique()} states/territories")
    conn.close()
    return df


def categorize_codes(df):
    """Map HCPCS codes to HH categories."""
    df["hh_category"] = df["HCPCS_CODE"].map(HH_CODE_CATEGORIES).fillna("non_hh")
    return df


def compute_hh_features(claims):
    """Compute home-health-specific provider-level features."""
    # --- Code mix: % of total_paid by category ---
    cat_paid = claims.groupby(["billing_npi", "hh_category"])["TOTAL_PAID"].sum().unstack(
        fill_value=0
    )
    # Ensure all categories exist
    for cat in HH_CATEGORIES + ["non_hh"]:
        if cat not in cat_paid.columns:
            cat_paid[cat] = 0.0

    provider_total = cat_paid.sum(axis=1)
    code_mix = pd.DataFrame(index=cat_paid.index)
    for cat in HH_CATEGORIES + ["non_hh"]:
        code_mix[f"pct_{cat}"] = cat_paid[cat] / provider_total

    code_mix["nursing_to_aide_ratio"] = (
        cat_paid["skilled_nursing"] / (cat_paid["aide"] + 1)
    )

    # --- Provider-level aggregates ---
    prov = claims.groupby("billing_npi").agg(
        total_paid=("TOTAL_PAID", "sum"),
        total_claims=("TOTAL_CLAIMS", "sum"),
        total_beneficiaries=("TOTAL_UNIQUE_BENEFICIARIES", "sum"),
        num_hcpcs_codes=("HCPCS_CODE", "nunique"),
        num_active_months=("CLAIM_FROM_MONTH", "nunique"),
        num_rows=("HCPCS_CODE", "count"),
        primary_taxonomy=("primary_taxonomy", "first"),
        provider_state=("provider_state", "first"),
        is_deactivated=("is_deactivated", "first"),
    )
    prov["paid_per_claim"] = prov["total_paid"] / prov["total_claims"]
    prov["claims_per_beneficiary"] = prov["total_claims"] / prov["total_beneficiaries"]
    prov["paid_per_beneficiary"] = prov["total_paid"] / prov["total_beneficiaries"]

    # Count distinct HH-category codes (exclude non_hh)
    hh_only = claims[claims["hh_category"] != "non_hh"]
    hh_code_counts = hh_only.groupby("billing_npi")["HCPCS_CODE"].nunique().rename("num_hh_codes")

    # Monthly CV
    monthly = claims.groupby(["billing_npi", "CLAIM_FROM_MONTH"])["TOTAL_PAID"].sum()
    monthly_stats = monthly.groupby("billing_npi").agg(["mean", "std"])
    monthly_stats["monthly_paid_cv"] = monthly_stats["std"] / monthly_stats["mean"].replace(0, np.nan)
    monthly_stats["monthly_paid_cv"] = monthly_stats["monthly_paid_cv"].fillna(0)

    # --- Per-code peer z-scores within HH cohort ---
    code_agg = claims.groupby(["billing_npi", "HCPCS_CODE"]).agg(
        code_paid=("TOTAL_PAID", "sum"),
        code_claims=("TOTAL_CLAIMS", "sum"),
    )
    code_agg["ppc"] = code_agg["code_paid"] / code_agg["code_claims"]

    # Peer stats per HCPCS (HH providers only)
    peer = code_agg.reset_index().groupby("HCPCS_CODE").agg(
        avg_ppc=("ppc", "mean"),
        std_ppc=("ppc", "std"),
        num_providers=("billing_npi", "nunique"),
    )
    peer = peer[peer["num_providers"] >= 3]  # need 3+ peers

    # Z-scores
    code_agg = code_agg.reset_index()
    code_agg = code_agg.merge(peer, on="HCPCS_CODE", how="left")
    code_agg["peer_zscore"] = np.where(
        (code_agg["std_ppc"].notna()) & (code_agg["std_ppc"] > 0),
        (code_agg["ppc"] - code_agg["avg_ppc"]) / code_agg["std_ppc"],
        0,
    )

    peer_rollup = code_agg.groupby("billing_npi").agg(
        hh_max_peer_zscore=("peer_zscore", "max"),
        hh_mean_peer_zscore=("peer_zscore", lambda x: np.average(
            x, weights=code_agg.loc[x.index, "code_paid"]
        ) if code_agg.loc[x.index, "code_paid"].sum() > 0 else 0),
    )

    # --- Assemble ---
    features = prov.join(code_mix).join(hh_code_counts).join(
        monthly_stats[["monthly_paid_cv"]]
    ).join(peer_rollup)
    features["num_hh_codes"] = features["num_hh_codes"].fillna(0).astype(int)

    return features, code_agg


def compute_risk_score(features):
    """Composite HH risk score from z-scored features."""
    score_cols = [
        "hh_max_peer_zscore", "hh_mean_peer_zscore",
        "nursing_to_aide_ratio", "claims_per_beneficiary",
        "paid_per_beneficiary", "pct_non_hh", "monthly_paid_cv",
    ]
    from sklearn.preprocessing import StandardScaler
    X = features[score_cols].copy().fillna(0)
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), columns=score_cols, index=X.index)

    # Extreme feature count (|z| > 2 for smaller cohort)
    features["hh_extreme_count"] = (Z.abs() > 2).sum(axis=1)

    # Weighted composite
    features["hh_risk_score"] = (
        Z["hh_max_peer_zscore"] * 1.0
        + Z["hh_mean_peer_zscore"] * 0.5
        + Z["nursing_to_aide_ratio"] * 0.3
        + Z["claims_per_beneficiary"] * 0.2
        + Z["paid_per_beneficiary"] * 0.2
        + Z["pct_non_hh"] * 0.1
    )
    features["hh_risk_rank"] = features["hh_risk_score"].rank(ascending=False).astype(int)
    return features, Z


def main():
    # === STEP 1: Extract HH claims ===
    claims = extract_hh_claims()
    claims = categorize_codes(claims)

    # === STEP 2: Compute features ===
    print("Computing HH-specific features...")
    features, code_detail = compute_hh_features(claims)
    features, Z = compute_risk_score(features)

    n_providers = len(features)
    n_states = features["provider_state"].nunique()
    print(f"  {n_providers:,} HH providers across {n_states} states/territories "
          f"with features computed")

    # ================================================================
    # REPORT
    # ================================================================

    # --- 1. Cohort Overview ---
    section("1. HOME HEALTH COHORT OVERVIEW (NATIONAL)")
    taxonomy_names = {
        "251E": "Home Health Agency",
        "253Z": "In Home Supportive Care",
        "251J": "Nursing Care Agency",
    }
    features["taxonomy_type"] = features["primary_taxonomy"].str[:4].map(taxonomy_names)
    for ttype, group in features.groupby("taxonomy_type"):
        print(f"  {ttype}: {len(group):>6,} providers, "
              f"${group['total_paid'].sum():>14,.0f} total paid")
    print(f"\n  TOTAL: {n_providers:,} providers across {n_states} states/territories, "
          f"${features['total_paid'].sum():,.0f} total Medicaid spending")
    print(f"  {claims['HCPCS_CODE'].nunique():,} distinct HCPCS codes billed")

    # Top 10 states by provider count
    state_counts = features["provider_state"].value_counts().head(10)
    print(f"\n  Top 10 states by HH provider count:")
    for st, cnt in state_counts.items():
        st_paid = features[features["provider_state"] == st]["total_paid"].sum()
        print(f"    {st:<4s} {cnt:>6,} providers  ${st_paid:>14,.0f}")

    # --- 2. Top HCPCS Codes ---
    section("2. TOP 20 HCPCS CODES (by total paid across HH cohort)")
    code_summary = claims.groupby("HCPCS_CODE").agg(
        total_paid=("TOTAL_PAID", "sum"),
        total_claims=("TOTAL_CLAIMS", "sum"),
        num_providers=("billing_npi", "nunique"),
        category=("hh_category", "first"),
    ).sort_values("total_paid", ascending=False)
    code_summary["avg_ppc"] = code_summary["total_paid"] / code_summary["total_claims"]

    print(f"\n  {'Code':<8s} {'Category':<17s} {'Providers':>9s} {'Total Paid':>14s} "
          f"{'Avg $/Claim':>12s}")
    print(f"  {'-'*62}")
    for code, r in code_summary.head(20).iterrows():
        print(f"  {code:<8s} {r['category']:<17s} {r['num_providers']:>9,} "
              f"${r['total_paid']:>13,.0f} ${r['avg_ppc']:>11,.2f}")

    # --- 3. Code Mix Analysis ---
    section("3. CODE MIX DISTRIBUTION (% of revenue by category)")
    mix_cols = [f"pct_{c}" for c in HH_CATEGORIES + ["non_hh"]]
    print(f"\n  {'Category':<18s} {'Median':>8s} {'Mean':>8s} {'p75':>8s} {'p95':>8s} {'Max':>8s}")
    print(f"  {'-'*52}")
    for col in mix_cols:
        s = features[col]
        cat_name = col[4:]  # strip "pct_"
        print(f"  {cat_name:<18s} {s.median():>7.0%} {s.mean():>7.0%} "
              f"{s.quantile(0.75):>7.0%} {s.quantile(0.95):>7.0%} {s.max():>7.0%}")

    print(f"\n  Nursing-to-aide ratio:")
    s = features["nursing_to_aide_ratio"]
    print(f"    Median: {s.median():>10,.1f}")
    print(f"    Mean:   {s.mean():>10,.1f}")
    print(f"    p95:    {s.quantile(0.95):>10,.1f}")
    print(f"    Max:    {s.max():>10,.1f}")

    # --- 4. Anomalous Pricing ---
    section("4. ANOMALOUS PRICING (highest per-code peer z-scores)")
    print("  Providers charging far more than HH peers for the same code:\n")

    # Add state to code_detail for display
    code_detail_st = code_detail.merge(
        features[["provider_state"]],
        left_on="billing_npi", right_index=True, how="left",
    )

    top_pricing = code_detail_st[code_detail_st["peer_zscore"] > 3].sort_values(
        "peer_zscore", ascending=False
    ).head(20)
    print(f"  {'NPI':<12s} {'St':<4s} {'Code':<8s} {'$/Claim':>10s} {'Peer Avg':>10s} "
          f"{'Z-Score':>8s} {'Paid':>14s}")
    print(f"  {'-'*68}")
    for _, r in top_pricing.iterrows():
        print(f"  {r['billing_npi']:<12s} {str(r.get('provider_state','')):<4s} "
              f"{r['HCPCS_CODE']:<8s} "
              f"${r['ppc']:>9,.2f} ${r['avg_ppc']:>9,.2f} "
              f"{r['peer_zscore']:>8.1f} ${r['code_paid']:>13,.0f}")

    # --- 5. Upcoding Signals ---
    section("5. UPCODING SIGNALS (extreme nursing-to-aide ratios)")
    print("  Providers with very high skilled nursing billing relative to aide billing")
    print("  (possible upcoding of aide services as skilled nursing):\n")

    # Only look at providers who bill BOTH categories
    has_nursing = features["pct_skilled_nursing"] > 0
    has_aide = features["pct_aide"] > 0
    both = features[has_nursing & has_aide].sort_values(
        "nursing_to_aide_ratio", ascending=False
    )
    print(f"  Providers billing both nursing and aide: {len(both):,}")
    print(f"  Providers billing nursing only (no aide): {(has_nursing & ~has_aide).sum():,}")
    print(f"  Providers billing aide only (no nursing): {(~has_nursing & has_aide).sum():,}")
    print(f"  Providers billing neither: {(~has_nursing & ~has_aide).sum():,}")

    if len(both) > 0:
        print(f"\n  Top 15 by nursing-to-aide ratio (among those billing both):\n")
        print(f"  {'NPI':<12s} {'St':<4s} {'Ratio':>8s} {'%Nurse':>7s} {'%Aide':>7s} "
              f"{'Total Paid':>14s} {'$/Claim':>10s} {'HH Risk':>8s}")
        print(f"  {'-'*72}")
        for _, r in both.head(15).iterrows():
            print(f"  {r.name:<12s} {str(r['provider_state']):<4s} "
                  f"{r['nursing_to_aide_ratio']:>8,.1f} "
                  f"{r['pct_skilled_nursing']:>6.0%} {r['pct_aide']:>6.0%} "
                  f"${r['total_paid']:>13,.0f} ${r['paid_per_claim']:>9,.2f} "
                  f"{r['hh_risk_rank']:>8d}")

    # --- 6. Top 30 Most Anomalous HH Providers ---
    section("6. TOP 30 HOME HEALTH ANOMALIES (NATIONAL, by composite risk score)")
    top30 = features.nsmallest(30, "hh_risk_rank")
    print(f"\n  {'Rank':>4s}  {'NPI':<12s} {'St':<4s} {'Type':<12s} {'Paid($M)':>10s} "
          f"{'$/Claim':>10s} {'PeerZ':>6s} {'%Nurse':>6s} {'%Aide':>6s} "
          f"{'N:A':>7s} {'%NonHH':>6s} {'Extr':>4s}")
    print(f"  {'-'*89}")
    for _, r in top30.iterrows():
        print(f"  {r['hh_risk_rank']:>4d}  {r.name:<12s} "
              f"{str(r['provider_state']):<4s} "
              f"{str(r.get('taxonomy_type','?')):<12s} "
              f"{r['total_paid']/1e6:>10.1f} "
              f"${r['paid_per_claim']:>9,.2f} "
              f"{r['hh_max_peer_zscore']:>6.1f} "
              f"{r['pct_skilled_nursing']:>5.0%} "
              f"{r['pct_aide']:>5.0%} "
              f"{r['nursing_to_aide_ratio']:>7,.1f} "
              f"{r['pct_non_hh']:>5.0%} "
              f"{r['hh_extreme_count']:>4.0f}")

    # --- 7. Cross-Reference with General Model (CA only) ---
    section("7. CROSS-REFERENCE: HH ANOMALIES vs GENERAL IF MODEL (CA only)")
    print("  Note: General IF model scores only available for CA providers.\n")
    ca_features = features[features["provider_state"] == "CA"]

    if os.path.exists(SCORES_PARQUET):
        scores = pd.read_parquet(SCORES_PARQUET)
        general = scores[["billing_npi", "if_rank", "if_anomaly_score", "zscore_count",
                           "provider_age_years", "entity_type"]].copy()
        general = general.set_index("billing_npi")

        ca_with_gen = ca_features.join(general, how="left")

        hh_top50 = set(features.nsmallest(50, "hh_risk_rank").index)
        gen_top200 = set(scores.nsmallest(200, "if_rank")["billing_npi"])
        overlap = hh_top50 & gen_top200
        print(f"  National HH top-50 ∩ General IF top-200: {len(overlap)} providers")
        if overlap:
            print(f"  Overlapping NPIs:")
            for npi in sorted(overlap):
                r = ca_with_gen.loc[npi] if npi in ca_with_gen.index else features.loc[npi]
                if_rank_str = f"{int(r['if_rank']):>5d}" if pd.notna(r.get("if_rank")) else "  N/A"
                print(f"    {npi}  HH rank: {r['hh_risk_rank']:>4d}, "
                      f"IF rank: {if_rank_str}, "
                      f"${r['total_paid']/1e6:.1f}M")

        # CA HH providers in general top-200
        hh_in_gen_top200 = ca_with_gen[ca_with_gen.index.isin(gen_top200)]
        print(f"\n  CA HH providers appearing in General IF top-200: "
              f"{len(hh_in_gen_top200)}")
    else:
        print(f"  {SCORES_PARQUET} not found — skipping cross-reference.")

    # --- 8. Spotlight: Suspicious HCPCS Patterns ---
    section("8. SPOTLIGHT: SUSPICIOUS HCPCS CODE PATTERNS")

    # Codes with very few providers but high $/claim
    rare_expensive = code_summary[
        (code_summary["num_providers"] <= 5) &
        (code_summary["total_paid"] > 100_000)
    ].sort_values("avg_ppc", ascending=False)

    if len(rare_expensive) > 0:
        print("  Codes billed by <=5 providers with >$100K total and high $/claim:\n")
        print(f"  {'Code':<8s} {'Category':<17s} {'#Prov':>5s} {'Total Paid':>14s} "
              f"{'Avg $/Claim':>12s}")
        print(f"  {'-'*58}")
        for code, r in rare_expensive.head(15).iterrows():
            print(f"  {code:<8s} {r['category']:<17s} {r['num_providers']:>5,} "
                  f"${r['total_paid']:>13,.0f} ${r['avg_ppc']:>11,.2f}")

    # Deep dive on T1020
    t1020_providers = claims[claims["HCPCS_CODE"] == "T1020"]
    if len(t1020_providers) > 0:
        print(f"\n  T1020 (Home Health Aide per visit) — Deep Dive:")
        t1020_by_npi = t1020_providers.groupby("billing_npi").agg(
            total_paid=("TOTAL_PAID", "sum"),
            total_claims=("TOTAL_CLAIMS", "sum"),
        )
        t1020_by_npi["ppc"] = t1020_by_npi["total_paid"] / t1020_by_npi["total_claims"]
        # Show top 15 by paid (national may have many)
        for npi, r in t1020_by_npi.sort_values("total_paid", ascending=False).head(15).iterrows():
            hh_rank = features.loc[npi, "hh_risk_rank"] if npi in features.index else "?"
            st = features.loc[npi, "provider_state"] if npi in features.index else "?"
            print(f"    {npi} ({st}): ${r['total_paid']:>12,.0f} paid, "
                  f"{r['total_claims']:>8,.0f} claims, "
                  f"${r['ppc']:>8,.2f}/claim, HH rank: {hh_rank}")

    # --- 9. Deactivated HH Providers ---
    section("9. DEACTIVATED HOME HEALTH PROVIDERS (NATIONAL)")
    deact = features[features["is_deactivated"] == 1]
    print(f"  Deactivated HH providers: {len(deact):,}")
    if len(deact) > 0:
        print(f"  Total spending: ${deact['total_paid'].sum():,.0f}")
        print(f"\n  Top 20 by spending:\n")
        print(f"  {'NPI':<12s} {'St':<4s} {'Total Paid':>14s} {'HH Rank':>8s} "
              f"{'Peer Z':>7s}")
        print(f"  {'-'*48}")
        for npi, r in deact.sort_values("total_paid", ascending=False).head(20).iterrows():
            print(f"  {npi:<12s} {str(r['provider_state']):<4s} "
                  f"${r['total_paid']:>13,.0f} "
                  f"{r['hh_risk_rank']:>8d} "
                  f"{r['hh_max_peer_zscore']:>7.1f}")

    # --- 10. Summary ---
    section("10. SUMMARY")
    top30_paid = top30["total_paid"].sum()
    total_paid = features["total_paid"].sum()
    print(f"  Home health cohort: {n_providers:,} providers across {n_states} "
          f"states/territories, ${total_paid:,.0f} total Medicaid spending")
    print(f"  Top-30 anomalies: ${top30_paid:,.0f} "
          f"({100*top30_paid/total_paid:.1f}% of HH spending)")
    print(f"  Providers with >= 2 extreme features: "
          f"{(features['hh_extreme_count'] >= 2).sum():,}")
    print(f"  Median paid_per_claim (HH cohort): "
          f"${features['paid_per_claim'].median():,.2f}")
    print(f"  Median HH max peer z-score: "
          f"{features['hh_max_peer_zscore'].median():.2f}")

    # --- 11. State-by-State Anomaly Concentration ---
    section("11. STATE-BY-STATE ANOMALY CONCENTRATION")
    top100_npis = set(features.nsmallest(100, "hh_risk_rank").index)
    state_stats = []
    for st, grp in features.groupby("provider_state"):
        n_total = len(grp)
        n_in_top100 = grp.index.isin(top100_npis).sum()
        top100_spending = grp[grp.index.isin(top100_npis)]["total_paid"].sum()
        st_total_spending = grp["total_paid"].sum()
        state_stats.append({
            "state": st,
            "n_providers": n_total,
            "n_in_top100": n_in_top100,
            "concentration_pct": 100 * n_in_top100 / n_total if n_total > 0 else 0,
            "top100_spending": top100_spending,
            "spending_exposure_pct": 100 * top100_spending / st_total_spending if st_total_spending > 0 else 0,
            "mean_risk_score": grp["hh_risk_score"].mean(),
            "total_spending": st_total_spending,
        })
    state_df = pd.DataFrame(state_stats).sort_values("concentration_pct", ascending=False)

    print(f"\n  {'St':<4s} {'Provs':>6s} {'#Top100':>7s} {'Conc%':>6s} "
          f"{'$Exposure%':>10s} {'MeanRisk':>9s} {'Total $M':>10s}")
    print(f"  {'-'*56}")
    for _, r in state_df.iterrows():
        print(f"  {r['state']:<4s} {r['n_providers']:>6,} {r['n_in_top100']:>7d} "
              f"{r['concentration_pct']:>5.1f}% "
              f"{r['spending_exposure_pct']:>9.1f}% "
              f"{r['mean_risk_score']:>9.2f} "
              f"{r['total_spending']/1e6:>10.1f}")

    # --- 12. CA vs National Comparison ---
    section("12. CA vs NATIONAL COMPARISON")
    if os.path.exists(HH_CLAIMS_CA_PARQUET):
        print("  Loading CA-only claims from previous run for comparison...\n")
        ca_claims = pd.read_parquet(HH_CLAIMS_CA_PARQUET)
        ca_claims = categorize_codes(ca_claims)

        # Need to add provider_state and is_deactivated if missing from old CA parquet
        if "provider_state" not in ca_claims.columns:
            ca_claims["provider_state"] = "CA"
        if "is_deactivated" not in ca_claims.columns:
            ca_claims["is_deactivated"] = 0

        ca_features_local, _ = compute_hh_features(ca_claims)
        ca_features_local, _ = compute_risk_score(ca_features_local)

        # Match CA providers: get their CA-only rank and national rank
        ca_npis = set(ca_features_local.index)
        national_ca = features[features.index.isin(ca_npis)].copy()

        print(f"  CA providers in CA-only run: {len(ca_features_local):,}")
        print(f"  CA providers matched in national run: {len(national_ca):,}")

        # How many CA providers in national top-30 / top-100
        nat_top30_npis = set(features.nsmallest(30, "hh_risk_rank").index)
        nat_top100_npis = set(features.nsmallest(100, "hh_risk_rank").index)
        ca_in_nat_top30 = ca_npis & nat_top30_npis
        ca_in_nat_top100 = ca_npis & nat_top100_npis
        print(f"  CA providers in national top-30: {len(ca_in_nat_top30)}")
        print(f"  CA providers in national top-100: {len(ca_in_nat_top100)}")

        # Top 20 CA providers by national rank
        comparison = national_ca[["hh_risk_rank", "hh_risk_score", "total_paid",
                                   "provider_state"]].copy()
        comparison = comparison.rename(columns={
            "hh_risk_rank": "nat_rank", "hh_risk_score": "nat_score"
        })
        comparison["ca_rank"] = ca_features_local.loc[comparison.index, "hh_risk_rank"]
        comparison["ca_score"] = ca_features_local.loc[comparison.index, "hh_risk_score"]
        comparison["rank_delta"] = comparison["ca_rank"] - comparison["nat_rank"]

        print(f"\n  Top 20 CA providers by national rank:\n")
        print(f"  {'NPI':<12s} {'NatRank':>8s} {'CARank':>8s} {'Delta':>7s} "
              f"{'NatScore':>9s} {'CAScore':>9s} {'Paid($M)':>10s}")
        print(f"  {'-'*66}")
        for npi, r in comparison.sort_values("nat_rank").head(20).iterrows():
            print(f"  {npi:<12s} {r['nat_rank']:>8d} {r['ca_rank']:>8d} "
                  f"{r['rank_delta']:>+7d} "
                  f"{r['nat_score']:>9.2f} {r['ca_score']:>9.2f} "
                  f"{r['total_paid']/1e6:>10.1f}")

        # Biggest rank movers
        print(f"\n  Biggest rank movers (CA rank → national rank):")
        print(f"  Climbers (more anomalous nationally):")
        climbers = comparison.nlargest(10, "rank_delta")
        for npi, r in climbers.iterrows():
            print(f"    {npi}: CA #{r['ca_rank']:>4d} → Nat #{r['nat_rank']:>4d} "
                  f"(Δ {r['rank_delta']:>+d})")
        print(f"  Fallers (less anomalous nationally):")
        fallers = comparison.nsmallest(10, "rank_delta")
        for npi, r in fallers.iterrows():
            print(f"    {npi}: CA #{r['ca_rank']:>4d} → Nat #{r['nat_rank']:>4d} "
                  f"(Δ {r['rank_delta']:>+d})")
    else:
        print(f"  {HH_CLAIMS_CA_PARQUET} not found — skipping CA comparison.")

    # --- 13. CA Stability Check ---
    section("13. CA STABILITY CHECK")
    if os.path.exists(HH_CLAIMS_CA_PARQUET) and 'ca_features_local' in dir():
        ca_top30 = ca_features_local.nsmallest(30, "hh_risk_rank")
        print("  For each CA-only top-30 provider, showing national rank:\n")
        print(f"  {'NPI':<12s} {'CA Rank':>8s} {'Nat Rank':>9s} {'In Top30?':>10s} "
              f"{'In Top100?':>10s} {'Paid($M)':>10s}")
        print(f"  {'-'*62}")

        still_top30 = 0
        still_top100 = 0
        for npi, ca_r in ca_top30.iterrows():
            if npi in features.index:
                nat_rank = features.loc[npi, "hh_risk_rank"]
                in_30 = "YES" if npi in nat_top30_npis else "no"
                in_100 = "YES" if npi in nat_top100_npis else "no"
                paid = features.loc[npi, "total_paid"]
                if npi in nat_top30_npis:
                    still_top30 += 1
                if npi in nat_top100_npis:
                    still_top100 += 1
            else:
                nat_rank = "N/A"
                in_30 = "N/A"
                in_100 = "N/A"
                paid = ca_r["total_paid"]

            nat_rank_str = f"{nat_rank:>9d}" if isinstance(nat_rank, (int, np.integer)) else f"{nat_rank:>9s}"
            print(f"  {npi:<12s} {ca_r['hh_risk_rank']:>8d} {nat_rank_str} "
                  f"{in_30:>10s} {in_100:>10s} "
                  f"{paid/1e6:>10.1f}")

        print(f"\n  Stability: {still_top30}/30 CA top-30 remain in national top-30")
        print(f"  Stability: {still_top100}/30 CA top-30 remain in national top-100")
    else:
        print(f"  {HH_CLAIMS_CA_PARQUET} not found — skipping stability check.")

    # --- Save national features ---
    features.to_parquet(HH_FEATURES_PARQUET)
    print(f"\n  Saved {len(features):,} provider features to {HH_FEATURES_PARQUET}")

    print(f"\n{'='*70}")
    print("  National analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
