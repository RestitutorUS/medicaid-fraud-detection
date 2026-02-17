"""
Analyze anomaly detection results for CA Medicaid providers.

Reads provider_scores_ca.parquet and produces:
  1. Score distribution summary
  2. Top 50 anomalies (IF, z-score, dollar-weighted) with feature profiles
  3. Feature contribution analysis for top anomalies
  4. Model agreement analysis
  5. Deactivated provider deep-dive
  6. Specialty (taxonomy) breakdown of top anomalies
  7. Exclusion/coverage report

Input:  provider_scores_ca.parquet (from train_model.py)
        provider_features_ca.parquet (for re-computing z-scores)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCORES_PARQUET = "provider_scores_ca.parquet"
FEATURES_PARQUET = "provider_features_ca.parquet"

# Same preprocessing as train_model.py to get z-scores per feature
LOG_FEATURES = [
    "total_paid", "total_claims", "total_beneficiaries",
    "paid_per_claim", "paid_per_beneficiary", "max_monthly_paid",
]
MODEL_FEATURES = [
    "total_paid", "total_claims", "total_beneficiaries",
    "num_hcpcs_codes", "num_active_months",
    "paid_per_claim", "claims_per_beneficiary", "paid_per_beneficiary",
    "pct_round_dollar", "pct_round_hundred", "pct_npi_mismatch",
    "revenue_concentration",
    "monthly_paid_cv", "max_monthly_paid",
    "max_peer_zscore", "mean_peer_zscore",
    "entity_type", "provider_age_years",
    "is_deactivated", "was_reactivated",
    "is_sole_proprietor", "is_org_subpart",
    "num_taxonomy_codes",
]


def compute_zscores(df):
    """Reproduce the z-score matrix from train_model.py preprocessing."""
    X = df[MODEL_FEATURES].copy()
    X["revenue_concentration"] = X["revenue_concentration"].fillna(1.0)
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col])
    scaler = StandardScaler()
    Z = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )
    return Z


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    df = pd.read_parquet(SCORES_PARQUET)
    Z = compute_zscores(df)
    print(f"Loaded {len(df):,} scored CA providers.")

    # ================================================================
    # 1. SCORE DISTRIBUTIONS
    # ================================================================
    section("1. SCORE DISTRIBUTIONS")

    print("\n  Isolation Forest anomaly score:")
    for p in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]:
        v = df["if_anomaly_score"].quantile(p)
        print(f"    p{int(p*100):03d}: {v:>9.4f}")
    print(f"    mean: {df['if_anomaly_score'].mean():>9.4f}")
    print(f"    std:  {df['if_anomaly_score'].std():>9.4f}")

    print("\n  Z-score count (features with |z| > 3):")
    dist = df["zscore_count"].value_counts().sort_index()
    for cnt, n in dist.items():
        pct = 100 * n / len(df)
        bar = "#" * max(1, int(pct / 2))
        print(f"    {cnt} extreme: {n:>6,} ({pct:>5.1f}%) {bar}")

    # ================================================================
    # 2. MODEL AGREEMENT
    # ================================================================
    section("2. MODEL AGREEMENT")

    for k in [20, 50, 100, 200]:
        top_if = set(df.nsmallest(k, "if_rank")["billing_npi"])
        top_zs = set(df.nsmallest(k, "zscore_rank")["billing_npi"])
        top_dw = set(df.nsmallest(k, "dollar_rank")["billing_npi"])
        print(f"\n  Top-{k} overlap:")
        print(f"    IF ∩ Z-score:        {len(top_if & top_zs):>4d} "
              f"({100*len(top_if & top_zs)/k:.0f}%)")
        print(f"    IF ∩ Dollar-weighted: {len(top_if & top_dw):>4d} "
              f"({100*len(top_if & top_dw)/k:.0f}%)")
        print(f"    Z-score ∩ Dollar-wt:  {len(top_zs & top_dw):>4d} "
              f"({100*len(top_zs & top_dw)/k:.0f}%)")
        all_three = top_if & top_zs & top_dw
        print(f"    All three:            {len(all_three):>4d} "
              f"({100*len(all_three)/k:.0f}%)")

    # ================================================================
    # 3. TOP 50 ANOMALIES — ISOLATION FOREST
    # ================================================================
    section("3. TOP 50 ANOMALIES (Isolation Forest)")

    top50 = df.nsmallest(50, "if_rank").copy()
    top50["total_paid_M"] = top50["total_paid"] / 1e6
    cols = [
        "if_rank", "billing_npi", "total_paid_M",
        "paid_per_claim", "max_peer_zscore", "num_hcpcs_codes",
        "num_taxonomy_codes", "pct_round_dollar",
        "monthly_paid_cv", "is_deactivated", "entity_type",
        "zscore_rank", "dollar_rank",
    ]
    fmt = {
        "total_paid_M": "{:>10,.1f}",
        "paid_per_claim": "{:>10,.2f}",
        "max_peer_zscore": "{:>8,.1f}",
        "pct_round_dollar": "{:>6.0%}",
        "monthly_paid_cv": "{:>6.2f}",
    }
    # Print header
    header = (f"{'Rank':>4s}  {'NPI':>10s}  {'Paid($M)':>10s}  "
              f"{'$/Claim':>10s}  {'PeerZ':>8s}  {'#Codes':>6s}  "
              f"{'#Tax':>4s}  {'Rnd$':>6s}  {'MoCV':>6s}  "
              f"{'Deact':>5s}  {'Type':>4s}  {'ZRank':>6s}  {'$Rank':>6s}")
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")
    for _, r in top50.iterrows():
        line = (f"  {r['if_rank']:>4d}  {r['billing_npi']:>10s}  "
                f"{r['total_paid_M']:>10,.1f}  "
                f"{r['paid_per_claim']:>10,.2f}  "
                f"{r['max_peer_zscore']:>8,.1f}  "
                f"{r['num_hcpcs_codes']:>6d}  "
                f"{r['num_taxonomy_codes']:>4d}  "
                f"{r['pct_round_dollar']:>5.0%}  "
                f"{r['monthly_paid_cv']:>6.2f}  "
                f"{r['is_deactivated']:>5d}  "
                f"{r['entity_type']:>4d}  "
                f"{r['zscore_rank']:>6d}  "
                f"{r['dollar_rank']:>6d}")
        print(line)

    # ================================================================
    # 4. FEATURE CONTRIBUTIONS (top 20 IF)
    # ================================================================
    section("4. FEATURE CONTRIBUTIONS (top 20 IF anomalies)")
    print("  Top 5 most extreme features (by |z-score|) for each:\n")

    top20 = df.nsmallest(20, "if_rank")
    for _, row in top20.iterrows():
        z_row = Z.loc[row.name].abs().sort_values(ascending=False).head(5)
        feats = []
        for feat in z_row.index:
            raw = df.loc[row.name, feat] if feat in df.columns else "?"
            feats.append(f"{feat}(z={z_row[feat]:.1f}, raw={raw})")
        npi = row["billing_npi"]
        rank = row["if_rank"]
        paid = row["total_paid"] / 1e6
        print(f"  #{rank:<3d} {npi}  ${paid:>10,.1f}M")
        for f in feats:
            print(f"        {f}")
        print()

    # ================================================================
    # 5. DEACTIVATED PROVIDERS
    # ================================================================
    section("5. DEACTIVATED PROVIDERS WITH MEDI-CAL SPENDING")

    deact = df[df["is_deactivated"] == 1].sort_values(
        "total_paid", ascending=False
    )
    print(f"  Total deactivated providers with spending: {len(deact):,}")
    print(f"  Total spending by deactivated providers: "
          f"${deact['total_paid'].sum():,.0f}")
    print(f"  Also reactivated: {deact['was_reactivated'].sum():,}")

    if len(deact) > 0:
        print(f"\n  Top 20 deactivated providers by total paid:\n")
        dcols = [
            "billing_npi", "total_paid", "total_claims",
            "paid_per_claim", "max_peer_zscore", "num_hcpcs_codes",
            "if_rank", "zscore_rank", "provider_age_years",
            "num_taxonomy_codes",
        ]
        header = (f"  {'NPI':>10s}  {'Total Paid':>14s}  {'Claims':>10s}  "
                  f"{'$/Claim':>10s}  {'PeerZ':>6s}  {'#Codes':>6s}  "
                  f"{'IFRank':>6s}  {'ZRank':>6s}  {'Age':>5s}  {'#Tax':>4s}")
        print(header)
        print(f"  {'-'*len(header.strip())}")
        for _, r in deact.head(20).iterrows():
            print(f"  {r['billing_npi']:>10s}  "
                  f"${r['total_paid']:>13,.0f}  "
                  f"{r['total_claims']:>10,.0f}  "
                  f"{r['paid_per_claim']:>10,.2f}  "
                  f"{r['max_peer_zscore']:>6.1f}  "
                  f"{r['num_hcpcs_codes']:>6d}  "
                  f"{r['if_rank']:>6d}  "
                  f"{r['zscore_rank']:>6d}  "
                  f"{r['provider_age_years']:>5.1f}  "
                  f"{r['num_taxonomy_codes']:>4d}")

    # ================================================================
    # 6. SPECIALTY BREAKDOWN OF TOP ANOMALIES
    # ================================================================
    section("6. SPECIALTY (TAXONOMY) BREAKDOWN")

    # Map first 3 chars → broad specialty group
    # Full taxonomy codes are like "207RC0000X" — first 3 chars give the group
    df["taxonomy_group"] = df["primary_taxonomy"].str[:3]

    print("\n  Top 15 taxonomy groups across ALL providers:")
    all_tax = df["taxonomy_group"].value_counts().head(15)
    for grp, cnt in all_tax.items():
        pct = 100 * cnt / len(df)
        print(f"    {grp}: {cnt:>6,} ({pct:>5.1f}%)")

    print("\n  Top 15 taxonomy groups in IF top-200 anomalies:")
    top200 = df.nsmallest(200, "if_rank")
    anom_tax = top200["taxonomy_group"].value_counts().head(15)
    for grp, cnt in anom_tax.items():
        base_pct = 100 * all_tax.get(grp, 0) / len(df)
        anom_pct = 100 * cnt / len(top200)
        ratio = anom_pct / base_pct if base_pct > 0 else float("inf")
        print(f"    {grp}: {cnt:>4,} ({anom_pct:>5.1f}%)  "
              f"[base {base_pct:.1f}%, ratio {ratio:.1f}x]")

    # ================================================================
    # 7. TOP 20 DOLLAR-WEIGHTED ANOMALIES
    # ================================================================
    section("7. TOP 20 DOLLAR-WEIGHTED ANOMALIES")
    print("  (Prioritized for investigators — highest anomaly × dollars)\n")

    top20_dw = df.nsmallest(20, "dollar_rank")
    header = (f"  {'$Rank':>5s}  {'NPI':>10s}  {'Paid($M)':>10s}  "
              f"{'IFScore':>8s}  {'IFRank':>6s}  {'$/Claim':>10s}  "
              f"{'PeerZ':>6s}  {'#Codes':>6s}  {'Deact':>5s}")
    print(header)
    print(f"  {'-'*len(header.strip())}")
    for _, r in top20_dw.iterrows():
        print(f"  {r['dollar_rank']:>5d}  {r['billing_npi']:>10s}  "
              f"{r['total_paid']/1e6:>10,.1f}  "
              f"{r['if_anomaly_score']:>8.4f}  "
              f"{r['if_rank']:>6d}  "
              f"{r['paid_per_claim']:>10,.2f}  "
              f"{r['max_peer_zscore']:>6.1f}  "
              f"{r['num_hcpcs_codes']:>6d}  "
              f"{r['is_deactivated']:>5d}")

    # ================================================================
    # 8. ENTITY TYPE COMPARISON
    # ================================================================
    section("8. ENTITY TYPE: INDIVIDUALS vs ORGANIZATIONS")

    for etype, label in [(1, "Individual"), (2, "Organization")]:
        subset = df[df["entity_type"] == etype]
        top200_subset = top200[top200["entity_type"] == etype]
        base_pct = 100 * len(subset) / len(df)
        anom_pct = 100 * len(top200_subset) / len(top200) if len(top200) > 0 else 0
        print(f"\n  {label}s (entity_type={etype}):")
        print(f"    Count:       {len(subset):>8,} ({base_pct:.1f}% of all)")
        print(f"    In top-200:  {len(top200_subset):>8,} ({anom_pct:.1f}% of top-200)")
        print(f"    Median paid: ${subset['total_paid'].median():>14,.0f}")
        print(f"    Mean paid:   ${subset['total_paid'].mean():>14,.0f}")
        print(f"    Median $/cl: ${subset['paid_per_claim'].median():>10,.2f}")
        print(f"    Median peer z: {subset['max_peer_zscore'].median():>8.2f}")

    # ================================================================
    # 9. COVERAGE REPORT
    # ================================================================
    section("9. COVERAGE & EXCLUSION REPORT")

    total_national = 617_503  # from feature_engineering.py output
    print(f"  National billing NPIs:       {total_national:>10,}")
    print(f"  CA providers scored:         {len(df):>10,}")
    print(f"  Excluded (non-CA/no NPPES):  {total_national - len(df):>10,}")
    print(f"  Coverage rate:               "
          f"{100*len(df)/total_national:>9.1f}%")
    print(f"\n  CA Medi-Cal total spending:   "
          f"${df['total_paid'].sum():>18,.0f}")
    print(f"  CA providers w/ IF rank ≤ 200: {len(top200):>8,}")
    print(f"  Their total spending:          "
          f"${top200['total_paid'].sum():>18,.0f} "
          f"({100*top200['total_paid'].sum()/df['total_paid'].sum():.1f}%"
          f" of CA total)")

    print(f"\n{'='*70}")
    print("  Analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
