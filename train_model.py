"""
Anomaly detection model training for CA Medicaid provider spending.

Two approaches:
  1. Statistical baseline — count features with |z-score| > 3
  2. Isolation Forest — unsupervised anomaly scoring

Input:  provider_features_ca.parquet (from feature_engineering.py)
Output: provider_scores_ca.parquet (same rows + anomaly scores + ranks)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

INPUT_PARQUET = "provider_features_ca.parquet"
OUTPUT_PARQUET = "provider_scores_ca.parquet"
RANDOM_SEED = 42

# Features to log-transform (heavily right-skewed, strictly positive)
LOG_FEATURES = [
    "total_paid", "total_claims", "total_beneficiaries",
    "paid_per_claim", "paid_per_beneficiary", "max_monthly_paid",
]

# All numeric features used by the models
MODEL_FEATURES = [
    # Volume (log-transformed versions)
    "total_paid", "total_claims", "total_beneficiaries",
    "num_hcpcs_codes", "num_active_months",
    # Intensity (some log-transformed)
    "paid_per_claim", "claims_per_beneficiary", "paid_per_beneficiary",
    # Billing patterns
    "pct_round_dollar", "pct_round_hundred", "pct_npi_mismatch",
    "revenue_concentration",
    # Volatility
    "monthly_paid_cv", "max_monthly_paid",
    # Peer comparison
    "max_peer_zscore", "mean_peer_zscore",
    # NPI enrichment
    "entity_type", "provider_age_years",
    "is_deactivated", "was_reactivated",
    "is_sole_proprietor", "is_org_subpart",
    "num_taxonomy_codes",
]


def preprocess(df):
    """Log-transform skewed features, fill NaN, standardize."""
    X = df[MODEL_FEATURES].copy()

    # Fill NaN in revenue_concentration (3,205 nulls — providers with
    # only one row where top_code_paid/total_paid produces NULL)
    X["revenue_concentration"] = X["revenue_concentration"].fillna(1.0)

    # Log-transform: log1p handles zeros gracefully
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col])

    # Standardize all features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_scaled, scaler


def statistical_baseline(X_scaled):
    """Count how many features have |z-score| > 3 for each provider."""
    extreme = (X_scaled.abs() > 3).sum(axis=1)
    return extreme


def isolation_forest(X_scaled):
    """Fit Isolation Forest and return anomaly scores.

    sklearn convention: decision_function returns values where
    lower = more anomalous. We negate so higher = more anomalous,
    which is more intuitive for ranking.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Negate so higher = more anomalous
    scores = -model.decision_function(X_scaled)
    return scores, model


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # --- Load ---
    print(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  {len(df):,} providers, {len(df.columns)} columns")

    # --- Preprocess ---
    print("Preprocessing: log-transform, fill NaN, standardize...")
    X_scaled, scaler = preprocess(df)
    print(f"  Model feature matrix: {X_scaled.shape}")

    # --- Statistical baseline ---
    print_section("STATISTICAL BASELINE (features with |z| > 3)")
    df["zscore_count"] = statistical_baseline(X_scaled)

    dist = df["zscore_count"].value_counts().sort_index()
    print("  Distribution of extreme-z-score counts:")
    for count, n_providers in dist.items():
        print(f"    {count} extreme features: {n_providers:>6,} providers")
    print(f"  Providers with >= 1 extreme feature: "
          f"{(df['zscore_count'] >= 1).sum():,}")
    print(f"  Providers with >= 3 extreme features: "
          f"{(df['zscore_count'] >= 3).sum():,}")

    # --- Isolation Forest ---
    print_section("ISOLATION FOREST (n_estimators=200)")
    df["if_anomaly_score"], model = isolation_forest(X_scaled)

    # Score distribution
    scores = df["if_anomaly_score"]
    print(f"  Score distribution (higher = more anomalous):")
    for p in [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]:
        print(f"    p{int(p*100):03d}: {scores.quantile(p):>8.4f}")

    # --- Rankings ---
    df["if_rank"] = df["if_anomaly_score"].rank(ascending=False).astype(int)
    df["zscore_rank"] = df["zscore_count"].rank(
        ascending=False, method="min"
    ).astype(int)

    # --- Compare top-50 overlap ---
    print_section("MODEL AGREEMENT (top 50)")
    top50_if = set(df.nsmallest(50, "if_rank")["billing_npi"])
    top50_zs = set(df.nsmallest(50, "zscore_rank")["billing_npi"])
    overlap = top50_if & top50_zs
    print(f"  Top-50 Isolation Forest ∩ Top-50 z-score: {len(overlap)} providers")

    # --- Top 20 by Isolation Forest ---
    print_section("TOP 20 ANOMALIES (Isolation Forest)")
    top20 = df.nsmallest(20, "if_rank")
    display_cols = [
        "billing_npi", "if_rank", "zscore_rank", "if_anomaly_score",
        "zscore_count", "total_paid", "total_claims",
        "paid_per_claim", "max_peer_zscore", "is_deactivated",
        "entity_type", "num_taxonomy_codes",
    ]
    print(top20[display_cols].to_string(index=False))

    # --- Top 20 by z-score count ---
    print_section("TOP 20 ANOMALIES (statistical baseline)")
    top20_zs = df.nsmallest(20, "zscore_rank")
    print(top20_zs[display_cols].to_string(index=False))

    # --- Feature contributions for top-10 IF anomalies ---
    print_section("FEATURE CONTRIBUTIONS (top 10 IF anomalies)")
    print("  Which features are most extreme (highest |z|) for each:")
    top10 = df.nsmallest(10, "if_rank")
    for _, row in top10.iterrows():
        npi = row["billing_npi"]
        z_vals = X_scaled.loc[row.name].abs().sort_values(ascending=False)
        top3 = z_vals.head(3)
        features_str = ", ".join(
            f"{feat}={z_vals[feat]:.1f}" for feat in top3.index
        )
        print(f"  {npi}  (IF rank {row['if_rank']:>3d}): {features_str}")

    # --- Dollar-weighted ranking ---
    df["dollar_weighted_score"] = df["if_anomaly_score"] * df["total_paid"]
    df["dollar_rank"] = df["dollar_weighted_score"].rank(
        ascending=False
    ).astype(int)

    print_section("TOP 20 ANOMALIES (dollar-weighted)")
    top20_dw = df.nsmallest(20, "dollar_rank")
    dw_cols = [
        "billing_npi", "dollar_rank", "if_rank", "if_anomaly_score",
        "total_paid", "paid_per_claim", "max_peer_zscore",
        "is_deactivated",
    ]
    print(top20_dw[dw_cols].to_string(index=False))

    # --- Save ---
    print(f"\nWriting {OUTPUT_PARQUET}...")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
