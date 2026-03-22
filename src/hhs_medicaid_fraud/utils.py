import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

def load_config(config_path: str = "../config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    with open(config_file) as f:
        return yaml.safe_load(f)

# Constants (deprecated, use load_config)
HH_TAXONOMY_PREFIXES = ["251E", "253Z", "251J"]
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
LOG_FEATURES = [
    "total_paid", "total_claims", "total_beneficiaries",
    "paid_per_claim", "paid_per_beneficiary", "max_monthly_paid",
]

def print_section(title: str) -> None:
    \"\"\"Print formatted section header.\"\"\"
    print(f"\\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def load_sample_duckdb(path: str, sample_size: int = 10000, seed: int = 42) -> Tuple[int, pd.DataFrame]:
    \"\"\"Load total count + random sample via DuckDB.\"\"\"
    conn = duckdb.connect()
    conn.sql(f"SELECT setseed({seed / 2**31:.10f})")
    count = conn.sql(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]
    df = conn.sql(f"SELECT * FROM '{path}' USING SAMPLE {sample_size}").df()
    conn.close()
    return int(count), df

def execute_query_to_parquet(query: str, output_path: str) -> None:
    \"\"\"Execute DuckDB query and write Parquet.\"\"\"
    conn = duckdb.connect()
    result = conn.sql(query)
    result.write_parquet(output_path)
    conn.close()

def preprocess_for_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    \"\"\"Log-transform, fill NaN, standardize for modeling.\"\"\"
    X = df[MODEL_FEATURES].copy()
    X["revenue_concentration"] = X["revenue_concentration"].fillna(1.0)
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler