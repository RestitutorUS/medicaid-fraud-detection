"""
NPI Laundering Analysis: Authorized Official Network Detection.

Identifies patterns where a single authorized official controls multiple
organizational NPIs — a potential indicator of NPI laundering, exclusion
evasion, or billing fragmentation.

Three analysis goals:
  1. Multi-NPI authorized officials (network size, geography, entity types)
  2. Cross-reference against OIG LEIE exclusion list (individuals + orgs)
     and deactivated NPI list
  3. Multi-state home health billing through shared officials

Input:  NPPES registry, deactivated NPI list, LEIE exclusion list,
        Medicaid spending CSV, hh_features_national.parquet
Output: ao_networks.parquet, laundering_flags.parquet, console report
"""

import argparse
import os
import urllib.request

import duckdb
import numpy as np
import pandas as pd

# --- Default file paths ---
DEFAULT_NPPES_CSV = "nppes/npidata_pfile_20050523-20260208.csv"
DEFAULT_DEACTIVATED_XLSX = "nppes_deactivated/NPPES Deactivated NPI Report 20260209.xlsx"
DEFAULT_SPENDING_CSV = "medicaid-provider-spending.csv"
DEFAULT_HH_FEATURES_PARQUET = "hh_features_national.parquet"

# LEIE: try download, fall back to local copy
DEFAULT_LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
DEFAULT_LEIE_DIR = "leie"
DEFAULT_LEIE_CSV = os.path.join(DEFAULT_LEIE_DIR, "UPDATED.csv")
DEFAULT_LEIE_FALLBACK = "leie_exclusions.csv"

# Output paths
DEFAULT_NETWORKS_PARQUET = "ao_networks.parquet"
DEFAULT_FLAGS_PARQUET = "laundering_flags.parquet"

# Home health taxonomy prefixes
HH_TAXONOMY_PREFIXES = ["251E", "253Z", "251J"]


from .utils import print_section


# =====================================================================
# Step 1: Load LEIE
# =====================================================================

def load_leie(leie_csv, leie_fallback):
    """Load OIG LEIE exclusion list. Download if needed, fall back to local."""
    leie_path = None

    # Try local download first
    if os.path.exists(leie_csv):
        leie_path = leie_csv
        print(f"  Using existing LEIE: {leie_csv}")
    else:
        # Try downloading
        try:
            os.makedirs(DEFAULT_LEIE_DIR, exist_ok=True)
            urllib.request.urlretrieve(DEFAULT_LEIE_URL, leie_csv)
            leie_path = leie_csv
            print(f"  Downloaded to {leie_csv}")
        except Exception as e:
            print(f"  Download failed: {e}")

    # Fall back to local copy
    if leie_path is None or not os.path.exists(leie_path):
        if os.path.exists(leie_fallback):
            leie_path = leie_fallback
            print(f"  Falling back to local copy: {leie_fallback}")
        else:
            print("  ERROR: No LEIE data available.")
            return pd.DataFrame()

    leie = pd.read_csv(leie_path, dtype=str).fillna("")
    print(f"  Loaded {len(leie):,} LEIE exclusion records")

    # Split into individuals and organizations
    leie["_has_name"] = leie["LASTNAME"].str.strip().str.len() > 0
    leie["_has_busname"] = leie["BUSNAME"].str.strip().str.len() > 0
    n_ind = leie["_has_name"].sum()
    n_org = (~leie["_has_name"] & leie["_has_busname"]).sum()
    n_both = (leie["_has_name"] & leie["_has_busname"]).sum()
    print(f"  LEIE breakdown: {n_ind:,} individuals, {n_org:,} orgs-only, "
          f"{n_both:,} with both name+busname")
    return leie


# =====================================================================
# Step 2: DuckDB Extraction
# =====================================================================

def extract_org_npis(nppes_csv):
    """Extract organizational NPIs with authorized officials from NPPES."""
    conn = duckdb.connect()

    query = f"""
    SELECT
        CAST("NPI" AS VARCHAR) AS npi,
        "Provider Organization Name (Legal Business Name)" AS org_name,
        "Provider Other Organization Name" AS org_dba,
        "Provider First Line Business Practice Location Address" AS address_line1,
        "Provider Business Practice Location Address City Name" AS city,
        "Provider Business Practice Location Address State Name" AS state,
        "Provider Business Practice Location Address Postal Code" AS zip5,
        CAST("Provider Enumeration Date" AS VARCHAR) AS enumeration_date,
        CAST("NPI Deactivation Date" AS VARCHAR) AS deactivation_date,
        CAST("NPI Reactivation Date" AS VARCHAR) AS reactivation_date,
        "Healthcare Provider Taxonomy Code_1" AS primary_taxonomy,
        "Is Organization Subpart" AS is_subpart,
        "Parent Organization LBN" AS parent_org_name,
        "Parent Organization TIN" AS parent_org_tin,
        UPPER(TRIM("Authorized Official Last Name")) AS ao_last,
        UPPER(TRIM("Authorized Official First Name")) AS ao_first,
        UPPER(TRIM(COALESCE("Authorized Official Middle Name", ''))) AS ao_middle,
        "Authorized Official Telephone Number" AS ao_phone
    FROM read_csv('{nppes_csv}', header=true, quote='"', all_varchar=true)
    WHERE "Entity Type Code" = '2'
      AND "Authorized Official Last Name" IS NOT NULL
      AND TRIM("Authorized Official Last Name") != ''
    """

    print("  Extracting organizational NPIs with authorized officials (NPPES scan)...")
    df = conn.sql(query).df()
    conn.close()

    # Derived flags
    df["is_deactivated"] = (df["deactivation_date"].fillna("").str.strip() != "").astype(int)
    df["is_reactivated"] = (df["reactivation_date"].fillna("").str.strip() != "").astype(int)
    df["is_hh"] = df["primary_taxonomy"].fillna("").str[:4].isin(HH_TAXONOMY_PREFIXES).astype(int)
    df["is_subpart_flag"] = (df["is_subpart"].fillna("").str.strip().str.upper() == "Y").astype(int)

    print(f"  Extracted {len(df):,} organizational NPIs with authorized officials")
    print(f"  States/territories: {df['state'].nunique()}")
    print(f"  Deactivated: {df['is_deactivated'].sum():,}, "
          f"Reactivated: {df['is_reactivated'].sum():,}, "
          f"HH taxonomy: {df['is_hh'].sum():,}")
    return df


def extract_deactivated_individuals(nppes_csv, deactivated_xlsx):
    """Extract deactivated individual providers from NPPES + deactivated list."""
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")

    query = f"""
    WITH deactivated_list AS (
        SELECT DISTINCT CAST("Field1" AS VARCHAR) AS npi
        FROM st_read('{deactivated_xlsx}')
        WHERE TRY_CAST("Field1" AS BIGINT) IS NOT NULL
    )
    SELECT
        CAST(n."NPI" AS VARCHAR) AS npi,
        UPPER(TRIM(n."Provider Last Name (Legal Name)")) AS ind_last,
        UPPER(TRIM(n."Provider First Name")) AS ind_first,
        n."Provider Business Practice Location Address State Name" AS ind_state,
        CAST(n."NPI Deactivation Date" AS VARCHAR) AS deactivation_date,
        n."Healthcare Provider Taxonomy Code_1" AS ind_taxonomy
    FROM read_csv('{nppes_csv}', header=true, quote='"', all_varchar=true) n
    LEFT JOIN deactivated_list d ON CAST(n."NPI" AS VARCHAR) = d.npi
    WHERE n."Entity Type Code" = '1'
      AND (
          (CAST(n."NPI Deactivation Date" AS VARCHAR) IS NOT NULL
           AND CAST(n."NPI Deactivation Date" AS VARCHAR) != '')
          OR d.npi IS NOT NULL
      )
    """

    print("  Extracting deactivated individual providers (NPPES scan)...")
    df = conn.sql(query).df()
    conn.close()
    print(f"  Extracted {len(df):,} deactivated individual providers")
    return df


def compute_name_frequencies(nppes_csv):
    """Count (last, first) name pairs among individual providers in NPPES.

    Returns a dict {(LAST, FIRST): count} used to penalize common-name
    matches and reduce false positives in LEIE cross-referencing.
    """
    conn = duckdb.connect()

    query = f"""
    SELECT
        UPPER(TRIM("Provider Last Name (Legal Name)")) AS last_name,
        UPPER(TRIM("Provider First Name")) AS first_name,
        COUNT(*) AS name_count
    FROM read_csv('{nppes_csv}', header=true, quote='"', all_varchar=true)
    WHERE "Entity Type Code" = '1'
      AND "Provider Last Name (Legal Name)" IS NOT NULL
      AND TRIM("Provider Last Name (Legal Name)") != ''
    GROUP BY 1, 2
    """

    print("  Computing name frequencies from NPPES individual providers...")
    df = conn.sql(query).df()
    conn.close()

    name_freq = dict(zip(
        zip(df["last_name"], df["first_name"]),
        df["name_count"].astype(int),
    ))
    total_names = len(name_freq)
    total_providers = df["name_count"].sum()
    print(f"  {total_names:,} unique (last, first) name pairs "
          f"across {total_names:,} individual providers")
    return name_freq


# =====================================================================
# Step 3: Build Authorized Official Networks
# =====================================================================

def build_ao_networks(org_npis):
    """Group org NPIs by authorized official, compute network features."""
    print("  Building authorized official networks...")

    # --- Address clustering: for each NPI, create normalized address key ---
    org_npis["addr_key"] = (
        org_npis["address_line1"].fillna("").str.upper().str.strip() + "|" +
        org_npis["city"].fillna("").str.upper().str.strip() + "|" +
        org_npis["state"].fillna("").str.strip() + "|" +
        org_npis["zip5"].fillna("").str.strip().str[:5]
    )

    # --- Group by authorized official ---
    def agg_states(x):
        return sorted(x.dropna().unique().tolist())

    def agg_list(x):
        return x.tolist()

    def max_addr_count(group):
        if len(group) == 0:
            return 0
        return group.value_counts().iloc[0]

    # Basic aggregation
    ao = org_npis.groupby(["ao_last", "ao_first"]).agg(
        npi_count=("npi", "nunique"),
        num_states=("state", "nunique"),
        num_deactivated=("is_deactivated", "sum"),
        num_reactivated=("is_reactivated", "sum"),
        num_hh=("is_hh", "sum"),
        num_subparts=("is_subpart_flag", "sum"),
        phone_count=("ao_phone", "nunique"),
    ).reset_index()

    # List aggregations (need separate groupby for list ops)
    ao_lists = org_npis.groupby(["ao_last", "ao_first"]).agg(
        npi_list=("npi", agg_list),
        states=("state", agg_states),
        org_names=("org_name", agg_list),
        cities=("city", lambda x: sorted(x.fillna("").str.upper().str.strip().unique().tolist())),
        ao_middle=("ao_middle", "first"),
    ).reset_index()

    ao = ao.merge(ao_lists, on=["ao_last", "ao_first"])

    # Address clustering: max NPIs at same address per official
    addr_max = org_npis.groupby(["ao_last", "ao_first"])["addr_key"].apply(
        max_addr_count
    ).reset_index().rename(columns={"addr_key": "same_address_count"})
    ao = ao.merge(addr_max, on=["ao_last", "ao_first"])
    ao["same_address_ratio"] = ao["same_address_count"] / ao["npi_count"]

    ao["has_hh_taxonomy"] = (ao["num_hh"] > 0).astype(int)
    ao["has_deactivated"] = (ao["num_deactivated"] > 0).astype(int)

    # Filter to multi-NPI officials
    multi = ao[ao["npi_count"] >= 2].copy().reset_index(drop=True)

    print(f"  Total unique authorized officials: {len(ao):,}")
    print(f"  Officials with 2+ NPIs: {len(multi):,}")
    print(f"  Officials with 5+ NPIs: {(ao['npi_count'] >= 5).sum():,}")
    print(f"  Officials with 10+ NPIs: {(ao['npi_count'] >= 10).sum():,}")
    print(f"  Officials with 20+ NPIs: {(ao['npi_count'] >= 20).sum():,}")

    return ao, multi


# =====================================================================
# Step 4: Cross-Reference
# =====================================================================

def crossref_leie_individuals_tiered(multi, leie, name_freq_dict):
    """Match LEIE excluded individuals against authorized officials with tiered confidence.

    Tier 1: LEIE NPI directly matches one of the AO's controlled org NPIs (highest)
    Tier 2: Name + state + city match (high confidence)
    Tier 3: Name + state + middle name match (medium confidence)
    Tier 4: Name + state only (low confidence — current behavior)

    Returns (all_matches_with_tier_column, name_freq_dict).
    """
    leie_ind = leie[leie["LASTNAME"].str.strip().str.len() > 0].copy()
    leie_ind["leie_last"] = leie_ind["LASTNAME"].str.upper().str.strip()
    leie_ind["leie_first"] = leie_ind["FIRSTNAME"].str.upper().str.strip()
    leie_ind["leie_mid"] = leie_ind["MIDNAME"].fillna("").str.upper().str.strip()
    leie_ind["leie_city"] = leie_ind["CITY"].fillna("").str.upper().str.strip()

    # ---- Tier 1: Direct NPI match ----
    leie_with_npi = leie_ind[
        leie_ind["NPI"].str.strip().ne("") &
        leie_ind["NPI"].ne("0000000000")
    ].copy()

    # Explode multi's npi_list to get (ao_last, ao_first, npi) rows
    npi_lookup = multi[["ao_last", "ao_first", "npi_list"]].explode("npi_list")
    tier1 = leie_with_npi.merge(
        npi_lookup,
        left_on="NPI",
        right_on="npi_list",
        how="inner",
    )
    tier1["tier"] = 1
    # Deduplicate to one row per (ao_last, ao_first) for tier assignment
    tier1_officials = set(zip(tier1["ao_last"], tier1["ao_first"]))
    print(f"  Tier 1 (direct NPI match): {len(tier1_officials)} officials")

    # ---- Name + state base matches (used by Tiers 2-4) ----
    leie_for_name = leie_ind[["leie_last", "leie_first", "leie_mid", "leie_city",
                               "STATE", "NPI", "EXCLTYPE", "EXCLDATE"]].rename(
        columns={"NPI": "leie_npi"}
    )
    name_matches = multi.merge(
        leie_for_name,
        left_on=["ao_last", "ao_first"],
        right_on=["leie_last", "leie_first"],
        how="inner",
    )
    # Filter to state overlap
    name_state = name_matches[name_matches.apply(
        lambda r: str(r["STATE"]).strip() in r["states"], axis=1
    )].copy()

    # Exclude Tier 1 officials from name_state pool
    name_state["_ao_key"] = list(zip(name_state["ao_last"], name_state["ao_first"]))
    not_in_t1 = ~name_state["_ao_key"].isin(tier1_officials)

    # ---- Tier 2: Name + state + city ----
    city_match = name_state.apply(
        lambda r: r["leie_city"] != "" and r["leie_city"] in r["cities"], axis=1
    )
    tier2 = name_state[not_in_t1 & city_match].copy()
    tier2["tier"] = 2
    tier2_officials = set(zip(tier2["ao_last"], tier2["ao_first"]))
    print(f"  Tier 2 (name + state + city): {len(tier2_officials)} officials")

    # ---- Tier 3: Name + state + middle name ----
    not_in_t1_or_t2 = not_in_t1 & ~name_state["_ao_key"].isin(tier2_officials)
    mid_match = (
        (name_state["leie_mid"] != "") &
        (name_state["ao_middle"] != "") &
        (name_state["leie_mid"] == name_state["ao_middle"])
    )
    tier3 = name_state[not_in_t1_or_t2 & mid_match].copy()
    tier3["tier"] = 3
    tier3_officials = set(zip(tier3["ao_last"], tier3["ao_first"]))
    print(f"  Tier 3 (name + state + middle): {len(tier3_officials)} officials")

    # ---- Tier 4: Name + state only (everything remaining) ----
    higher_tiers = tier1_officials | tier2_officials | tier3_officials
    tier4 = name_state[~name_state["_ao_key"].isin(higher_tiers)].copy()
    tier4["tier"] = 4
    tier4_officials = set(zip(tier4["ao_last"], tier4["ao_first"]))
    print(f"  Tier 4 (name + state only): {len(tier4_officials)} officials")

    # ---- Combine all tiers ----
    # Normalize tier1 columns to match the others
    keep_cols = ["ao_last", "ao_first", "leie_last", "leie_first", "leie_mid",
                 "leie_city", "STATE", "EXCLTYPE", "EXCLDATE", "tier"]
    # tier1 may lack some columns from leie_for_name — add them
    for col in ["leie_mid", "leie_city", "EXCLTYPE", "EXCLDATE", "STATE"]:
        if col not in tier1.columns:
            tier1[col] = ""

    all_matches = pd.concat([
        tier1[keep_cols],
        tier2[keep_cols],
        tier3[keep_cols],
        tier4[keep_cols],
    ], ignore_index=True)

    total_officials = len(set(zip(all_matches["ao_last"], all_matches["ao_first"])))
    print(f"  Total LEIE individual matches: {len(all_matches):,} rows, "
          f"{total_officials} unique officials")

    return all_matches


def crossref_leie_organizations(multi, org_npis, leie):
    """Match LEIE excluded organizations against NPPES org names (name+state)."""
    leie_org = leie[leie["BUSNAME"].str.strip().str.len() > 0].copy()
    leie_org["leie_busname"] = leie_org["BUSNAME"].str.upper().str.strip()

    # Build lookup of org NPIs with normalized names
    org_names = org_npis[["npi", "org_name", "org_dba", "state", "ao_last", "ao_first"]].copy()
    org_names["org_name_norm"] = org_names["org_name"].fillna("").str.upper().str.strip()
    org_names["org_dba_norm"] = org_names["org_dba"].fillna("").str.upper().str.strip()

    # Match on legal business name + state
    matches_lbn = org_names.merge(
        leie_org[["leie_busname", "STATE", "NPI", "EXCLTYPE", "EXCLDATE"]].rename(
            columns={"NPI": "leie_npi"}
        ),
        left_on=["org_name_norm", "state"],
        right_on=["leie_busname", "STATE"],
        how="inner",
    )
    matches_lbn["match_type"] = "legal_business_name"

    # Match on DBA name + state
    org_names_dba = org_names[org_names["org_dba_norm"] != ""]
    matches_dba = org_names_dba.merge(
        leie_org[["leie_busname", "STATE", "NPI", "EXCLTYPE", "EXCLDATE"]].rename(
            columns={"NPI": "leie_npi"}
        ),
        left_on=["org_dba_norm", "state"],
        right_on=["leie_busname", "STATE"],
        how="inner",
    )
    matches_dba["match_type"] = "dba_name"

    matches = pd.concat([matches_lbn, matches_dba]).drop_duplicates(subset=["npi"])

    # Filter to multi-NPI officials only
    multi_npis = set()
    for npi_list in multi["npi_list"]:
        multi_npis.update(npi_list)
    matches = matches[matches["npi"].isin(multi_npis)]

    print(f"  LEIE org matches (name + state): {len(matches):,} NPIs "
          f"({matches[['ao_last','ao_first']].drop_duplicates().shape[0]} unique officials)")

    return matches


def crossref_deactivated_individuals(multi, deact_ind):
    """Match deactivated individuals against authorized officials (name+state)."""
    matches = multi.merge(
        deact_ind[["npi", "ind_last", "ind_first", "ind_state", "deactivation_date",
                    "ind_taxonomy"]].rename(columns={"npi": "deact_npi"}),
        left_on=["ao_last", "ao_first"],
        right_on=["ind_last", "ind_first"],
        how="inner",
    )

    # Filter to state overlap
    matches = matches[matches.apply(
        lambda r: str(r["ind_state"]).strip() in r["states"], axis=1
    )].copy()

    # Common-name frequency
    name_freq = matches.groupby(["ao_last", "ao_first"])["deact_npi"].nunique().reset_index()
    name_freq = name_freq.rename(columns={"deact_npi": "deact_ind_match_count"})

    print(f"  Deactivated individual matches (name + state): {len(matches):,} "
          f"({matches[['ao_last','ao_first']].drop_duplicates().shape[0]} unique officials)")

    return matches, name_freq


# =====================================================================
# Step 5: Spending Exposure
# =====================================================================

def get_spending_exposure(multi, spending_csv):
    """Get Medicaid spending for all NPIs controlled by multi-NPI officials."""
    # Collect all NPIs
    all_npis = set()
    for npi_list in multi["npi_list"]:
        all_npis.update(npi_list)

    npi_df = pd.DataFrame({"npi": list(all_npis)})

    conn = duckdb.connect()
    conn.register("flagged_npis", npi_df)

    query = f"""
    SELECT
        s.BILLING_PROVIDER_NPI_NUM AS billing_npi,
        SUM(s.TOTAL_PAID) AS total_paid,
        SUM(s.TOTAL_CLAIMS) AS total_claims,
        SUM(s.TOTAL_UNIQUE_BENEFICIARIES) AS total_beneficiaries,
        COUNT(DISTINCT s.HCPCS_CODE) AS num_hcpcs,
        COUNT(DISTINCT s.CLAIM_FROM_MONTH) AS num_months
    FROM read_csv('{spending_csv}', header=true, auto_detect=true) s
    INNER JOIN flagged_npis f ON s.BILLING_PROVIDER_NPI_NUM = f.npi
    GROUP BY s.BILLING_PROVIDER_NPI_NUM
    """

    print(f"  Querying Medicaid spending for {len(all_npis):,} NPIs "
          f"(full 11 GB scan)...")
    spending = conn.sql(query).df()
    conn.close()

    print(f"  Found spending data for {len(spending):,} NPIs, "
          f"${spending['total_paid'].sum():,.0f} total")
    return spending


# =====================================================================
# Step 6: Risk Scoring
# =====================================================================

def compute_risk_scores(multi, leie_ind_matches, leie_org_matches,
                        deact_matches, spending, name_freq_dict):
    """Composite laundering risk score per authorized official.

    Uses tiered LEIE individual matching with name frequency penalty
    instead of a binary flag.
    """

    # --- Roll spending up to official level ---
    # Build npi → official mapping
    npi_to_official = {}
    for _, row in multi.iterrows():
        key = (row["ao_last"], row["ao_first"])
        for npi in row["npi_list"]:
            npi_to_official[npi] = key

    if len(spending) > 0:
        spending["ao_key"] = spending["billing_npi"].map(npi_to_official)
        spending_valid = spending.dropna(subset=["ao_key"])
        ao_spending = spending_valid.groupby("ao_key").agg(
            total_spending=("total_paid", "sum"),
            total_claims=("total_claims", "sum"),
            npis_with_spending=("billing_npi", "nunique"),
        ).reset_index()
        ao_spending[["ao_last", "ao_first"]] = pd.DataFrame(
            ao_spending["ao_key"].tolist(), index=ao_spending.index
        )
        ao_spending = ao_spending.drop(columns=["ao_key"])
    else:
        ao_spending = pd.DataFrame(columns=["ao_last", "ao_first",
                                             "total_spending", "total_claims",
                                             "npis_with_spending"])

    multi = multi.merge(ao_spending, on=["ao_last", "ao_first"], how="left")
    multi["total_spending"] = multi["total_spending"].fillna(0)
    multi["total_claims"] = multi["total_claims"].fillna(0)
    multi["npis_with_spending"] = multi["npis_with_spending"].fillna(0).astype(int)

    # --- Tiered LEIE individual scoring ---
    tier_weights = {1: 3.0, 2: 2.0, 3: 1.5, 4: 0.5}

    # Best (lowest) tier per official
    best_tier_map = {}
    if len(leie_ind_matches) > 0:
        best_tier = leie_ind_matches.groupby(
            ["ao_last", "ao_first"]
        )["tier"].min()
        for (last, first), tier in best_tier.items():
            best_tier_map[(last, first)] = tier

    # Name frequency penalty: 1 / log2(count + 1)
    multi["name_freq_count"] = multi.apply(
        lambda r: name_freq_dict.get((r["ao_last"], r["ao_first"]), 0), axis=1
    )
    multi["name_freq_penalty"] = 1.0 / np.log2(multi["name_freq_count"].clip(lower=1) + 1)

    # Best tier column
    multi["leie_best_tier"] = multi.apply(
        lambda r: best_tier_map.get((r["ao_last"], r["ao_first"]), 0), axis=1
    )

    # LEIE individual score = tier_weight * name_freq_penalty (0 if no match)
    multi["leie_ind_score"] = multi.apply(
        lambda r: tier_weights.get(r["leie_best_tier"], 0) * r["name_freq_penalty"]
        if r["leie_best_tier"] > 0 else 0.0,
        axis=1,
    )

    # Keep binary flags for reporting compatibility
    multi["has_leie_ind_match"] = (multi["leie_best_tier"] > 0).astype(int)

    # --- Other flag columns from cross-references ---
    leie_org_officials = set()
    if len(leie_org_matches) > 0:
        for _, r in leie_org_matches[["ao_last", "ao_first"]].drop_duplicates().iterrows():
            leie_org_officials.add((r["ao_last"], r["ao_first"]))

    deact_officials = set()
    if len(deact_matches) > 0:
        for _, r in deact_matches[["ao_last", "ao_first"]].drop_duplicates().iterrows():
            deact_officials.add((r["ao_last"], r["ao_first"]))

    multi["has_leie_org_match"] = multi.apply(
        lambda r: int((r["ao_last"], r["ao_first"]) in leie_org_officials), axis=1
    )
    multi["has_deact_ind_match"] = multi.apply(
        lambda r: int((r["ao_last"], r["ao_first"]) in deact_officials), axis=1
    )

    # Temporal flag: org created after deactivation
    multi["temporal_flag"] = multi["has_deact_ind_match"]  # simplified for v1

    # --- Percentile normalization ---
    score_features = [
        "npi_count", "num_states", "num_deactivated", "total_spending",
        "same_address_ratio",
    ]
    for col in score_features:
        multi[f"{col}_pct"] = multi[col].rank(pct=True)

    # --- Weighted composite ---
    # leie_ind_score replaces the old has_leie_ind_match * 2.0
    multi["risk_score"] = (
        multi["leie_ind_score"] +
        multi["has_leie_org_match"] * 2.0 +
        multi["has_deact_ind_match"] * 1.5 +
        multi["num_deactivated_pct"] * 1.5 +
        multi["npi_count_pct"] * 1.0 +
        multi["num_states_pct"] * 0.8 +
        multi["total_spending_pct"] * 0.5 +
        multi["same_address_ratio_pct"] * 0.4 +
        multi["has_hh_taxonomy"] * 0.3 +
        multi["temporal_flag"] * 1.0
    )
    multi["risk_rank"] = multi["risk_score"].rank(ascending=False).astype(int)

    return multi


# =====================================================================
# Main
# =====================================================================

def main():
    """Main entry point for the NPI laundering analysis script.

    Orchestrates the loading of data, building of networks, cross-referencing,
    risk scoring, and reporting.
    """
    parser = argparse.ArgumentParser(description="NPI Laundering Analysis")
    parser.add_argument('--nppes-csv', default=DEFAULT_NPPES_CSV, help='Path to NPPES CSV')
    parser.add_argument('--deactivated-xlsx', default=DEFAULT_DEACTIVATED_XLSX, help='Path to deactivated NPI XLSX')
    parser.add_argument('--spending-csv', default=DEFAULT_SPENDING_CSV, help='Path to Medicaid spending CSV')
    parser.add_argument('--hh-features-parquet', default=DEFAULT_HH_FEATURES_PARQUET, help='Path to HH features Parquet')
    parser.add_argument('--leie-csv', default=DEFAULT_LEIE_CSV, help='Path to LEIE CSV')
    parser.add_argument('--leie-fallback', default=DEFAULT_LEIE_FALLBACK, help='Path to LEIE fallback CSV')
    parser.add_argument('--networks-parquet', default=DEFAULT_NETWORKS_PARQUET, help='Output path for networks Parquet')
    parser.add_argument('--flags-parquet', default=DEFAULT_FLAGS_PARQUET, help='Output path for flags Parquet')
    parser.add_argument('--top', type=int, default=30, help='Number of top officials to display in reports')
    args = parser.parse_args()

    # === STEP 1: Load LEIE ===
    section("LOADING DATA")
    leie = load_leie(args.leie_csv, args.leie_fallback)

    # === STEP 2: DuckDB extraction ===
    org_npis = extract_org_npis(args.nppes_csv)
    deact_ind = extract_deactivated_individuals(args.nppes_csv, args.deactivated_xlsx)
    name_freq_dict = compute_name_frequencies(args.nppes_csv)

    # === STEP 3: Build networks ===
    print("\nBuilding authorized official networks...")
    all_ao, multi = build_ao_networks(org_npis)

    # === STEP 4: Cross-reference ===
    print("\nCross-referencing against exclusion/deactivation lists...")
    leie_ind_matches = crossref_leie_individuals_tiered(multi, leie, name_freq_dict)
    leie_org_matches = crossref_leie_organizations(multi, org_npis, leie)
    deact_matches, deact_name_freq = crossref_deactivated_individuals(multi, deact_ind)

    # === STEP 5: Spending exposure ===
    print("\nComputing spending exposure...")
    spending = get_spending_exposure(multi, args.spending_csv)

    # === STEP 6: Risk scoring ===
    print("\nComputing risk scores...")
    multi = compute_risk_scores(
        multi, leie_ind_matches, leie_org_matches, deact_matches, spending,
        name_freq_dict
    )

    # ================================================================
    # REPORT
    # ================================================================

    # --- 1. Overview ---
    section("1. AUTHORIZED OFFICIAL NETWORK OVERVIEW")
    n_total_ao = len(all_ao)
    n_total_orgs = len(org_npis)
    print(f"  Total organizational NPIs with authorized officials: {n_total_orgs:,}")
    print(f"  Unique authorized officials: {n_total_ao:,}")

    bins = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, None)]
    bin_labels = ["1", "2-4", "5-9", "10-19", "20-49", "50+"]
    print(f"\n  {'NPI Count':>10s} {'Officials':>10s} {'% of Total':>10s}")
    print(f"  {'-'*32}")
    for (lo, hi), label in zip(bins, bin_labels):
        if hi is None:
            mask = all_ao["npi_count"] >= lo
        else:
            mask = (all_ao["npi_count"] >= lo) & (all_ao["npi_count"] <= hi)
        n = mask.sum()
        print(f"  {label:>10s} {n:>10,} {100*n/n_total_ao:>9.1f}%")

    multi_spending = multi["total_spending"].sum()
    print(f"\n  Multi-NPI officials (2+): {len(multi):,}")
    print(f"  Total Medicaid spending through multi-NPI networks: "
          f"${multi_spending:,.0f}")

    # --- 2. Top 30 Multi-NPI Officials ---
    section(f"2. TOP {args.top} MULTI-NPI AUTHORIZED OFFICIALS (by risk score)")
    topN = multi.nsmallest(args.top, "risk_rank")

    print(f"\n  {'Rank':>4s}  {'Last Name':<16s} {'First':<12s} "
          f"{'NPIs':>5s} {'Sts':>3s} {'Deact':>5s} {'LEIE':>5s} "
          f"{'Spending':>14s} {'Score':>6s}")
    print(f"  {'-'*74}")
    for _, r in topN.iterrows():
        leie_flag = ""
        if r["has_leie_ind_match"]:
            leie_flag += f"I{int(r['leie_best_tier'])}"
        if r["has_leie_org_match"]:
            leie_flag += "O"
        if not leie_flag:
            leie_flag = "-"
        print(f"  {r['risk_rank']:>4d}  {r['ao_last']:<16s} {r['ao_first']:<12s} "
              f"{r['npi_count']:>5d} {r['num_states']:>3d} "
              f"{r['num_deactivated']:>5d} {leie_flag:>5s} "
              f"${r['total_spending']:>13,.0f} {r['risk_score']:>6.1f}")

    # Expand top 5
    print(f"\n  Expanded detail for top 5:\n")
    for _, r in topN.head(5).iterrows():
        print(f"  [{r['risk_rank']}] {r['ao_last']}, {r['ao_first']} "
              f"— {r['npi_count']} NPIs across {r['num_states']} states "
              f"({', '.join(r['states'][:10])})")
        # Get their individual NPIs
        npi_detail = org_npis[
            (org_npis["ao_last"] == r["ao_last"]) &
            (org_npis["ao_first"] == r["ao_first"])
        ].sort_values("state")
        for _, nd in npi_detail.head(10).iterrows():
            deact_str = " [DEACT]" if nd["is_deactivated"] else ""
            hh_str = " [HH]" if nd["is_hh"] else ""
            # Get spending for this NPI
            npi_spend = spending[spending["billing_npi"] == nd["npi"]]
            spend_str = f"${npi_spend['total_paid'].iloc[0]:>12,.0f}" if len(npi_spend) > 0 else "      no data"
            print(f"      {nd['npi']} {nd['state']:<3s} {nd['org_name'][:40]:<40s} "
                  f"{spend_str}{deact_str}{hh_str}")
        if len(npi_detail) > 10:
            print(f"      ... and {len(npi_detail) - 10} more NPIs")
        print()

    # --- 3. Multi-State Officials ---
    section("3. MULTI-STATE AUTHORIZED OFFICIALS (3+ states)")
    multi_state = multi[multi["num_states"] >= 3].sort_values(
        "num_states", ascending=False
    )
    print(f"  Officials with NPIs in 3+ states: {len(multi_state):,}")
    print(f"\n  {'Last Name':<16s} {'First':<12s} {'NPIs':>5s} {'Sts':>3s} "
          f"{'States (first 8)':<30s} {'Spending':>14s}")
    print(f"  {'-'*82}")
    for _, r in multi_state.head(30).iterrows():
        states_str = ", ".join(r["states"][:8])
        if len(r["states"]) > 8:
            states_str += "..."
        print(f"  {r['ao_last']:<16s} {r['ao_first']:<12s} "
              f"{r['npi_count']:>5d} {r['num_states']:>3d} "
              f"{states_str:<30s} ${r['total_spending']:>13,.0f}")

    # --- 4. LEIE Cross-Reference ---
    section("4. LEIE CROSS-REFERENCE (excluded individuals + organizations)")

    # Individual matches — tiered breakdown
    print("  A. Excluded INDIVIDUALS matched to authorized officials (tiered):")
    if len(leie_ind_matches) > 0:
        unique_officials = leie_ind_matches[["ao_last", "ao_first"]].drop_duplicates()
        for t in [1, 2, 3, 4]:
            tier_off = leie_ind_matches[leie_ind_matches["tier"] == t][
                ["ao_last", "ao_first"]
            ].drop_duplicates()
            tier_labels = {
                1: "Tier 1 (direct NPI match)",
                2: "Tier 2 (name + state + city)",
                3: "Tier 3 (name + state + middle)",
                4: "Tier 4 (name + state only)",
            }
            print(f"     {tier_labels[t]+':':<36s} {len(tier_off):>6,} officials")
        print(f"     {'Total unique officials:':<36s} {len(unique_officials):>6,}")
        print(f"\n     Name frequency penalty applied "
              f"(NPPES individual population base: {len(name_freq_dict):,} name pairs)")

        # Merge current multi to get spending/rank/tier/freq info
        ind_with_spend = leie_ind_matches[
            ["ao_last", "ao_first", "tier", "EXCLTYPE", "EXCLDATE"]
        ].copy()
        # Keep best tier per official
        ind_best = ind_with_spend.sort_values("tier").drop_duplicates(
            subset=["ao_last", "ao_first"], keep="first"
        )
        ind_best = ind_best.merge(
            multi[["ao_last", "ao_first", "total_spending", "npi_count",
                   "num_states", "risk_rank", "name_freq_count",
                   "name_freq_penalty", "leie_ind_score"]],
            on=["ao_last", "ao_first"], how="left",
        )

        # Display by tier groupings
        def _print_tier_table(df, label, n=10):
            if len(df) == 0:
                print(f"\n     --- {label} ---")
                print(f"     (no matches)")
                return
            df = df.sort_values("total_spending", ascending=False)
            print(f"\n     --- {label} ---")
            print(f"     {'Last Name':<16s} {'First':<12s} {'NPIs':>5s} {'Sts':>3s} "
                  f"{'Spending':>14s} {'Freq':>6s} {'Score':>6s} {'Rank':>5s}")
            print(f"     {'-'*70}")
            for _, r in df.head(n).iterrows():
                print(f"     {r['ao_last']:<16s} {r['ao_first']:<12s} "
                      f"{r['npi_count']:>5d} {r['num_states']:>3d} "
                      f"${r['total_spending']:>13,.0f} "
                      f"{r['name_freq_count']:>6.0f} "
                      f"{r['leie_ind_score']:>6.2f} "
                      f"{r['risk_rank']:>5d}")

        _print_tier_table(
            ind_best[ind_best["tier"] == 1],
            "TIER 1: Direct NPI match (highest confidence)",
        )
        _print_tier_table(
            ind_best[ind_best["tier"] == 2],
            "TIER 2: Name + state + city (high confidence)",
        )
        _print_tier_table(
            ind_best[ind_best["tier"].isin([3, 4])],
            "TIERS 3-4: Name + state + middle/only (medium/low confidence)",
        )

    # Organization matches
    print(f"\n  B. Excluded ORGANIZATIONS matched to NPPES org names (name + state):")
    print(f"     Total matches: {len(leie_org_matches):,}")
    if len(leie_org_matches) > 0:
        print(f"     Unique NPIs matched: "
              f"{leie_org_matches['npi'].nunique():,}")
        unique_ao = leie_org_matches[["ao_last", "ao_first"]].drop_duplicates()
        print(f"     Unique officials controlling matched orgs: {len(unique_ao):,}")

        # Show matches side-by-side for reviewer
        print(f"\n     {'LEIE Excluded Org Name':<40s} {'NPPES Org Name':<40s} "
              f"{'St':>3s} {'Match':>5s}")
        print(f"     {'-'*90}")
        for _, r in leie_org_matches.head(20).iterrows():
            leie_name = str(r["leie_busname"])[:38]
            nppes_name = str(r["org_name"])[:38]
            print(f"     {leie_name:<40s} {nppes_name:<40s} "
                  f"{str(r['state']):>3s} {str(r['match_type'])[:5]:>5s}")

    # --- 5. Deactivated Individual Cross-Reference ---
    section("5. DEACTIVATED INDIVIDUAL CROSS-REFERENCE (name + state)")
    print(f"  Total matches: {len(deact_matches):,}")
    if len(deact_matches) > 0:
        unique_officials = deact_matches[["ao_last", "ao_first"]].drop_duplicates()
        print(f"  Unique officials matched: {len(unique_officials):,}")

        high_freq = deact_name_freq[deact_name_freq["deact_ind_match_count"] >= 10]
        print(f"  Common-name matches (10+ deact hits): {len(high_freq):,}")

        # Top 20 by spending — merge current multi (which has spending/rank)
        deact_with_spend = deact_matches[
            ["ao_last", "ao_first", "deactivation_date"]
        ].drop_duplicates(subset=["ao_last", "ao_first"]).merge(
            multi[["ao_last", "ao_first", "total_spending", "npi_count",
                   "num_states", "risk_rank"]],
            on=["ao_last", "ao_first"], how="left",
        ).sort_values("total_spending", ascending=False)

        print(f"\n  {'Last Name':<16s} {'First':<12s} {'NPIs':>5s} {'Sts':>3s} "
              f"{'Spending':>14s} {'Deact Date':>10s} {'Rank':>5s}")
        print(f"  {'-'*68}")
        for _, r in deact_with_spend.head(20).iterrows():
            print(f"  {r['ao_last']:<16s} {r['ao_first']:<12s} "
                  f"{r['npi_count']:>5d} {r['num_states']:>3d} "
                  f"${r['total_spending']:>13,.0f} "
                  f"{str(r.get('deactivation_date',''))[:10]:>10s} "
                  f"{r['risk_rank']:>5d}")

    # --- 6. Deactivated Org NPI Cycling ---
    section("6. DEACTIVATED ORG NPI CYCLING ('burn and churn')")
    cycling = multi[(multi["num_deactivated"] > 0) &
                    (multi["npi_count"] > multi["num_deactivated"])].sort_values(
        "num_deactivated", ascending=False
    )
    print(f"  Officials with BOTH active and deactivated org NPIs: {len(cycling):,}")

    if len(cycling) > 0:
        print(f"\n  {'Last Name':<16s} {'First':<12s} {'Active':>6s} {'Deact':>5s} "
              f"{'Total':>5s} {'Sts':>3s} {'Spending':>14s}")
        print(f"  {'-'*64}")
        for _, r in cycling.head(20).iterrows():
            active = r["npi_count"] - r["num_deactivated"]
            print(f"  {r['ao_last']:<16s} {r['ao_first']:<12s} "
                  f"{active:>6d} {r['num_deactivated']:>5d} "
                  f"{r['npi_count']:>5d} {r['num_states']:>3d} "
                  f"${r['total_spending']:>13,.0f}")

    # --- 7. Address Clustering ---
    section("7. ADDRESS CLUSTERING (multiple NPIs at same address)")
    addr_cluster = multi[
        (multi["same_address_count"] >= 3) & (multi["npi_count"] >= 3)
    ].sort_values("same_address_count", ascending=False)
    print(f"  Officials with 3+ NPIs at the same address: {len(addr_cluster):,}")

    if len(addr_cluster) > 0:
        print(f"\n  {'Last Name':<16s} {'First':<12s} {'SameAddr':>8s} "
              f"{'Total':>5s} {'Ratio':>6s} {'Spending':>14s}")
        print(f"  {'-'*64}")
        for _, r in addr_cluster.head(20).iterrows():
            print(f"  {r['ao_last']:<16s} {r['ao_first']:<12s} "
                  f"{r['same_address_count']:>8d} {r['npi_count']:>5d} "
                  f"{r['same_address_ratio']:>5.0%} "
                  f"${r['total_spending']:>13,.0f}")

    # --- 8. Home Health Multi-State ---
    section("8. HOME HEALTH MULTI-STATE BILLING")
    hh_multi = multi[
        (multi["has_hh_taxonomy"] == 1) & (multi["num_states"] >= 2)
    ].sort_values("total_spending", ascending=False)
    print(f"  HH officials with NPIs in 2+ states: {len(hh_multi):,}")

    if len(hh_multi) > 0:
        # Cross-reference with HH features if available
        hh_features = None
        if os.path.exists(args.hh_features_parquet):
            hh_features = pd.read_parquet(args.hh_features_parquet)[
                ["hh_risk_rank", "hh_risk_score"]
            ]

        print(f"\n  {'Last Name':<16s} {'First':<12s} {'NPIs':>5s} {'Sts':>3s} "
              f"{'HH NPIs':>7s} {'Spending':>14s} {'HH Ranks':>20s}")
        print(f"  {'-'*80}")
        for _, r in hh_multi.head(20).iterrows():
            # Get HH NPI details
            npi_detail = org_npis[
                (org_npis["ao_last"] == r["ao_last"]) &
                (org_npis["ao_first"] == r["ao_first"]) &
                (org_npis["is_hh"] == 1)
            ]
            hh_npi_count = len(npi_detail)

            # Get HH risk ranks for these NPIs
            hh_ranks_str = ""
            if hh_features is not None:
                for npi in npi_detail["npi"]:
                    if npi in hh_features.index:
                        hh_ranks_str += f"#{int(hh_features.loc[npi, 'hh_risk_rank'])} "

            print(f"  {r['ao_last']:<16s} {r['ao_first']:<12s} "
                  f"{r['npi_count']:>5d} {r['num_states']:>3d} "
                  f"{hh_npi_count:>7d} ${r['total_spending']:>13,.0f} "
                  f"{hh_ranks_str[:20]:>20s}")

    # --- 9. Spending Exposure ---
    section("9. SPENDING EXPOSURE ANALYSIS")
    total_network_spending = multi["total_spending"].sum()
    print(f"  Total Medicaid spending through multi-NPI official networks: "
          f"${total_network_spending:,.0f}")
    print(f"  NPIs with active spending: "
          f"{multi['npis_with_spending'].sum():,.0f}")

    # Spending tiers
    tiers = [
        (100_000_000, "100M+"),
        (10_000_000, "10M-100M"),
        (1_000_000, "1M-10M"),
        (100_000, "100K-1M"),
        (0, "< 100K"),
    ]
    print(f"\n  {'Tier':>12s} {'Officials':>10s} {'Total Spending':>18s}")
    print(f"  {'-'*42}")
    for threshold, label in tiers:
        if threshold == 0:
            mask = multi["total_spending"] < 100_000
        elif threshold == 100_000_000:
            mask = multi["total_spending"] >= threshold
        elif threshold == 10_000_000:
            mask = (multi["total_spending"] >= threshold) & (multi["total_spending"] < 100_000_000)
        elif threshold == 1_000_000:
            mask = (multi["total_spending"] >= threshold) & (multi["total_spending"] < 10_000_000)
        else:
            mask = (multi["total_spending"] >= threshold) & (multi["total_spending"] < 1_000_000)
        n = mask.sum()
        tier_spend = multi[mask]["total_spending"].sum()
        print(f"  {label:>12s} {n:>10,} ${tier_spend:>17,.0f}")

    # Top 10 officials by spending
    print(f"\n  Top 10 officials by total network spending:")
    top_spend = multi.nlargest(10, "total_spending")
    for _, r in top_spend.iterrows():
        print(f"    {r['ao_last']}, {r['ao_first']}: "
              f"${r['total_spending']:>14,.0f} "
              f"({r['npi_count']} NPIs, {r['num_states']} states, "
              f"risk rank #{r['risk_rank']})")

    # --- 10. Summary & Gaps ---
    section("10. SUMMARY & GAPS")
    n_flagged = ((multi["has_leie_ind_match"] == 1) |
                 (multi["has_leie_org_match"] == 1) |
                 (multi["has_deact_ind_match"] == 1)).sum()

    print(f"  Organizational NPIs analyzed: {n_total_orgs:,}")
    print(f"  Unique authorized officials: {n_total_ao:,}")
    print(f"  Multi-NPI officials (2+): {len(multi):,}")
    print(f"  Officials with LEIE individual match: "
          f"{multi['has_leie_ind_match'].sum():,}")
    for t in [1, 2, 3, 4]:
        tier_labels = {1: "Tier 1 (NPI)", 2: "Tier 2 (city)",
                       3: "Tier 3 (middle)", 4: "Tier 4 (name only)"}
        n_tier = (multi["leie_best_tier"] == t).sum()
        print(f"    {tier_labels[t]}: {n_tier:,}")
    print(f"  Officials with LEIE org name match: "
          f"{multi['has_leie_org_match'].sum():,}")
    print(f"  Officials with deactivated individual match: "
          f"{multi['has_deact_ind_match'].sum():,}")
    print(f"  Officials with any exclusion/deactivation flag: {n_flagged:,}")
    print(f"  Total Medicaid exposure (multi-NPI networks): "
          f"${total_network_spending:,.0f}")

    print(f"\n  KNOWN GAPS:")
    print(f"  - Fuzzy/substring org name matching: excluded orgs may re-register")
    print(f"    under slightly different names (e.g., 'ABC Home Health LLC' →")
    print(f"    'ABC Home Health Services LLC'). V1 uses exact match only.")
    print(f"  - Fuzzy individual name matching: name variants (Jr/Sr, hyphens,")
    print(f"    abbreviations) are not caught by exact match.")
    print(f"  - Temporal deep-dive: should analyze sequence of deactivation →")
    print(f"    org NPI creation → billing start dates for strongest signals.")

    # === SAVE ===
    # Save networks parquet (drop list columns that don't serialize well)
    save_cols = [c for c in multi.columns
                 if c not in ("npi_list", "states", "org_names", "cities")]
    multi[save_cols].to_parquet(args.networks_parquet, index=False)
    print(f"\n  Saved {len(multi):,} multi-NPI official records to {args.networks_parquet}")

    # Save flagged subset
    flags = multi[
        (multi["has_leie_ind_match"] == 1) |
        (multi["has_leie_org_match"] == 1) |
        (multi["has_deact_ind_match"] == 1) |
        (multi["risk_rank"] <= 100)
    ].copy()
    flags[save_cols].to_parquet(args.flags_parquet, index=False)
    print(f"  Saved {len(flags):,} flagged officials to {args.flags_parquet}")

    print(f"\n{'='*70}")
    print("  NPI laundering analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
