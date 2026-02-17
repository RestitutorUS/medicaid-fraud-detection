# Devlog

## 2026-02-17 — Initial EDA on first 1M rows

**What:** Ran `explore_data.py` — sampled 10,000 rows from the first 1M rows of `medicaid-provider-spending.csv` and computed distributions, missing values, cardinality, and format checks.

**Key findings:**

- File is sorted by `TOTAL_PAID` descending, so first-1M-row sample is biased toward high spenders. Full random sample needed later.
- 13.25% of `SERVICING_PROVIDER_NPI_NUM` values are missing. All other columns complete.
- A few NPIs have non-standard lengths (6 or 9 digits instead of 10) — data quality issue.
- All three numeric columns are heavily right-skewed (mean >> median). `TOTAL_PAID` median ~$237K, p99 ~$4M, max ~$48.7M.
- `TOTAL_UNIQUE_BENEFICIARIES` min is 12, consistent with cell suppression threshold.
- 75.3% of rows have billing NPI == servicing NPI.
- T1019 (personal care services) is the most common HCPCS code at 10.5%.
- 7.5% of `TOTAL_PAID` values are exact whole dollars.
- 84 distinct months (Jan 2018 – Dec 2024), ~6,100 distinct billing NPIs and ~550 HCPCS codes in this sample slice.

**Next steps:** Get full row count, take a truly random sample across the entire file, and start thinking about feature engineering for anomaly detection.

## 2026-02-17 — Unbiased full-file EDA via DuckDB

**What:** Replaced the biased first-1M-row sampling in `explore_data.py` with DuckDB's `USING SAMPLE` to draw 10,000 rows uniformly from the entire 11 GB file. Also got the exact total row count.

**Key findings:**

- **Total rows: 227,083,361** (~227M). This is the full dataset size.
- The biased sample was dramatically skewed. Comparison of key metrics:

| Metric | Biased (first 1M) | Unbiased (full file) |
|---|---|---|
| TOTAL_PAID median | ~$237K | **$631** |
| TOTAL_PAID mean | ~$474K | **$4,710** |
| TOTAL_PAID p99 | ~$4M | **$70K** |
| TOTAL_CLAIMS median | ~540 | **28** |
| TOTAL_UNIQUE_BENEFICIARIES median | ~206 | **22** |
| Billing == Servicing NPI | 75.3% | **30.8%** |
| SERVICING_PROVIDER_NPI_NUM missing | 13.25% | **4.09%** |
| Top HCPCS code | T1019 (10.5%) | **99213 (5.9%)** |

- The typical Medicaid claim row is far smaller than the biased sample suggested — median payment is $631 with 22 beneficiaries and 28 claims.
- Top HCPCS codes shifted from personal care (T1019) to office visits (99213/99214), which makes more sense as the most common billing codes.
- NPI string lengths are now all 10 digits for billing NPIs (the 6/9-digit anomalies were in the high-spend rows only).
- Round-dollar payments jumped from 7.5% to 23.3%, and $100-multiple payments from unreported to 16.3% — interesting pattern for anomaly detection.
- Unique billing NPIs in sample: ~8,153; HCPCS codes: ~1,438 — much broader coverage than the biased sample.

**Technical note:** DuckDB scanned the full 11 GB CSV in about 2 minutes for the count, then another pass for the sample. Much faster than chunked pandas and gives truly uniform sampling.

**Next steps:** Start feature engineering for anomaly detection — aggregate to provider level, compute per-provider statistics, and look for outlier patterns.

## 2026-02-17 — Feature engineering pipeline (CA only)

**What:** Built `feature_engineering.py` — DuckDB pipeline that joins the 227M-row spending CSV to the NPPES provider registry (9.4M NPIs) and deactivated NPI list, filters to California providers, and aggregates to provider-level features. Output: `provider_features_ca.parquet`.

**Scope decision:** Filtered to California only (via NPPES practice state = 'CA') to keep the dataset manageable for iteration. CA has the largest Medi-Cal program, so it's a rich subset. Can scale national later by removing the filter.

**Scale:**
- 617,503 distinct billing NPIs nationally in the spending file
- **52,782 CA providers** after NPPES join (8.5% of national)
- 564,721 NPIs dropped (non-CA or not in NPPES)

**Features computed (24 total):**
- **Billing behavior (14):** total_paid, total_claims, total_beneficiaries, num_hcpcs_codes, num_active_months, paid_per_claim, claims_per_beneficiary, paid_per_beneficiary, pct_round_dollar, pct_round_hundred, pct_npi_mismatch, revenue_concentration, monthly_paid_cv, max_monthly_paid
- **Peer comparison (2):** max_peer_zscore, mean_peer_zscore (z-scores of paid_per_claim vs HCPCS code peers, CA only)
- **NPI enrichment (8):** entity_type, provider_age_years, is_deactivated, was_reactivated, is_sole_proprietor, is_org_subpart, num_taxonomy_codes, primary_taxonomy

**Key observations:**
- 34,166 orgs (65%) vs 18,616 individuals (35%) — organizations dominate CA Medicaid billing
- 142 deactivated providers with active spending data — potential red flags
- All 142 deactivated providers were also reactivated at some point
- Median total_paid per provider: $63K; mean: $2.45M; max: $6.78B (extreme right skew persists at provider level)
- Median paid_per_claim: $30.67 — typical CA Medi-Cal claim is small
- Revenue concentration median 0.63 — most providers get 63%+ of revenue from a single HCPCS code
- max_peer_zscore median 0.37 but max 79.8 — some providers charge dramatically more than peers for the same code
- Provider age median 17.7 years; a few are very new

**Technical note:** DuckDB processed both 11 GB files (spending CSV + NPPES CSV) and the Excel deactivated list in a single query with CTEs. The INNER JOIN to CA NPIs efficiently filters the spending data without needing a state column in the spending file.

**Next steps:** Train anomaly detection models (statistical baseline + Isolation Forest) on the 52,782 CA providers.

## 2026-02-17 — Anomaly detection models (CA)

**What:** Built `train_model.py` — two unsupervised anomaly detection approaches on 52,782 CA providers (23 numeric features). Output: `provider_scores_ca.parquet`.

**Preprocessing:** Log-transformed 6 heavily right-skewed features (total_paid, total_claims, total_beneficiaries, paid_per_claim, paid_per_beneficiary, max_monthly_paid), filled 3,205 NaN values in revenue_concentration (→ 1.0 for single-code providers), StandardScaler on all 23 features.

**Model 1 — Statistical baseline (z-score count):**
- Counts how many features have |z| > 3 for each provider
- 45,092 providers (85%) have zero extreme features — they're "normal"
- 7,690 (14.6%) have at least one extreme feature
- 168 (0.3%) have 3+ extreme features — the most suspicious
- 1 provider has 6 extreme features (NPI 1699703827: $6.78B total, the biggest CA biller)

**Model 2 — Isolation Forest (200 trees):**
- Scores range from -0.12 (most normal) to +0.19 (most anomalous)
- Scores follow a long tail: p50=-0.06, p95=0.04, p99=0.08, max=0.19

**Model agreement:**
- Top-50 overlap between the two models: 15 providers (30%)
- This is expected and healthy — IF captures multivariate interactions that univariate z-scores miss
- The models complement each other

**Top anomaly profile (Isolation Forest top 10):**
- All are organizations (entity_type=2), none are individuals
- Top feature contributions: `num_hcpcs_codes` and `num_taxonomy_codes` dominate — the IF is most sensitive to providers billing an unusually broad range of procedure codes with many specialty classifications
- `max_peer_zscore` and `is_org_subpart` are secondary contributors
- No deactivated providers in top 20 IF — the 142 deactivated providers don't happen to be extreme on other dimensions

**Dollar-weighted ranking:**
- Re-ranks by anomaly_score × total_paid to prioritize high-dollar suspicious providers
- Top dollar-weighted provider: NPI 1699703827 ($6.78B total, max_peer_zscore=20.9, paid_per_claim=$220)
- This ranking is most useful for investigators who want to focus on the largest potential losses

**Key insight — feature dominance issue:** The IF is heavily influenced by `num_hcpcs_codes` and `num_taxonomy_codes` (which measure breadth of services/specialties). These are genuinely unusual but may not be the strongest fraud signals. For the next iteration, consider: (a) capping or winsorizing these features, (b) running IF without them to see what else surfaces, or (c) feature importance analysis to understand the model better.

**Next steps:** Build `analyze_results.py` for deeper investigation of top anomalies, then iterate on the model.

## 2026-02-17 — Results analysis (CA)

**What:** Built `analyze_results.py` — comprehensive analysis of anomaly detection results across 52,782 CA providers. Nine sections covering score distributions, model agreement, top anomalies, feature contributions, deactivated providers, specialty breakdowns, and coverage.

**Model agreement (deeper look):**
- Top-50 overlap: IF ∩ z-score = 30%, IF ∩ dollar-weighted = 30%, all three = 8%
- Top-200 overlap: IF ∩ z-score = 36%, IF ∩ dollar-weighted = 48%, all three = 22%
- Agreement increases at larger k — the models converge on the most extreme outliers but diverge on marginal cases

**Deactivated providers:**
- 142 deactivated providers have $244.7M total Medi-Cal spending
- All 142 were subsequently reactivated, so these aren't providers billing after losing their NPI — they're providers with an interrupted history
- Top deactivated provider (NPI 1942537733): $112M paid, 916K claims, $123/claim — large but not extreme on peer z-scores
- The deactivated providers generally don't rank highly on IF or z-score, suggesting deactivation-reactivation is not strongly correlated with other anomalous patterns in this dataset

**Specialty breakdown (taxonomy groups in IF top-200):**
- Taxonomy `282` (hospitals — General Acute Care, Long Term Care, etc.) appears in 18% of top-200 anomalies but <0.1% of all providers — massively over-represented (~inf ratio). Note: initially mislabeled as "home health" — actual home health is taxonomy `251E`. Hospitals billing many different codes across many specialties is what the IF detects as anomalous.
- Taxonomy `171` (speech-language pathology) at 10% of top-200 vs 1.8% base (5.5x over-represented)
- Taxonomy `207` (physicians/allopathic) is 25% of all providers but only 13.5% of top-200 (0.5x) — physicians are under-represented in anomalies
- Taxonomy `122` (dentists) is 11% of all but only 2% of top-200 (0.2x) — dentists are very under-represented

**Entity type split:**
- Organizations are 65% of all providers but 89% of top-200 anomalies — strongly over-represented
- Individuals have much lower median billing ($13K vs $171K) and lower peer z-scores (0.09 vs 0.54)
- This makes sense: large organizations have more diverse billing patterns that the IF picks up

**Coverage:**
- 200 providers (0.4% of CA) account for $15.7B (12.2%) of CA total Medi-Cal spending
- CA total Medi-Cal spending: $129.4B across 52,782 providers

**Dollar-weighted top anomaly:** NPI 1699703827 — $6.78B total (5.2% of all CA Medi-Cal spending), $220/claim, peer z-score 20.9. This is by far the largest single biller, ranks IF #322 but dollar-weighted #1.

**Next steps:** Consider model iteration: (a) address feature dominance (num_hcpcs_codes/num_taxonomy_codes), (b) try separate models for individuals vs organizations, (c) look up taxonomy codes to add human-readable specialty names.

## 2026-02-17 — Home health fraud deep-dive (CA)

**What:** Built `analyze_home_health.py` — domain-specific analysis of CA home health providers. Uses DuckDB to extract HCPCS-level claims for 1,126 home health providers (taxonomy codes 251E, 253Z, 251J), computes HH-specific features and within-cohort peer comparisons, and produces a 10-section report. Intermediate output: `hh_claims_ca.parquet`.

**Cohort:**
- 1,072 Home Health Agencies (251E): $1.89B
- 46 In Home Supportive Care (253Z): $149M
- 8 Nursing Care Agencies (251J): $15M
- **Total: 1,126 providers, $2.06B Medi-Cal spending**
- 729 distinct HCPCS codes billed, 84,954 claim-level rows

**Top HCPCS codes by spending:**
- G0300 (LPN skilled nursing, 15min): $795M across 348 providers, avg $284/claim
- G0299 (RN skilled nursing, 15min): $281M across 412 providers, avg $120/claim
- S9124 (skilled nursing, 8hr block): $267M across 66 providers, avg $453/claim
- T1019 (personal care, 15min): $109M across 33 providers, avg $204/claim
- G0151 (physical therapy): $108M across 345 providers, avg $100/claim

**Code mix findings:**
- Median HH provider gets 56% of revenue from skilled nursing codes
- Mean 32% of revenue from non-HH codes — many home health agencies bill substantially outside their scope
- Nursing-to-aide ratio is extremely skewed: providers bill millions in nursing with almost no aide services, suggesting possible upcoding of aide visits as skilled nursing

**Key anomalies (composite risk score):**

| Rank | NPI | Total Paid | $/Claim | Why flagged |
|---|---|---|---|---|
| 1 | 1245588581 | $3.2M | $169 | Peer z-score 20.6 on Q5001 (nursing supervision), 73% non-HH billing |
| 2 | 1114246816 | $32M | $1,216 | $1,216/claim (!), 99% skilled nursing, peer z 6.0 on G0300 |
| 3 | 1437400165 | $19.9M | $461 | Peer z 9.0 on revenue code 0551, 97% skilled nursing |
| 4 | 1316970064 | $6.7M | $396 | Peer z 8.7 on 0551, 82% skilled nursing |
| 5 | 1003849886 | $7.4M | $438 | Peer z 7.8 on G0299, 99% skilled nursing, $778/claim for RN visits |

**T1020 (Home Health Aide per visit) investigation:**
- 11 providers bill T1020 (not 3 as initially estimated from a smaller query)
- NPI 1275030298: $16.6M at $1,776/claim — wildly above cohort average of $418/claim
- NPI 1952945628: $11M at $342/claim — also elevated
- NPI 1316596000: $2.5K at $18/claim — normal pricing for comparison

**Cross-reference with general model:**
- HH top-50 has **zero overlap** with General IF top-200 — the domain-specific model surfaces completely different providers than the general pipeline
- Only 4 HH providers appear in the General IF top-200 at all
- This validates the approach: general IF was dominated by num_hcpcs_codes/num_taxonomy_codes, while HH-specific features (pricing, code mix, upcoding ratios) identify different and more actionable anomalies

**Deactivated HH providers:** Only 5, with $13M total spending. Not a major signal in this cohort.

**Coverage:** Top-30 HH anomalies account for $359M (17.5% of all HH spending in CA).

**NPI 1114246816 is particularly suspicious:** $32M total, $1,216/claim average across all codes. For G0300 (LPN skilled nursing, 15-min increments), they charge $1,264/claim vs cohort average of $160. That's 8x the peer average. 99% of revenue is skilled nursing with zero aide billing.

**Next steps:** (a) Look up the flagged NPIs in NPPES for provider names and addresses, (b) investigate NPI 1114246816 and 1275030298 in more detail (claim volume over time, geographic concentration), (c) consider a similar deep-dive for other over-represented taxonomy groups (282/hospitals, 171/speech-language pathology).

## 2026-02-17 — Scaled Home Health Analysis Nationally

**What:** Modified `analyze_home_health.py` to run against all states/territories instead of CA only. Added `provider_state` and `is_deactivated` (from NPPES) directly to the DuckDB extraction, removing the dependency on `provider_scores_ca.parquet` for deactivation data. Added three new report sections: state-by-state anomaly concentration (§11), CA vs national comparison (§12), and CA stability check (§13). Saves `hh_features_national.parquet` for downstream use.

**Scale:** 15,697 HH providers across 53 states/territories, 1,651,339 claim rows, $163.8B total Medicaid spending (vs 1,126 CA providers with $2.06B in the CA-only run). 2,236 distinct HCPCS codes billed.

**Key findings:**

- **NY dominates HH spending:** $73.2B (45% of national HH total) from only 945 providers. TX (1,568 providers, $13.5B) and FL (1,264 providers, $6.5B) are distant second/third by provider count.
- **National top-30 is geographically diverse:** MD (5 providers in top-30), OR, KS, CO, UT each have 2. CA has only 2 (NPI 1114246816 at #6, NPI 1114919099 at #17).
- **CA providers mostly drop in national ranking:** Only 1 of 30 CA top-30 providers remains in the national top-30 (NPI 1114246816, CA #2 → national #6). 13/30 remain in the national top-100. The CA-only #1 (NPI 1245588581) dropped to national #34.
- **State anomaly concentration:** Oregon (9.1% of its HH providers in national top-100), Maryland (7.4%), and Utah (5.5%) have the highest concentration of nationally-anomalous HH providers. Oregon's top-100 providers account for 27% of its HH spending.
- **Top national anomaly (NPI 1801216254, MD):** $1.0M total, but $1,617/claim for G0299 (skilled nursing) vs $57 peer avg — a 20.8σ outlier. 100% skilled nursing, zero aide billing.
- **Deactivated providers nationally:** 40 (vs 5 in CA), $670M total. NPI 1720471568 (NY) alone has $293M. NPI 1790001840 (RI) has a 6.6 peer z-score while deactivated.
- **Zero overlap** between national HH top-50 and General IF top-200 persists — the HH-specific model surfaces entirely different providers.
- **"Fallers" from CA to national** are interesting: providers that looked anomalous in CA's small peer group become unremarkable against national peers (e.g., NPI 1750708574: CA #460 → national #15,643). This confirms the value of national peer comparison.

**Interpretation:** The CA-only run's top findings are largely validated — NPI 1114246816 remains a strong national outlier (#6). But many CA providers that appeared anomalous were merely unusual within a small state peer group. The national model reveals MD and OR as states with unusually concentrated HH anomalies, and surfaces new high-priority providers (especially in MD with extreme skilled nursing pricing) that were invisible in the CA-only analysis.

## 2026-02-17 — NPI Laundering Analysis via Authorized Official Cross-Reference

**What:** Built `npi_laundering_analysis.py` — a new analysis that detects NPI laundering patterns by cross-referencing authorized officials across multiple organizational NPIs against the OIG LEIE exclusion list and deactivated NPI list. Downloaded LEIE data (82,714 exclusion records) from OIG. Three DuckDB scans: NPPES orgs, NPPES deactivated individuals, and Medicaid spending.

**Scale:**
- 1,894,688 organizational NPIs with authorized officials
- 999,906 unique authorized officials
- 242,247 officials controlling 2+ NPIs (24% of all officials)
- 44,908 with 5+ NPIs, 15,599 with 10+ NPIs, 5,705 with 20+ NPIs
- $842.9B total Medicaid spending flows through multi-NPI official networks

**LEIE cross-reference findings:**
- 82,714 LEIE records: 79,322 individuals, 3,390 excluded organizations
- 4,116 authorized officials matched an excluded individual by name + state (4,880 match pairs). Most are likely common-name coincidences (e.g., SMITH, JOHN), but the matches flagged for manual review include officials controlling large networks.
- 123 organizational NPIs matched an excluded organization by exact business name + state. These are the highest-confidence signals — e.g., PASSAGES HOSPICE LLC (IL), PHARMACADE PHARMACY INC (NY), A CARING ALTERNATIVE INC (OH) all match excluded orgs exactly.
- 72 unique authorized officials control NPIs that match excluded organizations.

**Deactivated individual cross-reference:**
- 15,354 deactivated individual providers extracted from NPPES
- 1,554 authorized officials matched a deactivated individual by name + state
- Zero common-name matches (10+ hits), suggesting these are higher quality than LEIE name matches

**Key patterns detected:**
- **Burn and churn:** 1,142 officials control both active and deactivated org NPIs. Top: FRANCESCUTTO, SARA (310 active, 70 deactivated, $9.3M); SUTTON, SHIRLEY (2 active, 18 deactivated, $121M).
- **Address clustering:** 16,970 officials have 3+ NPIs at the same physical address. ALPERT, ELLIOT has 59 of 61 NPIs at one address (97% ratio). CANNA, JAY has 51/51 at one address (100%).
- **Multi-state HH billing:** 16,621 HH officials span 2+ states. WILLIAMS, KIMBERLY controls 1,133 NPIs across 47 states with $5.6B in spending.
- **Mega-networks with LEIE flags:** LEE, JAE (#1 risk rank) has both individual and organizational LEIE matches, 27 NPIs across 10 states, $2M spending. Many of these appear to be different people with the same common name, underscoring the need for additional identifiers.

**Observation on common names:** The top-30 risk-scored officials are dominated by common names (SMITH, JONES, LEE, BROWN, etc.) because the LEIE individual match flag adds +2.0 to the risk score. These are likely false positives. The more actionable findings are: (a) the 123 exact org name matches in section 4B, (b) the burn-and-churn patterns in section 6, and (c) the address clustering in section 7. A future iteration should add a common-name penalty that reduces the LEIE individual match weight when the name is very common.

**Output files:** `ao_networks.parquet` (242,247 officials), `laundering_flags.parquet` (5,584 flagged officials), `leie/UPDATED.csv` (82,714 LEIE records).

**Known gaps:** (a) Fuzzy/substring org name matching for re-registered excluded orgs, (b) fuzzy individual name matching, (c) temporal sequence analysis (deactivation date → org NPI creation date → first billing date).

## 2026-02-17 — Tiered LEIE Individual Matching with Name Frequency Penalty

**What:** Replaced the binary LEIE individual match flag (weight 2.0) with a 4-tier confidence system and a population-level name frequency penalty. The previous top-30 was dominated by common names (SMITH, JONES, LEE) that were almost certainly false positives. Added a third NPPES scan to count (last, first) name pairs among ~7.1M individual providers.

**Tier system:**
- **Tier 1 (direct NPI match):** LEIE entry's NPI appears in the AO's controlled org NPIs — 2 officials (KARAPETYAN, LIANA and FRIED, JOYCE). Highest confidence; these are excluded entities whose NPIs are still in a live network. KARAPETYAN has $9.8M spending.
- **Tier 2 (name + state + city):** 1,011 officials. City match dramatically reduces false positive rate for common names.
- **Tier 3 (name + state + middle name):** 168 officials. Middle name corroboration on top of name + state.
- **Tier 4 (name + state only):** 2,935 officials. The old matching behavior — now heavily discounted.

**Name frequency penalty:** `1 / log2(count + 1)` where count is the number of individual providers in NPPES with the same (last, first) name. Unique names (count=1) get full weight; SMITH, JOHN (count ~10,000) gets ~0.07x. Combined with tier weights (3.0, 2.0, 1.5, 0.5), the effective LEIE individual contribution ranges from 3.0 (Tier 1 unique name) down to ~0.035 (Tier 4 common name).

**Impact on top-30 rankings:** The top-30 is no longer dominated by common names. Instead, officials with org LEIE matches + individual matches (LEE JAE, YOON ERIC, MURPHY JERRY) and high-tier individual matches rise to the top. Common-name Tier 4 matches (YOUNG ROBERT, SMITH STEVEN) still appear but with much lower LEIE contribution scores (0.07-0.08 vs the old 2.0 flat).

**Tier 2 findings:** The 1,011 Tier 2 matches (name + state + city) are the sweet spot — they have enough corroboration to be taken seriously while remaining a manageable review set. Examples: BLAIR MICHAEL (114 NPIs, $241M, freq=17), MURPHY SHARON (15 NPIs, $30M), GOLDIS MICHAEL (2 NPIs, $60K, freq=1 — nearly unique name).

**5,194,198 unique name pairs** in the NPPES individual provider population — this is the denominator for the frequency penalty.
