# HHS Medicaid Provider Spending - Official Data Dictionary

| Name | Type | Description |
|---|---|---|
| BILLING_PROVIDER_NPI_NUM | STRING | National Provider Identifier of the billing provider |
| SERVICING_PROVIDER_NPI_NUM | STRING | National Provider Identifier of the servicing provider |
| HCPCS_CODE | STRING | Healthcare Common Procedure Coding System code for the service |
| CLAIM_FROM_MONTH | DATE | Month for which claims are aggregated (YYYY-MM-01 format) |
| TOTAL_UNIQUE_BENEFICIARIES | INTEGER | Count of unique beneficiaries for this provider/procedure/month |
| TOTAL_CLAIMS | INTEGER | Total number of claims for this provider/procedure/month |
| TOTAL_PAID | FLOAT | Total amount paid by Medicaid (in USD) |

# Medicaid Provider Spending

This dataset contains provider-level Medicaid spending data aggregated from outpatient and professional claims with valid HCPCS codes, covering January 2018 through December 2024. It provides insights into how Medicaid dollars are distributed across providers and procedures nationwide.

## Data Description

| Attribute | Value |
|---|---|
| Time Period | January 2018 - December 2024 |
| Granularity | Provider (NPI) × HCPCS Code × Month |
| Geographic Scope | National (all states and territories) |
| Coverage | Fee-for-service, managed care, and CHIP |

This dataset aggregates individual claims to the provider-procedure-month level, providing counts of beneficiaries served, claims submitted, and total amounts paid by Medicaid.

## Use Cases

- **Provider spending analysis**: Identify top Medicaid providers by total spending or volume
- **Procedure utilization trends**: Track how utilization of specific procedures changes over time
- **Geographic comparisons**: Compare provider spending patterns across states
- **Outlier detection**: Identify unusual billing patterns for further investigation
- **Policy research**: Analyze the impact of policy changes on Medicaid spending

## About T-MSIS

The **Transformed Medicaid Statistical Information System (T-MSIS)** is CMS's comprehensive data system for collecting Medicaid and CHIP data from all 50 states, the District of Columbia, and US territories.

T-MSIS data is submitted monthly by states to CMS and includes information on:

- Beneficiary enrollment and eligibility
- Fee-for-service claims
- Managed care encounter data
- Provider information

## Cell Suppression Methodology

To protect beneficiary privacy, this dataset applies cell suppression:

- **Threshold**: Rows with fewer than 12 total claims are dropped entirely
- **Purpose**: Prevents re-identification of individuals who received uncommon procedures or visited low-volume providers

This means the dataset represents the majority of Medicaid spending but excludes low-volume provider-procedure combinations.

## Data Accuracy

This data is derived from T-MSIS submissions and is only as accurate as the data submitted by each state. State Medicaid agencies should be considered the authoritative source for all provider and claims data. T-MSIS has known data quality issues that vary by state and data element. For detailed information on data quality concerns, refer to CMS's DQ Atlas.
