"""
listed_analysis.py — Complete CRMLS Listing Data Pipeline (Weeks 1–7)

This script mirrors sold_analysis.py but operates on the active/historical
listing dataset instead of closed sales. Key differences from sold_analysis.py:
  - Source files  : CRMLSListing*.csv  (not CRMLSSold*)
  - Primary price : ListPrice  (not ClosePrice — many listings never closed)
  - Mortgage merge: keyed on ListingContractDate (not CloseDate)
  - Time columns  : derived from ListingContractDate
  - Outlier IQR   : applied to ListPrice, LivingArea, DaysOnMarket
  - Duplicate cols: listing CSVs have duplicate column headers — we drop the
                    '.1'-suffixed copies that pandas creates automatically
  - Output files  : listed_*.csv  (not sold_*.csv)

Run from the project root:
  python scripts/listed_analysis.py
"""

import sys
import pandas as pd
import glob
import os
import io
import numpy as np
import urllib.request

# Force UTF-8 output so Unicode characters print correctly on Windows
sys.stdout.reconfigure(encoding='utf-8')

# ── Working directory: always run relative to the project root ───────────────
# This block finds the project root regardless of where you launch the script from.
# __file__ = the path to this script; dirname twice walks up from scripts/ to root.
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

os.makedirs("output/reports", exist_ok=True)

print("=" * 70)
print("LISTING DATA PIPELINE — CRMLS MLS Data")
print("=" * 70)


# =============================================================================
# WEEK 1 — AGGREGATION
# Goal: combine all monthly listing files into one Residential-only dataset
# =============================================================================
print("\n" + "=" * 70)
print("WEEK 1 — AGGREGATION")
print("=" * 70)

# glob.glob finds every file matching the pattern.
# sorted() loads them in chronological order (Jan 2024 first).
import re
# Only match files with a 6-digit YYYYMM suffix to exclude any derived files
listing_files = sorted(
    f for f in glob.glob("data/CRMLSListing*.csv")
    if re.search(r'CRMLSListing\d{6}\.csv$', f)
)
print(f"Found {len(listing_files)} listing CSV files\n")

# ── KEY NOTE: Duplicate column headers in listing CSVs ──────────────────────
# The API extraction script that produced these files had some columns named
# twice in its $select parameter (e.g., PropertyType, LivingArea, DaysOnMarket).
# When pandas reads a CSV with duplicate headers, it automatically renames
# the second occurrence by appending ".1" (PropertyType -> PropertyType.1).
# After loading, we drop all ".1" columns because they're exact duplicates
# of the originals — keeping them would double-count those fields.
dfs = []
for f in listing_files:
    df_temp = pd.read_csv(f, low_memory=False, encoding='latin-1')

    # List comprehension: build a list of all column names ending in ".1"
    # [col for col in df_temp.columns if col.endswith('.1')] iterates every column
    # name and keeps only the ones that end with ".1".
    dupe_cols = [col for col in df_temp.columns if col.endswith('.1')]
    df_temp   = df_temp.drop(columns=dupe_cols)

    print(f"  {os.path.basename(f)}: {len(df_temp):,} rows  ({len(dupe_cols)} duplicate cols dropped)")
    dfs.append(df_temp)

total_before_concat = sum(len(d) for d in dfs)
print(f"\nRow count BEFORE concat (sum of all files): {total_before_concat:,}")

# pd.concat stacks all monthly DataFrames vertically (same as SQL UNION ALL).
# ignore_index=True resets row numbers from 0 continuously.
df_all = pd.concat(dfs, ignore_index=True)
print(f"Row count AFTER  concat                   : {len(df_all):,}")
print(f"Total columns                             : {df_all.shape[1]}")

# Show the full property type breakdown BEFORE filtering
print("\nPropertyType distribution BEFORE filter:")
print(df_all['PropertyType'].value_counts(dropna=False).to_string())
print()

# For listings, show MlsStatus breakdown too — this tells us the mix of
# Active, Pending, Closed, and other statuses across all listing records
print("MlsStatus distribution BEFORE filter:")
print(df_all['MlsStatus'].value_counts(dropna=False).to_string())
print()

rows_before_filter = len(df_all)

# Filter to Residential only — same reason as in sold_analysis.py:
# commercial, land, and manufactured homes have different price drivers.
df_listed = df_all[df_all['PropertyType'] == 'Residential'].copy()

rows_after_filter = len(df_listed)
print(f"Row count BEFORE PropertyType filter       : {rows_before_filter:,}")
print(f"Row count AFTER  filtering to Residential  : {rows_after_filter:,}")
print(f"Rows removed (non-residential)             : {rows_before_filter - rows_after_filter:,}")

# Save the combined, unmodified Residential listing dataset
df_listed.to_csv("data/listed_combined.csv", index=False)
print(f"\n[OK] Saved: data/listed_combined.csv  ({len(df_listed):,} rows × {df_listed.shape[1]} columns)")


# =============================================================================
# WEEKS 2-3 — STRUCTURING, VALIDATION, AND EDA
# =============================================================================
print("\n" + "=" * 70)
print("WEEKS 2-3 — STRUCTURING, VALIDATION, AND EDA")
print("=" * 70)

# ── Shape, dtypes, column names ─────────────────────────────────────────────
print(f"\nDataset shape: {df_listed.shape[0]:,} rows × {df_listed.shape[1]} columns")
print("\nAll column names:")
print(list(df_listed.columns))
print("\nData types per column:")
print(df_listed.dtypes.to_string())

# ── Missing value analysis ───────────────────────────────────────────────────
print("\n\n--- Missing Value Analysis ---")
missing_count = df_listed.isnull().sum()
missing_pct   = (missing_count / len(df_listed) * 100).round(2)

missing_summary = pd.DataFrame({
    'missing_count': missing_count,
    'missing_pct':   missing_pct,
}).sort_values('missing_pct', ascending=False)

print(missing_summary.to_string())

# Flag columns with >90% missing — candidates to drop in the cleaning step
high_missing = missing_summary[missing_summary['missing_pct'] > 90]
print(f"\n--- Columns with >90% missing data ({len(high_missing)} found) ---")
if high_missing.empty:
    print("None — every column has ≥10% coverage.")
    high_missing_cols = []
else:
    print(high_missing.to_string())
    high_missing_cols = list(high_missing.index)
    print(f"\nThese will be considered for removal in the cleaning step:")
    print(high_missing_cols)

missing_summary.to_csv("output/reports/listed_missing_value_report.csv")
print("\n[OK] Missing value report -> output/reports/listed_missing_value_report.csv")

# ── Unique PropertyType values after filter ──────────────────────────────────
print("\n--- Unique PropertyType values (after Residential filter) ---")
print(df_listed['PropertyType'].value_counts(dropna=False).to_string())

# ── MlsStatus breakdown (listing-specific insight) ──────────────────────────
# For listings, MlsStatus tells us how many are currently active, pending,
# closed, expired, etc. This is critical context: Closed listings here
# partially overlap with the sold dataset.
print("\n--- MlsStatus breakdown (Residential only) ---")
print(df_listed['MlsStatus'].value_counts(dropna=False).to_string())

# ── Numeric distribution summary ────────────────────────────────────────────
# For listings, we use ListPrice (not ClosePrice) because many active listings
# never close within this dataset.
print("\n--- Numeric Distribution Summary ---")
numeric_fields = [
    'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt',
]
numeric_fields = [f for f in numeric_fields if f in df_listed.columns]

dist_rows = []
for field in numeric_fields:
    col = pd.to_numeric(df_listed[field], errors='coerce').dropna()
    if col.empty:
        continue
    dist_rows.append({
        'field':  field,
        'count':  int(col.count()),
        'min':    round(col.min(), 2),
        'p25':    round(np.percentile(col, 25), 2),
        'median': round(np.percentile(col, 50), 2),
        'mean':   round(col.mean(), 2),
        'p75':    round(np.percentile(col, 75), 2),
        'p90':    round(np.percentile(col, 90), 2),
        'p99':    round(np.percentile(col, 99), 2),
        'max':    round(col.max(), 2),
    })

dist_df = pd.DataFrame(dist_rows).set_index('field')
print(dist_df.to_string())
dist_df.to_csv("output/reports/listed_numeric_distribution.csv")
print("\n[OK] Numeric distribution -> output/reports/listed_numeric_distribution.csv")

# ── Business Questions (listing-adapted) ────────────────────────────────────
print("\n\n--- Business Questions ---")

# Q1: Residential vs other property type share
res_count   = (df_all['PropertyType'] == 'Residential').sum()
total_count = len(df_all)
print(f"\nQ1 — Property Type Share (of all {total_count:,} listing records):")
print(f"  Residential     : {res_count:,}  ({res_count/total_count*100:.1f}%)")
print(f"  Non-residential : {total_count - res_count:,}  ({(total_count - res_count)/total_count*100:.1f}%)")
print("  Non-residential breakdown:")
non_res = df_all[df_all['PropertyType'] != 'Residential']
print(non_res['PropertyType'].value_counts(dropna=False).to_string())

# Q2: Median and average list prices
median_price = df_listed['ListPrice'].median()
mean_price   = df_listed['ListPrice'].mean()
print(f"\nQ2 — List Price Statistics (Residential):")
print(f"  Median list price : ${median_price:,.0f}")
print(f"  Average list price: ${mean_price:,.0f}")
print(f"  Mean > median by ${mean_price - median_price:,.0f} — right-skew from luxury listings")

# Q3: Days on Market distribution
dom_valid = pd.to_numeric(df_listed['DaysOnMarket'], errors='coerce').dropna()
print(f"\nQ3 — Days on Market ({dom_valid.count():,} valid records):")
print(f"  Min      : {dom_valid.min():.0f} days")
print(f"  Median   : {dom_valid.median():.0f} days")
print(f"  Average  : {dom_valid.mean():.1f} days")
print(f"  75th pct : {np.percentile(dom_valid, 75):.0f} days")
print(f"  90th pct : {np.percentile(dom_valid, 90):.0f} days")
print(f"  99th pct : {np.percentile(dom_valid, 99):.0f} days")
print(f"  Max      : {dom_valid.max():.0f} days")

# Q4: Price reduction analysis
# For listings, the "above vs below" question from sold data becomes:
# how many listings have been reduced from their original asking price?
# A positive reduction % = price was lowered = seller had to adjust to market.
price_valid = df_listed[
    df_listed['ListPrice'].notna() &
    df_listed['OriginalListPrice'].notna() &
    (df_listed['OriginalListPrice'] > 0)
].copy()
price_valid['_reduction_pct'] = (
    (price_valid['OriginalListPrice'] - price_valid['ListPrice'])
    / price_valid['OriginalListPrice'] * 100
).round(2)

reduced     = (price_valid['_reduction_pct'] > 0).sum()
at_original = (price_valid['_reduction_pct'] == 0).sum()
raised      = (price_valid['_reduction_pct'] < 0).sum()
n_pv        = len(price_valid)

print(f"\nQ4 — Price Changes from Original List Price ({n_pv:,} valid records):")
print(f"  Price reduced from original  : {reduced:,}  ({reduced/n_pv*100:.1f}%)")
print(f"  At original list price       : {at_original:,}  ({at_original/n_pv*100:.1f}%)")
print(f"  Price raised from original   : {raised:,}  ({raised/n_pv*100:.1f}%)")
if reduced > 0:
    avg_red = price_valid[price_valid['_reduction_pct'] > 0]['_reduction_pct'].mean()
    print(f"  Avg reduction among reduced  : {avg_red:.1f}%")

# Q5: Date consistency issues
list_dt  = pd.to_datetime(df_listed['ListingContractDate'], errors='coerce')
close_dt = pd.to_datetime(df_listed['CloseDate'],           errors='coerce')
purch_dt = pd.to_datetime(df_listed['PurchaseContractDate'], errors='coerce')

# Check for close before listing (impossible)
both_valid_lc = list_dt.notna() & close_dt.notna()
close_before_list = (close_dt < list_dt) & both_valid_lc

# Check for future listing dates (listed in the future — likely a data error)
today         = pd.Timestamp('today').normalize()
future_listed = list_dt.notna() & (list_dt > today)

print(f"\nQ5 — Date Consistency Issues:")
print(f"  Records with both ListingContractDate + CloseDate : {both_valid_lc.sum():,}")
print(f"  CloseDate BEFORE ListingContractDate (invalid)    : {close_before_list.sum():,}  ({close_before_list.sum()/max(both_valid_lc.sum(),1)*100:.2f}%)")
print(f"  ListingContractDate in the future (data error)    : {future_listed.sum():,}")

# Q6: Counties with highest median list prices
county_stats = (
    df_listed.groupby('CountyOrParish')['ListPrice']
    .agg(median_list_price='median', listing_count='count')
    .sort_values('median_list_price', ascending=False)
)
print(f"\nQ6 — Top 10 Counties by Median List Price:")
print(county_stats.head(10).to_string())
print(f"\n  (Full list: {len(county_stats)} counties total)")


# =============================================================================
# WEEKS 2-3 — MORTGAGE RATE ENRICHMENT
# Goal: attach the prevailing 30-year fixed rate to each listing by its listing month
# WHY: for active listings, we want the rate at the time the home was listed —
# that's when the seller's pricing decision was made relative to affordability.
# =============================================================================
print("\n" + "=" * 70)
print("WEEKS 2-3 — MORTGAGE RATE ENRICHMENT")
print("=" * 70)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
print(f"\nFetching MORTGAGE30US series from FRED...")
print(f"  URL: {FRED_URL}")

try:
    with urllib.request.urlopen(FRED_URL, timeout=30) as resp:
        raw = resp.read().decode('utf-8')

    df_fred = pd.read_csv(io.StringIO(raw))
    df_fred.columns = ['date', 'mortgage_rate_30yr']

    # FRED uses '.' for missing values — replace before converting to float
    df_fred['mortgage_rate_30yr'] = pd.to_numeric(
        df_fred['mortgage_rate_30yr'].replace('.', np.nan), errors='coerce'
    )
    df_fred = df_fred.dropna(subset=['mortgage_rate_30yr'])
    df_fred['date'] = pd.to_datetime(df_fred['date'])

    print(f"  Raw FRED data: {len(df_fred):,} weekly observations")
    print(f"  Date range   : {df_fred['date'].min().date()} -> {df_fred['date'].max().date()}")

    # Resample weekly rates to monthly averages.
    # resample('MS') groups by month start (1st of each month).
    df_fred_monthly = (
        df_fred.set_index('date')['mortgage_rate_30yr']
        .resample('MS')
        .mean()
        .reset_index()
    )
    df_fred_monthly['mortgage_rate_30yr'] = df_fred_monthly['mortgage_rate_30yr'].round(4)
    df_fred_monthly['year_month'] = df_fred_monthly['date'].dt.to_period('M').astype(str)

    print(f"\n  Monthly averages computed ({len(df_fred_monthly)} months). Last 6:")
    print(df_fred_monthly.tail(6)[['year_month', 'mortgage_rate_30yr']].to_string(index=False))

    # KEY DIFFERENCE FROM SOLD: merge on ListingContractDate (not CloseDate)
    # WHY: listing data is about when properties entered the market — the mortgage
    # rate at that moment set buyer affordability expectations for that listing.
    df_listed['ListingContractDate_dt'] = pd.to_datetime(df_listed['ListingContractDate'], errors='coerce')
    df_listed['year_month'] = df_listed['ListingContractDate_dt'].dt.to_period('M').astype(str)

    df_listed = df_listed.merge(
        df_fred_monthly[['year_month', 'mortgage_rate_30yr']],
        on='year_month',
        how='left'
    )
    df_listed.drop(columns=['ListingContractDate_dt'], errors='ignore', inplace=True)

    null_rates = df_listed['mortgage_rate_30yr'].isnull().sum()
    print(f"\n  Merge results:")
    print(f"    Total rows after merge       : {len(df_listed):,}")
    print(f"    Rows with null mortgage rate : {null_rates:,}",
          "(listings with missing/out-of-range ListingContractDate)" if null_rates else "— all rates populated [OK]")

except Exception as e:
    print(f"\n  WARNING: FRED fetch failed — {e}")
    print("  Continuing without mortgage rate enrichment (column set to NaN).")
    df_listed['mortgage_rate_30yr'] = np.nan
    df_listed['year_month'] = pd.to_datetime(df_listed['ListingContractDate'], errors='coerce').dt.to_period('M').astype(str)

df_listed.to_csv("data/listed_enriched.csv", index=False)
print(f"\n[OK] Saved: data/listed_enriched.csv  ({len(df_listed):,} rows × {df_listed.shape[1]} columns)")


# =============================================================================
# WEEKS 4-5 — DATA CLEANING
# =============================================================================
print("\n" + "=" * 70)
print("WEEKS 4-5 — DATA CLEANING")
print("=" * 70)

rows_before_cleaning = len(df_listed)
print(f"\nStarting row count: {rows_before_cleaning:,}")

# ── Step 1: Convert date fields to datetime ──────────────────────────────────
print("\n-- Step 1: Convert date columns to datetime --")
date_cols = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
for col in date_cols:
    if col in df_listed.columns:
        before_nulls = df_listed[col].isnull().sum()
        df_listed[col] = pd.to_datetime(df_listed[col], errors='coerce')
        after_nulls  = df_listed[col].isnull().sum()
        new_nulls    = after_nulls - before_nulls
        status = f"({new_nulls} unparseable -> NaT)" if new_nulls > 0 else "(all parsed cleanly)"
        print(f"  {col}: converted  {status}")

# ── Step 2: Drop non-analytical columns ─────────────────────────────────────
# Same logic as sold_analysis.py — remove agent personal info, school districts,
# tax fields, system metadata, and the listing-specific compensation fields
# (BuyerAgencyCompensation/BuyerAgencyCompensationType) that are not price metrics.
print("\n-- Step 2: Drop non-analytical columns --")
cols_to_drop = [
    # Agent personal info
    'ListAgentEmail', 'ListAgentFirstName', 'ListAgentLastName', 'ListAgentFullName',
    'CoListAgentFirstName', 'CoListAgentLastName',
    'BuyerAgentFirstName', 'BuyerAgentLastName', 'CoBuyerAgentFirstName',
    # AOR codes
    'ListAgentAOR', 'BuyerAgentAOR', 'BuyerOfficeAOR',
    # School district fields
    'ElementarySchool', 'ElementarySchoolDistrict',
    'MiddleOrJuniorSchool', 'MiddleOrJuniorSchoolDistrict',
    'HighSchool', 'HighSchoolDistrict',
    # Tax fields
    'TaxYear', 'TaxAnnualAmount',
    # System metadata
    'OriginatingSystemName', 'OriginatingSystemSubName',
    'LotSizeDimensions',
    'BusinessType',
    # Listing-specific compensation fields (not a market price metric)
    'BuyerAgencyCompensationType',
    'BuyerAgencyCompensation',
]

essential_cols = {
    'ListPrice', 'OriginalListPrice', 'ClosePrice', 'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt',
    'CloseDate', 'PurchaseContractDate', 'ListingContractDate',
    'PropertyType', 'PropertySubType', 'MlsStatus',
    'CountyOrParish', 'City', 'PostalCode', 'StateOrProvince',
    'Latitude', 'Longitude',
}
extra_from_missing = [
    c for c in high_missing_cols
    if c not in essential_cols and c not in cols_to_drop
]
if extra_from_missing:
    print(f"  Also dropping {len(extra_from_missing)} columns with >90% missing: {extra_from_missing}")
    cols_to_drop.extend(extra_from_missing)

drop_actual = [c for c in cols_to_drop if c in df_listed.columns]
df_listed.drop(columns=drop_actual, inplace=True)
print(f"  Dropped {len(drop_actual)} columns. Remaining: {df_listed.shape[1]}")

# ── Step 3: Ensure numeric columns are properly typed ───────────────────────
print("\n-- Step 3: Coerce numeric column types --")
numeric_coerce = [
    'ListPrice', 'OriginalListPrice', 'ClosePrice',
    'LivingArea', 'LotSizeAcres', 'LotSizeArea', 'LotSizeSquareFeet',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt',
    'AboveGradeFinishedArea', 'BelowGradeFinishedArea', 'BuildingAreaTotal',
    'GarageSpaces', 'CoveredSpaces', 'ParkingTotal', 'FireplacesTotal',
    'AssociationFee', 'Latitude', 'Longitude', 'Stories', 'MainLevelBedrooms',
]
for col in numeric_coerce:
    if col in df_listed.columns:
        before_nulls = df_listed[col].isnull().sum()
        df_listed[col] = pd.to_numeric(df_listed[col], errors='coerce')
        new_nulls = df_listed[col].isnull().sum() - before_nulls
        if new_nulls > 0:
            print(f"  {col}: {new_nulls} non-numeric values -> NaN")

# ── Step 4: Remove structurally invalid records ──────────────────────────────
# For listings, ListPrice > 0 is the key validity check —
# a listing without a price has no analytical value.
# We allow missing ClosePrice (most active listings won't have one).
print("\n-- Step 4: Remove invalid records --")
rows_start_step4 = len(df_listed)

if 'ListPrice' in df_listed.columns:
    before = len(df_listed)
    df_listed = df_listed[df_listed['ListPrice'].notna() & (df_listed['ListPrice'] > 0)]
    print(f"  Removed {before - len(df_listed):,} rows  — ListPrice ≤ 0 or missing")

if 'LivingArea' in df_listed.columns:
    before = len(df_listed)
    df_listed = df_listed[df_listed['LivingArea'].isna() | (df_listed['LivingArea'] > 0)]
    print(f"  Removed {before - len(df_listed):,} rows  — LivingArea ≤ 0")

if 'DaysOnMarket' in df_listed.columns:
    before = len(df_listed)
    df_listed = df_listed[df_listed['DaysOnMarket'].isna() | (df_listed['DaysOnMarket'] >= 0)]
    print(f"  Removed {before - len(df_listed):,} rows  — DaysOnMarket < 0")

for col in ['BedroomsTotal', 'BathroomsTotalInteger']:
    if col in df_listed.columns:
        before = len(df_listed)
        df_listed = df_listed[df_listed[col].isna() | (df_listed[col] >= 0)]
        print(f"  Removed {before - len(df_listed):,} rows  — {col} < 0")

print(f"\n  Total removed in this step: {rows_start_step4 - len(df_listed):,}")
print(f"  Rows remaining            : {len(df_listed):,}")

# ── Step 5: Date consistency flags ───────────────────────────────────────────
print("\n-- Step 5: Date consistency flags --")

if 'ListingContractDate' in df_listed.columns and 'CloseDate' in df_listed.columns:
    both   = df_listed['ListingContractDate'].notna() & df_listed['CloseDate'].notna()
    df_listed['listing_after_close_flag'] = (
        both & (df_listed['ListingContractDate'] > df_listed['CloseDate'])
    )
    n = df_listed['listing_after_close_flag'].sum()
    print(f"  listing_after_close_flag     : {n:,} records  ({n/len(df_listed)*100:.2f}%)")

if 'PurchaseContractDate' in df_listed.columns and 'CloseDate' in df_listed.columns:
    both   = df_listed['PurchaseContractDate'].notna() & df_listed['CloseDate'].notna()
    df_listed['purchase_after_close_flag'] = (
        both & (df_listed['PurchaseContractDate'] > df_listed['CloseDate'])
    )
    n = df_listed['purchase_after_close_flag'].sum()
    print(f"  purchase_after_close_flag    : {n:,} records  ({n/len(df_listed)*100:.2f}%)")

flag_cols_avail = [c for c in ['listing_after_close_flag', 'purchase_after_close_flag'] if c in df_listed.columns]
if flag_cols_avail:
    df_listed['negative_timeline_flag'] = df_listed[flag_cols_avail].any(axis=1)
    n = df_listed['negative_timeline_flag'].sum()
    print(f"  negative_timeline_flag (OR)  : {n:,} records  ({n/len(df_listed)*100:.2f}%)")

# ── Step 6: Geographic quality flags ────────────────────────────────────────
print("\n-- Step 6: Geographic quality flags --")

if 'Latitude' in df_listed.columns and 'Longitude' in df_listed.columns:

    df_listed['missing_coords_flag'] = df_listed['Latitude'].isna() | df_listed['Longitude'].isna()
    n = df_listed['missing_coords_flag'].sum()
    print(f"  missing_coords_flag          : {n:,} records  ({n/len(df_listed)*100:.2f}%)")

    # (0, 0) is in the Atlantic Ocean — a common "unknown" placeholder in MLS systems
    df_listed['zero_coords_flag'] = (df_listed['Latitude'] == 0) & (df_listed['Longitude'] == 0)
    n = df_listed['zero_coords_flag'].sum()
    print(f"  zero_coords_flag             : {n:,} records")

    # California must have NEGATIVE longitude — west of the prime meridian
    df_listed['positive_longitude_flag'] = (
        df_listed['Longitude'].notna() & (df_listed['Longitude'] > 0)
    )
    n = df_listed['positive_longitude_flag'].sum()
    print(f"  positive_longitude_flag      : {n:,} records")

    # California bounding box: lat 32.5–42.1, lng -124.5 to -114.1
    lat_ok   = df_listed['Latitude'].between(32.5, 42.1)
    lng_ok   = df_listed['Longitude'].between(-124.5, -114.1)
    has_both = df_listed['Latitude'].notna() & df_listed['Longitude'].notna()
    df_listed['out_of_state_flag'] = has_both & ~(lat_ok & lng_ok)
    n = df_listed['out_of_state_flag'].sum()
    print(f"  out_of_state_flag (CA bounds): {n:,} records")

print(f"\nRow count after cleaning: {len(df_listed):,}  (started: {rows_before_cleaning:,}, removed: {rows_before_cleaning - len(df_listed):,})")

df_listed.to_csv("data/listed_cleaned.csv", index=False)
print(f"\n[OK] Saved: data/listed_cleaned.csv  ({len(df_listed):,} rows × {df_listed.shape[1]} columns)")


# =============================================================================
# WEEK 6 — FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("WEEK 6 — FEATURE ENGINEERING")
print("=" * 70)

# ── Price Ratio: ListPrice / OriginalListPrice ───────────────────────────────
# WHY: for listings, this reveals whether the seller has reduced their price.
# Ratio < 1.0 = price was cut (seller is more motivated or market rejected price)
# Ratio = 1.0 = no change from original listing
# Ratio > 1.0 = price was raised (rare — usually a re-list at higher price)
if 'ListPrice' in df_listed.columns and 'OriginalListPrice' in df_listed.columns:
    df_listed['price_ratio'] = np.where(
        df_listed['OriginalListPrice'] > 0,
        (df_listed['ListPrice'] / df_listed['OriginalListPrice']).round(4),
        np.nan
    )
    print(f"  price_ratio (ListPrice / OriginalListPrice)    : computed")

# ── Price Per Square Foot: ListPrice / LivingArea ───────────────────────────
# Normalizes list price by size for fair cross-property comparisons
if 'ListPrice' in df_listed.columns and 'LivingArea' in df_listed.columns:
    df_listed['price_per_sqft'] = np.where(
        df_listed['LivingArea'] > 0,
        (df_listed['ListPrice'] / df_listed['LivingArea']).round(2),
        np.nan
    )
    print(f"  price_per_sqft (ListPrice / LivingArea)        : computed")

# ── Close to Original List Ratio ─────────────────────────────────────────────
# For listings that closed, this is the actual close ratio; for active listings
# that haven't closed, this computes ListPrice / OriginalListPrice instead.
# We use ClosePrice if available, otherwise fall back to ListPrice.
if 'OriginalListPrice' in df_listed.columns:
    if 'ClosePrice' in df_listed.columns:
        numerator = df_listed['ClosePrice'].where(
            df_listed['ClosePrice'].notna() & (df_listed['ClosePrice'] > 0),
            df_listed['ListPrice']
        )
    else:
        numerator = df_listed['ListPrice']
    df_listed['close_to_original_list_ratio'] = np.where(
        df_listed['OriginalListPrice'] > 0,
        (numerator / df_listed['OriginalListPrice']).round(4),
        np.nan
    )
    print(f"  close_to_original_list_ratio                   : computed")

# ── Listing to Contract Days ─────────────────────────────────────────────────
# Days from initial listing to going under contract — market demand speed signal
if 'PurchaseContractDate' in df_listed.columns and 'ListingContractDate' in df_listed.columns:
    df_listed['listing_to_contract_days'] = (
        df_listed['PurchaseContractDate'] - df_listed['ListingContractDate']
    ).dt.days
    print(f"  listing_to_contract_days                       : computed")

# ── Contract to Close Days ───────────────────────────────────────────────────
# Days from going under contract to closing — reflects transaction complexity
if 'CloseDate' in df_listed.columns and 'PurchaseContractDate' in df_listed.columns:
    df_listed['contract_to_close_days'] = (
        df_listed['CloseDate'] - df_listed['PurchaseContractDate']
    ).dt.days
    print(f"  contract_to_close_days                         : computed")

# ── Time columns from ListingContractDate ────────────────────────────────────
# KEY DIFFERENCE FROM SOLD: we use ListingContractDate, not CloseDate.
# WHY: for active listings, CloseDate is often missing. The listing date is the
# event we know happened — it's when the property entered the market.
if 'ListingContractDate' in df_listed.columns:
    df_listed['listing_year']  = df_listed['ListingContractDate'].dt.year
    df_listed['listing_month'] = df_listed['ListingContractDate'].dt.month
    df_listed['listing_yrmo']  = df_listed['ListingContractDate'].dt.to_period('M').astype(str)
    print(f"  listing_year, listing_month, listing_yrmo      : computed from ListingContractDate")

# ── Print sample output ──────────────────────────────────────────────────────
print("\n--- Sample rows showing new engineered columns (5 rows) ---")
show_cols = [
    'ListPrice', 'OriginalListPrice', 'LivingArea', 'ListingContractDate',
    'price_ratio', 'price_per_sqft', 'listing_to_contract_days',
    'contract_to_close_days', 'listing_year', 'listing_month', 'listing_yrmo',
]
show_cols = [c for c in show_cols if c in df_listed.columns]
sample = df_listed[show_cols].dropna(subset=['price_per_sqft']).head(5)
print(sample.to_string())

# ── Segmented summary by CountyOrParish ─────────────────────────────────────
print("\n--- County Summary (segmented by CountyOrParish) ---")
if 'CountyOrParish' in df_listed.columns:
    county_summary = df_listed.groupby('CountyOrParish').agg(
        listing_count           = ('ListPrice',               'count'),
        median_list_price       = ('ListPrice',               'median'),
        median_price_per_sqft   = ('price_per_sqft',          'median'),
        median_price_ratio      = ('price_ratio',             'median'),
        median_dom              = ('DaysOnMarket',            'median'),
        median_list_to_contract = ('listing_to_contract_days','median'),
    ).sort_values('median_list_price', ascending=False)
    print(county_summary.to_string())
    county_summary.to_csv("output/reports/listed_county_summary.csv")
    print("\n[OK] County summary -> output/reports/listed_county_summary.csv")


# =============================================================================
# WEEK 7 — OUTLIER DETECTION
# =============================================================================
print("\n" + "=" * 70)
print("WEEK 7 — OUTLIER DETECTION")
print("=" * 70)

# IQR method: flags values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] as outliers.
# We flag (not delete) to preserve the full dataset in listed_flagged.csv.
# Listed_final.csv is the filtered version with outliers removed.
# KEY DIFFERENCE FROM SOLD: we apply IQR to ListPrice (not ClosePrice).
print("\nApplying IQR outlier detection (flag only, records NOT deleted):")

outlier_fields = ['ListPrice', 'LivingArea', 'DaysOnMarket']

for field in outlier_fields:
    if field not in df_listed.columns:
        continue
    col = df_listed[field].dropna()
    if col.empty:
        continue

    q1    = col.quantile(0.25)
    q3    = col.quantile(0.75)
    iqr   = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    flag_col = f'{field}_outlier_flag'
    df_listed[flag_col] = (df_listed[field] < lower) | (df_listed[field] > upper)

    n_flagged = df_listed[flag_col].sum()
    pct       = n_flagged / len(df_listed) * 100
    print(f"\n  {field}:")
    print(f"    Q1={q1:,.2f}  Q3={q3:,.2f}  IQR={iqr:,.2f}")
    print(f"    Lower fence={lower:,.2f}  Upper fence={upper:,.2f}")
    print(f"    Flagged as outlier: {n_flagged:,}  ({pct:.1f}%)")

# Save full dataset with outlier flag columns (all rows kept)
df_listed.to_csv("data/listed_flagged.csv", index=False)
print(f"\n[OK] Saved: data/listed_flagged.csv  ({len(df_listed):,} rows × {df_listed.shape[1]} columns)")

# Build listed_final.csv: remove any row where at least one outlier flag is True
outlier_flag_cols = [f'{f}_outlier_flag' for f in outlier_fields if f'{f}_outlier_flag' in df_listed.columns]
if outlier_flag_cols:
    is_outlier  = df_listed[outlier_flag_cols].any(axis=1)
    df_final    = df_listed[~is_outlier].copy()
else:
    df_final    = df_listed.copy()

rows_removed_outlier = len(df_listed) - len(df_final)
print(f"\nOutlier removal summary:")
print(f"  Rows before removal : {len(df_listed):,}")
print(f"  Rows removed        : {rows_removed_outlier:,}  (any outlier flag = True)")
print(f"  Rows remaining      : {len(df_final):,}")

print(f"\nMedian comparison (flagged full set vs final clean set):")
for field in outlier_fields:
    if field in df_listed.columns:
        med_before = df_listed[field].median()
        med_after  = df_final[field].median()
        shift = med_after - med_before
        print(f"  {field:20s}: {med_before:>12,.2f} -> {med_after:>12,.2f}  (shift: {shift:+,.2f})")

df_final.to_csv("data/listed_final.csv", index=False)
print(f"\n[OK] Saved: data/listed_final.csv  ({len(df_final):,} rows × {df_final.shape[1]} columns)")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY — LISTING PIPELINE COMPLETE")
print("=" * 70)

print(f"\nlisted_final.csv (clean analytical dataset):")
print(f"  Total rows    : {len(df_final):,}")
print(f"  Total columns : {df_final.shape[1]}")

if 'ListingContractDate' in df_final.columns:
    list_dates = df_final['ListingContractDate'].dropna()
    if not list_dates.empty:
        print(f"  Date range    : {list_dates.min().date()} -> {list_dates.max().date()}")

print(f"\nKey stats (outliers removed):")
for field in ['ListPrice', 'price_per_sqft', 'DaysOnMarket', 'mortgage_rate_30yr']:
    if field in df_final.columns:
        col = df_final[field].dropna()
        if not col.empty:
            print(f"  {field:30s}: median={col.median():>10,.2f}  mean={col.mean():>10,.2f}  "
                  f"min={col.min():>10,.2f}  max={col.max():>10,.2f}")

print("\nOutput files generated:")
print("  data/listed_combined.csv        — Residential records from all monthly files")
print("  data/listed_enriched.csv        — + 30-yr mortgage rate merged by listing month")
print("  data/listed_cleaned.csv         — + cleaned, typed, date/geo quality flags")
print("  data/listed_flagged.csv         — + IQR outlier flags (all rows preserved)")
print("  data/listed_final.csv           — outliers removed (primary analytical dataset)")
print("  output/reports/listed_missing_value_report.csv")
print("  output/reports/listed_numeric_distribution.csv")
print("  output/reports/listed_county_summary.csv")
print("=" * 70)
