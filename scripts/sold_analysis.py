"""
sold_analysis.py — Complete CRMLS Sold Data Pipeline (Weeks 1–7)

Pipeline stages:
  Week 1      : Load + concatenate all monthly sold files, filter to Residential
  Weeks 2-3   : EDA — shape, dtypes, missing values, distributions, business questions
  Weeks 2-3   : Mortgage rate enrichment via FRED API
  Weeks 4-5   : Data cleaning — dates, column pruning, invalid values, flags
  Week 6      : Feature engineering — derived price/time metrics
  Week 7      : Outlier detection via IQR method

Run from the project root:
  python scripts/sold_analysis.py
"""

import sys
import pandas as pd
import glob
import os
import io
import numpy as np
import urllib.request

# Force UTF-8 output so Unicode characters (arrows, check marks, etc.)
# print correctly regardless of the Windows console's default encoding.
sys.stdout.reconfigure(encoding='utf-8')

# ── Working directory: always run relative to the project root ───────────────
# os.path.abspath(__file__)  -> full path to this script file
# os.path.dirname(...)       -> the directory it lives in (scripts/)
# os.path.dirname(...) again -> one level up (the project root)
# os.chdir(...)              -> changes Python's cwd so "data/..." paths work
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Create output folders if they don't already exist.
# exist_ok=True means: don't raise an error if the folder is already there.
os.makedirs("output/reports", exist_ok=True)

print("=" * 70)
print("SOLD DATA PIPELINE — CRMLS MLS Data")
print("=" * 70)


# =============================================================================
# WEEK 1 — AGGREGATION
# Goal: combine all monthly sold CSVs into one Residential-only dataset
# =============================================================================
print("\n" + "=" * 70)
print("WEEK 1 — AGGREGATION")
print("=" * 70)

# glob.glob finds every file whose name matches the pattern.
# The * wildcard matches any characters — here it catches all months:
# CRMLSSold202401.csv, CRMLSSold202402.csv, etc.
# sorted() loads them in chronological order (Jan 2024 first).
import re
# Only match files with a 6-digit YYYYMM suffix to exclude derived files
# like CRMLSSold_Cleaned.csv that also match the CRMLSSold* pattern.
sold_files = sorted(
    f for f in glob.glob("data/CRMLSSold*.csv")
    if re.search(r'CRMLSSold\d{6}\.csv$', f)
)
print(f"Found {len(sold_files)} sold CSV files\n")

# Read every monthly CSV into a list of DataFrames.
# low_memory=False: pandas scans the whole column at once to infer types
#   correctly, instead of guessing row-by-row (which causes mixed-type warnings).
# encoding='latin-1': handles accented characters (é, ñ, etc.) that appear in
#   agent names and addresses and would crash plain UTF-8 reading.
dfs = []
for f in sold_files:
    df_temp = pd.read_csv(f, low_memory=False, encoding='latin-1')
    print(f"  {os.path.basename(f)}: {len(df_temp):,} rows")
    dfs.append(df_temp)

total_before_concat = sum(len(d) for d in dfs)
print(f"\nRow count BEFORE concat (sum of all files): {total_before_concat:,}")

# pd.concat stacks all DataFrames vertically — same as SQL UNION ALL.
# ignore_index=True resets the row index to run 0, 1, 2, ... continuously
# so we don't have repeated indices from each monthly file.
df_all = pd.concat(dfs, ignore_index=True)
print(f"Row count AFTER  concat                   : {len(df_all):,}")
print(f"Total columns                             : {df_all.shape[1]}")

# Show the full property type breakdown BEFORE filtering.
# This is important context: how much of the MLS is residential vs commercial/land?
print("\nPropertyType distribution BEFORE filter:")
print(df_all['PropertyType'].value_counts(dropna=False).to_string())
print()

rows_before_filter = len(df_all)

# Filter to Residential only.
# WHY: Commercial, Land, CommercialLease have completely different price scales
# and valuation logic. Mixing them would distort every average and distribution.
# .copy() creates an independent DataFrame so later edits don't trigger
# the "SettingWithCopyWarning" that appears when modifying a filtered view.
df_sold = df_all[df_all['PropertyType'] == 'Residential'].copy()

rows_after_filter = len(df_sold)
print(f"Row count BEFORE PropertyType filter       : {rows_before_filter:,}")
print(f"Row count AFTER  filtering to Residential  : {rows_after_filter:,}")
print(f"Rows removed (non-residential)             : {rows_before_filter - rows_after_filter:,}")

# Save the combined, unmodified Residential dataset.
# This is the Week 1 checkpoint — raw data, no cleaning yet.
# index=False prevents writing a useless "0, 1, 2..." column into the CSV.
df_sold.to_csv("data/sold_combined.csv", index=False)
print(f"\n[OK] Saved: data/sold_combined.csv  ({len(df_sold):,} rows × {df_sold.shape[1]} columns)")


# =============================================================================
# WEEKS 2-3 — STRUCTURING, VALIDATION, AND EDA
# Goal: understand the shape and quality of the data before doing anything to it
# =============================================================================
print("\n" + "=" * 70)
print("WEEKS 2-3 — STRUCTURING, VALIDATION, AND EDA")
print("=" * 70)

# ── Shape, dtypes, column names ─────────────────────────────────────────────
# .shape returns a tuple (rows, columns).
# .dtypes tells you what data type pandas assigned to each column:
#   object  = text or mixed types (dates come in as object from CSV)
#   float64 = decimal numbers
#   int64   = whole numbers
print(f"\nDataset shape: {df_sold.shape[0]:,} rows × {df_sold.shape[1]} columns")
print("\nAll column names:")
print(list(df_sold.columns))
print("\nData types per column:")
print(df_sold.dtypes.to_string())

# ── Missing value analysis ───────────────────────────────────────────────────
# .isnull() returns a True/False table — True wherever a cell is empty.
# .sum() counts the True values per column.
# Divide by total rows × 100 to get missing percentage.
print("\n\n--- Missing Value Analysis ---")
missing_count = df_sold.isnull().sum()
missing_pct   = (missing_count / len(df_sold) * 100).round(2)

# Build a clean summary table sorted by worst (most missing) first
missing_summary = pd.DataFrame({
    'missing_count': missing_count,
    'missing_pct':   missing_pct,
}).sort_values('missing_pct', ascending=False)

print(missing_summary.to_string())

# Columns with >90% missing are nearly useless for analysis.
# We track them here and will drop them in the cleaning step.
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

missing_summary.to_csv("output/reports/sold_missing_value_report.csv")
print("\n[OK] Missing value report -> output/reports/sold_missing_value_report.csv")

# ── Unique PropertyType values ───────────────────────────────────────────────
# Even after filtering, this confirms the filter worked correctly.
print("\n--- Unique PropertyType values (after Residential filter) ---")
print(df_sold['PropertyType'].value_counts(dropna=False).to_string())

# ── Numeric distribution summary ────────────────────────────────────────────
# For each key metric we compute a percentile profile:
#   p25 = 25th percentile (lower quartile)
#   p50 = median (50th percentile) — robust to extreme values
#   p75 = 75th percentile (upper quartile)
#   p90 = 90th percentile — threshold beyond which homes are significantly pricier/larger
#   p99 = 99th percentile — near-extreme values
# WHY percentiles instead of just mean? The mean is heavily pulled by outliers
# (a $20M mansion skews the average but not the median).
print("\n--- Numeric Distribution Summary ---")
numeric_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt',
]
# Only include fields that actually exist — defensive check in case column names differ
numeric_fields = [f for f in numeric_fields if f in df_sold.columns]

dist_rows = []
for field in numeric_fields:
    # pd.to_numeric converts the column to numbers; errors='coerce' turns any
    # non-numeric value (like "N/A" text) into NaN so the math still works.
    # .dropna() removes NaN before computing percentiles — np.percentile
    # can't handle NaN values.
    col = pd.to_numeric(df_sold[field], errors='coerce').dropna()
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
dist_df.to_csv("output/reports/sold_numeric_distribution.csv")
print("\n[OK] Numeric distribution -> output/reports/sold_numeric_distribution.csv")

# ── Business Questions ───────────────────────────────────────────────────────
print("\n\n--- Business Questions ---")

# Q1: Residential vs other property type share
# We use df_all (before filtering) as the denominator for a true share.
res_count   = (df_all['PropertyType'] == 'Residential').sum()
total_count = len(df_all)
print(f"\nQ1 — Property Type Share (of all {total_count:,} records):")
print(f"  Residential     : {res_count:,}  ({res_count/total_count*100:.1f}%)")
print(f"  Non-residential : {total_count - res_count:,}  ({(total_count - res_count)/total_count*100:.1f}%)")
print("  Non-residential breakdown:")
non_res = df_all[df_all['PropertyType'] != 'Residential']
print(non_res['PropertyType'].value_counts(dropna=False).to_string())

# Q2: Median and average close prices
# WHY median matters more than mean in real estate: the mean is pulled upward
# by rare ultra-high sales ($10M+ homes). Median = middle of the market.
median_price = df_sold['ClosePrice'].median()
mean_price   = df_sold['ClosePrice'].mean()
print(f"\nQ2 — Close Price Statistics (Residential):")
print(f"  Median close price : ${median_price:,.0f}")
print(f"  Average close price: ${mean_price:,.0f}")
print(f"  Note: mean > median by ${mean_price - median_price:,.0f} — indicates right-skew from high-end sales")

# Q3: Days on Market distribution
dom_valid = pd.to_numeric(df_sold['DaysOnMarket'], errors='coerce').dropna()
print(f"\nQ3 — Days on Market Distribution ({dom_valid.count():,} valid records):")
print(f"  Min      : {dom_valid.min():.0f} days")
print(f"  Median   : {dom_valid.median():.0f} days  (half of homes sold faster than this)")
print(f"  Average  : {dom_valid.mean():.1f} days")
print(f"  75th pct : {np.percentile(dom_valid, 75):.0f} days")
print(f"  90th pct : {np.percentile(dom_valid, 90):.0f} days  (10% of homes took longer)")
print(f"  99th pct : {np.percentile(dom_valid, 99):.0f} days")
print(f"  Max      : {dom_valid.max():.0f} days")

# Q4: Percentage sold above vs below list price
# ratio > 1.0 -> sold ABOVE list (competitive bidding)
# ratio < 1.0 -> sold BELOW list (negotiated down or price reduction)
# ratio = 1.0 -> exactly at list price
price_valid = df_sold[
    df_sold['ClosePrice'].notna() &
    df_sold['ListPrice'].notna() &
    (df_sold['ListPrice'] > 0)
].copy()
price_valid['_ratio'] = price_valid['ClosePrice'] / price_valid['ListPrice']

above = (price_valid['_ratio'] > 1.0).sum()
at    = (price_valid['_ratio'] == 1.0).sum()
below = (price_valid['_ratio'] < 1.0).sum()
n_pv  = len(price_valid)

print(f"\nQ4 — Sold vs List Price ({n_pv:,} valid records):")
print(f"  Sold ABOVE list price : {above:,}  ({above/n_pv*100:.1f}%)")
print(f"  Sold AT list price    : {at:,}  ({at/n_pv*100:.1f}%)")
print(f"  Sold BELOW list price : {below:,}  ({below/n_pv*100:.1f}%)")

# Q5: Date consistency issues — close date before listing date is impossible
# pd.to_datetime converts text dates to datetime objects.
# errors='coerce' turns bad/missing date strings into NaT (Not a Time = null date).
close_dt = pd.to_datetime(df_sold['CloseDate'],           errors='coerce')
list_dt  = pd.to_datetime(df_sold['ListingContractDate'], errors='coerce')
purch_dt = pd.to_datetime(df_sold['PurchaseContractDate'], errors='coerce')

both_valid_cl = close_dt.notna() & list_dt.notna()
close_before_list = (close_dt < list_dt) & both_valid_cl

both_valid_cp = close_dt.notna() & purch_dt.notna()
purch_after_close = (purch_dt > close_dt) & both_valid_cp

print(f"\nQ5 — Date Consistency Issues:")
print(f"  Records with CloseDate + ListingContractDate    : {both_valid_cl.sum():,}")
print(f"  CloseDate BEFORE ListingContractDate (invalid)  : {close_before_list.sum():,}  ({close_before_list.sum()/both_valid_cl.sum()*100:.2f}%)")
print(f"  Records with CloseDate + PurchaseContractDate   : {both_valid_cp.sum():,}")
print(f"  PurchaseContractDate AFTER CloseDate (unusual)  : {purch_after_close.sum():,}  ({purch_after_close.sum()/both_valid_cp.sum()*100:.2f}%)")

# Q6: Counties with highest median close prices
# .groupby() splits the data by county, .agg() computes stats per group
county_stats = (
    df_sold.groupby('CountyOrParish')['ClosePrice']
    .agg(median_close_price='median', sale_count='count')
    .sort_values('median_close_price', ascending=False)
)
print(f"\nQ6 — Top 10 Counties by Median Close Price:")
print(county_stats.head(10).to_string())
print(f"\n  (Full list: {len(county_stats)} counties total)")


# =============================================================================
# WEEKS 2-3 — MORTGAGE RATE ENRICHMENT
# Goal: attach the prevailing 30-year fixed rate to each sale by its close month
# WHY: mortgage rates directly affect affordability. Knowing the rate at the
# time of each sale lets us analyze how price trends tracked rate changes.
# =============================================================================
print("\n" + "=" * 70)
print("WEEKS 2-3 — MORTGAGE RATE ENRICHMENT")
print("=" * 70)

# FRED (Federal Reserve Economic Data) publishes the weekly national average
# 30-year fixed mortgage rate as series MORTGAGE30US.
# We download it as a CSV, resample from weekly to monthly average,
# then merge it onto the sold dataset by the year-month of each CloseDate.
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
print(f"\nFetching MORTGAGE30US series from FRED...")
print(f"  URL: {FRED_URL}")

try:
    with urllib.request.urlopen(FRED_URL, timeout=30) as resp:
        raw = resp.read().decode('utf-8')

    # Parse the downloaded CSV text into a DataFrame
    df_fred = pd.read_csv(io.StringIO(raw))
    df_fred.columns = ['date', 'mortgage_rate_30yr']

    # FRED uses '.' (a literal period) to represent missing observations —
    # not NaN, not blank, but a dot character. We convert '.' to NaN first,
    # then cast to float, so numeric math works correctly.
    df_fred['mortgage_rate_30yr'] = pd.to_numeric(
        df_fred['mortgage_rate_30yr'].replace('.', np.nan), errors='coerce'
    )
    df_fred = df_fred.dropna(subset=['mortgage_rate_30yr'])
    df_fred['date'] = pd.to_datetime(df_fred['date'])

    print(f"  Raw FRED data: {len(df_fred):,} weekly observations")
    print(f"  Date range   : {df_fred['date'].min().date()} -> {df_fred['date'].max().date()}")
    print(f"  Rate range   : {df_fred['mortgage_rate_30yr'].min():.2f}% -> {df_fred['mortgage_rate_30yr'].max():.2f}%")

    # Resample weekly data to monthly averages.
    # .set_index('date') makes the date column the index so resample() can use it.
    # resample('MS') groups by Month Start (the 1st of each month).
    # .mean() averages all weekly values that fall in that month.
    # .reset_index() turns the index back into a regular column.
    df_fred_monthly = (
        df_fred.set_index('date')['mortgage_rate_30yr']
        .resample('MS')
        .mean()
        .reset_index()
    )
    df_fred_monthly['mortgage_rate_30yr'] = df_fred_monthly['mortgage_rate_30yr'].round(4)

    # Create a year_month merge key — e.g., "2024-01" — for both datasets.
    # dt.to_period('M') converts a date to its month period.
    # .astype(str) converts "2024-01" Period object to a plain string for merging.
    df_fred_monthly['year_month'] = df_fred_monthly['date'].dt.to_period('M').astype(str)

    print(f"\n  Monthly averages computed ({len(df_fred_monthly)} months). Last 6:")
    print(df_fred_monthly.tail(6)[['year_month', 'mortgage_rate_30yr']].to_string(index=False))

    # Create the same year_month key on the sold dataset using CloseDate
    df_sold['CloseDate_dt'] = pd.to_datetime(df_sold['CloseDate'], errors='coerce')
    df_sold['year_month'] = df_sold['CloseDate_dt'].dt.to_period('M').astype(str)

    # Left merge: every sold record keeps all its data and gets a rate if its
    # close month exists in the FRED data. Records with missing CloseDate get NaN.
    # how='left' = keep ALL rows from df_sold, fill NaN where no match in fred.
    df_sold = df_sold.merge(
        df_fred_monthly[['year_month', 'mortgage_rate_30yr']],
        on='year_month',
        how='left'
    )

    # Drop the temporary datetime column we created just for the merge key
    df_sold.drop(columns=['CloseDate_dt'], errors='ignore', inplace=True)

    # Validate: how many rows ended up with no rate?
    null_rates = df_sold['mortgage_rate_30yr'].isnull().sum()
    print(f"\n  Merge results:")
    print(f"    Total rows after merge       : {len(df_sold):,}")
    print(f"    Rows with null mortgage rate : {null_rates:,}",
          "(typically records with missing or out-of-range CloseDates)" if null_rates else "— all rates populated [OK]")

except Exception as e:
    # If the FRED fetch fails (network issue, etc.), we continue without rates
    # rather than crashing the whole pipeline. We add the column as NaN.
    print(f"\n  WARNING: FRED fetch failed — {e}")
    print("  Continuing without mortgage rate enrichment (column set to NaN).")
    df_sold['mortgage_rate_30yr'] = np.nan
    df_sold['year_month'] = pd.to_datetime(df_sold['CloseDate'], errors='coerce').dt.to_period('M').astype(str)

df_sold.to_csv("data/sold_enriched.csv", index=False)
print(f"\n[OK] Saved: data/sold_enriched.csv  ({len(df_sold):,} rows × {df_sold.shape[1]} columns)")


# =============================================================================
# WEEKS 4-5 — DATA CLEANING
# Goal: produce a trustworthy analytical dataset by fixing types, removing
# structurally invalid rows, and adding quality flags
# =============================================================================
print("\n" + "=" * 70)
print("WEEKS 4-5 — DATA CLEANING")
print("=" * 70)

rows_before_cleaning = len(df_sold)
print(f"\nStarting row count: {rows_before_cleaning:,}")

# ── Step 1: Convert date fields from strings to datetime ────────────────────
# CSV files store everything as text. Before we can subtract dates (to compute
# days between events) we must parse them into Python datetime objects.
# errors='coerce' converts unparseable strings to NaT (Not a Time = null date)
# instead of raising an exception.
print("\n-- Step 1: Convert date columns to datetime --")
date_cols = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
for col in date_cols:
    if col in df_sold.columns:
        before_nulls = df_sold[col].isnull().sum()
        df_sold[col] = pd.to_datetime(df_sold[col], errors='coerce')
        after_nulls  = df_sold[col].isnull().sum()
        new_nulls    = after_nulls - before_nulls
        status = f"({new_nulls} unparseable -> NaT)" if new_nulls > 0 else "(all parsed cleanly)"
        print(f"  {col}: converted  {status}")

# ── Step 2: Drop unnecessary or redundant columns ───────────────────────────
# WHY drop columns at all? Fewer columns = faster processing, smaller CSV files,
# and easier-to-read analysis. We remove:
#   - Agent personal info (names, emails): not useful for market analysis;
#     fine for compliance tracking but not for pricing or geographic work
#   - AOR (Association of Realtors) codes: internal MLS routing, not analytical
#   - School district fields: out of scope for our market analysis
#   - Tax fields: TaxAnnualAmount is missing for most records; not central
#   - System metadata: OriginatingSystemName is just "CRMLS" for everything
#   - LotSizeDimensions: a free-text field like "100x200" — superseded by
#     numeric LotSizeAcres / LotSizeSquareFeet
#   - BusinessType: meaningful for commercial, irrelevant for residential
print("\n-- Step 2: Drop non-analytical columns --")
cols_to_drop = [
    # Agent personal info
    'ListAgentEmail', 'ListAgentFirstName', 'ListAgentLastName', 'ListAgentFullName',
    'CoListAgentFirstName', 'CoListAgentLastName',
    'BuyerAgentFirstName', 'BuyerAgentLastName', 'CoBuyerAgentFirstName',
    # AOR (broker association codes — internal routing)
    'ListAgentAOR', 'BuyerAgentAOR', 'BuyerOfficeAOR',
    # School district fields (out of scope)
    'ElementarySchool', 'ElementarySchoolDistrict',
    'MiddleOrJuniorSchool', 'MiddleOrJuniorSchoolDistrict',
    'HighSchool', 'HighSchoolDistrict',
    # Tax fields (mostly missing, not central)
    'TaxYear', 'TaxAnnualAmount',
    # System metadata (same value for every row)
    'OriginatingSystemName', 'OriginatingSystemSubName',
    # Redundant / non-numeric text fields
    'LotSizeDimensions',  # e.g. "100x200" — superseded by LotSizeAcres
    'BusinessType',       # commercial field, meaningless for residential
]

# Also drop columns flagged as >90% missing (tracked in EDA section),
# but never drop columns that are essential to our analysis.
essential_cols = {
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea', 'LotSizeAcres',
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

# Only attempt to drop columns that actually exist
drop_actual = [c for c in cols_to_drop if c in df_sold.columns]
df_sold.drop(columns=drop_actual, inplace=True)
print(f"  Dropped {len(drop_actual)} columns. Remaining: {df_sold.shape[1]}")

# ── Step 3: Ensure numeric columns are properly typed ───────────────────────
# Sometimes numeric columns get read as 'object' (text) if even one cell
# contained something like "—" or "N/A". pd.to_numeric with errors='coerce'
# forces conversion and turns any leftovers into NaN.
print("\n-- Step 3: Coerce numeric column types --")
numeric_coerce = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres', 'LotSizeArea', 'LotSizeSquareFeet',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt',
    'AboveGradeFinishedArea', 'BelowGradeFinishedArea', 'BuildingAreaTotal',
    'GarageSpaces', 'CoveredSpaces', 'ParkingTotal', 'FireplacesTotal',
    'AssociationFee', 'Latitude', 'Longitude', 'Stories', 'MainLevelBedrooms',
]
for col in numeric_coerce:
    if col in df_sold.columns:
        before_nulls = df_sold[col].isnull().sum()
        df_sold[col] = pd.to_numeric(df_sold[col], errors='coerce')
        new_nulls = df_sold[col].isnull().sum() - before_nulls
        if new_nulls > 0:
            print(f"  {col}: {new_nulls} non-numeric values -> NaN")

# ── Step 4: Remove structurally invalid records ──────────────────────────────
# These are logical impossibilities — not just outliers, but values that make
# analysis fundamentally meaningless (can't compute price/sqft without LivingArea).
print("\n-- Step 4: Remove invalid records --")
rows_start_step4 = len(df_sold)

# ClosePrice must be positive: a $0 or negative sale doesn't represent a real transaction
if 'ClosePrice' in df_sold.columns:
    before = len(df_sold)
    df_sold = df_sold[df_sold['ClosePrice'].notna() & (df_sold['ClosePrice'] > 0)]
    print(f"  Removed {before - len(df_sold):,} rows  — ClosePrice ≤ 0 or missing")

# LivingArea must be positive: we need it to compute price-per-sqft
# We allow NaN (not every record has sq footage), but never 0 or negative.
if 'LivingArea' in df_sold.columns:
    before = len(df_sold)
    df_sold = df_sold[df_sold['LivingArea'].isna() | (df_sold['LivingArea'] > 0)]
    print(f"  Removed {before - len(df_sold):,} rows  — LivingArea ≤ 0")

# DaysOnMarket cannot be negative — time doesn't run backwards
if 'DaysOnMarket' in df_sold.columns:
    before = len(df_sold)
    df_sold = df_sold[df_sold['DaysOnMarket'].isna() | (df_sold['DaysOnMarket'] >= 0)]
    print(f"  Removed {before - len(df_sold):,} rows  — DaysOnMarket < 0")

# Bedrooms and bathrooms must be non-negative
for col in ['BedroomsTotal', 'BathroomsTotalInteger']:
    if col in df_sold.columns:
        before = len(df_sold)
        df_sold = df_sold[df_sold[col].isna() | (df_sold[col] >= 0)]
        print(f"  Removed {before - len(df_sold):,} rows  — {col} < 0")

rows_removed_step4 = rows_start_step4 - len(df_sold)
print(f"\n  Total removed in this step: {rows_removed_step4:,}")
print(f"  Rows remaining            : {len(df_sold):,}")

# ── Step 5: Add date consistency flags ───────────────────────────────────────
# Instead of deleting suspicious date records (which might be real edge cases),
# we add boolean flag columns. The analyst can choose to filter on these later.
# True = the flag condition was met (a potential data issue).
print("\n-- Step 5: Date consistency flags --")

# listing_after_close_flag: the listing date is AFTER the close date.
# This is physically impossible — you can't close before you list.
if 'ListingContractDate' in df_sold.columns and 'CloseDate' in df_sold.columns:
    both   = df_sold['ListingContractDate'].notna() & df_sold['CloseDate'].notna()
    df_sold['listing_after_close_flag'] = (
        both & (df_sold['ListingContractDate'] > df_sold['CloseDate'])
    )
    n = df_sold['listing_after_close_flag'].sum()
    print(f"  listing_after_close_flag     : {n:,} records  ({n/len(df_sold)*100:.2f}%)")

# purchase_after_close_flag: contract date is AFTER the close date.
# A contract must be signed before you can close escrow.
if 'PurchaseContractDate' in df_sold.columns and 'CloseDate' in df_sold.columns:
    both   = df_sold['PurchaseContractDate'].notna() & df_sold['CloseDate'].notna()
    df_sold['purchase_after_close_flag'] = (
        both & (df_sold['PurchaseContractDate'] > df_sold['CloseDate'])
    )
    n = df_sold['purchase_after_close_flag'].sum()
    print(f"  purchase_after_close_flag    : {n:,} records  ({n/len(df_sold)*100:.2f}%)")

# negative_timeline_flag: either of the above date issues is present
flag_cols_avail = [c for c in ['listing_after_close_flag', 'purchase_after_close_flag'] if c in df_sold.columns]
if flag_cols_avail:
    df_sold['negative_timeline_flag'] = df_sold[flag_cols_avail].any(axis=1)
    n = df_sold['negative_timeline_flag'].sum()
    print(f"  negative_timeline_flag (OR)  : {n:,} records  ({n/len(df_sold)*100:.2f}%)")

# ── Step 6: Geographic quality flags ────────────────────────────────────────
# California MLS data should have negative longitudes (west of prime meridian)
# and coordinates within California's bounding box. Anomalies suggest data errors.
print("\n-- Step 6: Geographic quality flags --")

if 'Latitude' in df_sold.columns and 'Longitude' in df_sold.columns:

    # missing_coords_flag: either Lat or Lng is NaN — can't map or geo-analyze this record
    df_sold['missing_coords_flag'] = df_sold['Latitude'].isna() | df_sold['Longitude'].isna()
    n = df_sold['missing_coords_flag'].sum()
    print(f"  missing_coords_flag          : {n:,} records  ({n/len(df_sold)*100:.2f}%)")

    # zero_coords_flag: (0, 0) is a common database placeholder for "unknown location"
    # (it's in the Atlantic Ocean off Africa — definitely not California)
    df_sold['zero_coords_flag'] = (df_sold['Latitude'] == 0) & (df_sold['Longitude'] == 0)
    n = df_sold['zero_coords_flag'].sum()
    print(f"  zero_coords_flag             : {n:,} records")

    # positive_longitude_flag: California is west of the prime meridian, so all
    # valid longitudes must be NEGATIVE (e.g., -117.5, not 117.5).
    # A positive Longitude indicates the sign was dropped during data entry.
    df_sold['positive_longitude_flag'] = (
        df_sold['Longitude'].notna() & (df_sold['Longitude'] > 0)
    )
    n = df_sold['positive_longitude_flag'].sum()
    print(f"  positive_longitude_flag      : {n:,} records")

    # out_of_state_flag: California's bounding box (approximate):
    #   Latitude  : 32.5° N  to 42.1° N
    #   Longitude : -124.5° W to -114.1° W
    # Anything outside these bounds is geographically outside California.
    lat_ok = df_sold['Latitude'].between(32.5, 42.1)
    lng_ok = df_sold['Longitude'].between(-124.5, -114.1)
    has_both = df_sold['Latitude'].notna() & df_sold['Longitude'].notna()
    df_sold['out_of_state_flag'] = has_both & ~(lat_ok & lng_ok)
    n = df_sold['out_of_state_flag'].sum()
    print(f"  out_of_state_flag (CA bounds): {n:,} records")

print(f"\nRow count after cleaning: {len(df_sold):,}  (started: {rows_before_cleaning:,}, removed: {rows_before_cleaning - len(df_sold):,})")

df_sold.to_csv("data/sold_cleaned.csv", index=False)
print(f"\n[OK] Saved: data/sold_cleaned.csv  ({len(df_sold):,} rows × {df_sold.shape[1]} columns)")


# =============================================================================
# WEEK 6 — FEATURE ENGINEERING
# Goal: create derived columns that make analysis richer and comparisons fairer
# =============================================================================
print("\n" + "=" * 70)
print("WEEK 6 — FEATURE ENGINEERING")
print("=" * 70)

# ── Price Ratio (ClosePrice / OriginalListPrice) ─────────────────────────────
# WHY: tells us how aggressively buyers bid relative to the original ask price.
# Ratio > 1.0 = bidding war (close above original list)
# Ratio < 1.0 = negotiation or price reduction before close
# np.where(condition, value_if_true, value_if_false) avoids division-by-zero errors
# by returning NaN whenever OriginalListPrice is 0 or negative.
if 'ClosePrice' in df_sold.columns and 'OriginalListPrice' in df_sold.columns:
    df_sold['price_ratio'] = np.where(
        df_sold['OriginalListPrice'] > 0,
        (df_sold['ClosePrice'] / df_sold['OriginalListPrice']).round(4),
        np.nan
    )
    print(f"  price_ratio (ClosePrice / OriginalListPrice)  : computed")

# ── Price Per Square Foot (ClosePrice / LivingArea) ──────────────────────────
# WHY: normalizes price by size so we can compare a 1,000 sqft condo to a
# 3,000 sqft house on equal footing. Essential metric for appraisers and analysts.
if 'ClosePrice' in df_sold.columns and 'LivingArea' in df_sold.columns:
    df_sold['price_per_sqft'] = np.where(
        df_sold['LivingArea'] > 0,
        (df_sold['ClosePrice'] / df_sold['LivingArea']).round(2),
        np.nan
    )
    print(f"  price_per_sqft (ClosePrice / LivingArea)      : computed")

# ── Close to Original List Ratio ─────────────────────────────────────────────
# Same formula as price_ratio — the spec requests both names so we include both.
# This one is explicitly labeled for reporting clarity.
if 'price_ratio' in df_sold.columns:
    df_sold['close_to_original_list_ratio'] = df_sold['price_ratio']
    print(f"  close_to_original_list_ratio (= price_ratio)  : computed")

# ── Listing to Contract Days (PurchaseContractDate − ListingContractDate) ────
# WHY: how quickly the home went under contract after hitting the market.
# A short window = high demand. This is more granular than DaysOnMarket.
# Subtracting two datetime columns returns a Timedelta; .dt.days extracts
# it as an integer number of days.
if 'PurchaseContractDate' in df_sold.columns and 'ListingContractDate' in df_sold.columns:
    df_sold['listing_to_contract_days'] = (
        df_sold['PurchaseContractDate'] - df_sold['ListingContractDate']
    ).dt.days
    print(f"  listing_to_contract_days                      : computed")

# ── Contract to Close Days (CloseDate − PurchaseContractDate) ────────────────
# WHY: reflects the escrow/closing period length. Longer = more complex
# financing or title issues. Shorter = smooth, often cash transactions.
if 'CloseDate' in df_sold.columns and 'PurchaseContractDate' in df_sold.columns:
    df_sold['contract_to_close_days'] = (
        df_sold['CloseDate'] - df_sold['PurchaseContractDate']
    ).dt.days
    print(f"  contract_to_close_days                        : computed")

# ── Time components from CloseDate ───────────────────────────────────────────
# WHY: breaking date into components lets us group by year, month, or year-month
# for trend charts and seasonal analysis without string manipulation.
# .dt.year/.dt.month access the year and month integer from a datetime column.
# dt.to_period('M') converts to a "2024-01" period, .astype(str) gives the string.
if 'CloseDate' in df_sold.columns:
    df_sold['close_year']  = df_sold['CloseDate'].dt.year
    df_sold['close_month'] = df_sold['CloseDate'].dt.month
    df_sold['close_yrmo']  = df_sold['CloseDate'].dt.to_period('M').astype(str)
    print(f"  close_year, close_month, close_yrmo           : computed from CloseDate")

# ── Print sample output ──────────────────────────────────────────────────────
print("\n--- Sample rows showing new engineered columns (5 rows) ---")
show_cols = [
    'ClosePrice', 'OriginalListPrice', 'LivingArea', 'CloseDate',
    'price_ratio', 'price_per_sqft', 'listing_to_contract_days',
    'contract_to_close_days', 'close_year', 'close_month', 'close_yrmo',
]
show_cols = [c for c in show_cols if c in df_sold.columns]
sample = df_sold[show_cols].dropna(subset=['price_per_sqft']).head(5)
print(sample.to_string())

# ── Segmented summary by CountyOrParish ─────────────────────────────────────
# WHY: county-level aggregation reveals regional price variation — which markets
# are expensive, which are affordable, how quickly homes sell per region.
# .groupby().agg() computes multiple statistics at once for each group.
print("\n--- County Summary (segmented by CountyOrParish) ---")
if 'CountyOrParish' in df_sold.columns:
    county_summary = df_sold.groupby('CountyOrParish').agg(
        sale_count              = ('ClosePrice',              'count'),
        median_close_price      = ('ClosePrice',              'median'),
        median_price_per_sqft   = ('price_per_sqft',          'median'),
        median_price_ratio      = ('price_ratio',             'median'),
        median_dom              = ('DaysOnMarket',            'median'),
        median_list_to_contract = ('listing_to_contract_days','median'),
    ).sort_values('median_close_price', ascending=False)
    print(county_summary.to_string())
    county_summary.to_csv("output/reports/sold_county_summary.csv")
    print("\n[OK] County summary -> output/reports/sold_county_summary.csv")


# =============================================================================
# WEEK 7 — OUTLIER DETECTION
# Goal: flag statistical outliers using the IQR method without deleting records
# =============================================================================
print("\n" + "=" * 70)
print("WEEK 7 — OUTLIER DETECTION")
print("=" * 70)

# The IQR (Interquartile Range) method:
#   Q1 = 25th percentile, Q3 = 75th percentile
#   IQR = Q3 - Q1  (the spread of the middle 50% of the data)
#   Lower fence = Q1 - 1.5 × IQR  (anything below this is a low outlier)
#   Upper fence = Q3 + 1.5 × IQR  (anything above this is a high outlier)
#
# WHY 1.5×IQR? It's the standard Tukey method. For a normal distribution,
# this catches ~0.7% of values as outliers — a reasonable threshold.
# WHY flag instead of delete? Context matters. A $15M sale in Beverly Hills
# is a legitimate outlier — we want it in the data for some analyses but
# excluded from median/mean summaries. Flags give flexibility.
print("\nApplying IQR outlier detection (flag only, records NOT deleted):")

outlier_fields = ['ClosePrice', 'LivingArea', 'DaysOnMarket']
rows_before_outlier = len(df_sold)

for field in outlier_fields:
    if field not in df_sold.columns:
        continue
    col = df_sold[field].dropna()
    if col.empty:
        continue

    q1    = col.quantile(0.25)
    q3    = col.quantile(0.75)
    iqr   = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    flag_col = f'{field}_outlier_flag'
    # Flag True where the value is outside [lower, upper] bounds
    # NaN values are not flagged (they can't be compared numerically)
    df_sold[flag_col] = (df_sold[field] < lower) | (df_sold[field] > upper)

    n_flagged = df_sold[flag_col].sum()
    pct       = n_flagged / len(df_sold) * 100
    print(f"\n  {field}:")
    print(f"    Q1={q1:,.2f}  Q3={q3:,.2f}  IQR={iqr:,.2f}")
    print(f"    Lower fence={lower:,.2f}  Upper fence={upper:,.2f}")
    print(f"    Flagged as outlier: {n_flagged:,}  ({pct:.1f}%)")

# Save the FULL dataset with all flags intact — no records removed
df_sold.to_csv("data/sold_flagged.csv", index=False)
print(f"\n[OK] Saved: data/sold_flagged.csv  ({len(df_sold):,} rows × {df_sold.shape[1]} columns)")

# Build sold_final.csv: remove rows where ANY outlier flag is True
# axis=1 means we check across columns (row-wise). any() = True if at least one flag is True.
outlier_flag_cols = [f'{f}_outlier_flag' for f in outlier_fields if f'{f}_outlier_flag' in df_sold.columns]
if outlier_flag_cols:
    is_outlier  = df_sold[outlier_flag_cols].any(axis=1)
    df_final    = df_sold[~is_outlier].copy()   # ~ is the NOT operator
else:
    df_final    = df_sold.copy()

rows_removed_outlier = len(df_sold) - len(df_final)
print(f"\nOutlier removal summary:")
print(f"  Rows before removal : {len(df_sold):,}")
print(f"  Rows removed        : {rows_removed_outlier:,}  (any outlier flag = True)")
print(f"  Rows remaining      : {len(df_final):,}")

# Compare medians before and after to verify the outliers were skewing things
print(f"\nMedian comparison (flagged full set vs final clean set):")
for field in outlier_fields:
    if field in df_sold.columns:
        med_before = df_sold[field].median()
        med_after  = df_final[field].median()
        shift = med_after - med_before
        print(f"  {field:20s}: {med_before:>12,.2f} -> {med_after:>12,.2f}  (shift: {shift:+,.2f})")

df_final.to_csv("data/sold_final.csv", index=False)
print(f"\n[OK] Saved: data/sold_final.csv  ({len(df_final):,} rows × {df_final.shape[1]} columns)")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY — SOLD PIPELINE COMPLETE")
print("=" * 70)

print(f"\nsold_final.csv (clean analytical dataset):")
print(f"  Total rows    : {len(df_final):,}")
print(f"  Total columns : {df_final.shape[1]}")

if 'CloseDate' in df_final.columns:
    close_dates = df_final['CloseDate'].dropna()
    if not close_dates.empty:
        print(f"  Date range    : {close_dates.min().date()} -> {close_dates.max().date()}")

print(f"\nKey stats (outliers removed):")
for field in ['ClosePrice', 'price_per_sqft', 'DaysOnMarket', 'mortgage_rate_30yr']:
    if field in df_final.columns:
        col = df_final[field].dropna()
        if not col.empty:
            print(f"  {field:30s}: median={col.median():>10,.2f}  mean={col.mean():>10,.2f}  "
                  f"min={col.min():>10,.2f}  max={col.max():>10,.2f}")

print("\nOutput files generated:")
print("  data/sold_combined.csv          — Residential records from all monthly files")
print("  data/sold_enriched.csv          — + 30-yr mortgage rate merged by close month")
print("  data/sold_cleaned.csv           — + cleaned, typed, date/geo quality flags")
print("  data/sold_flagged.csv           — + IQR outlier flags (all rows preserved)")
print("  data/sold_final.csv             — outliers removed (primary analytical dataset)")
print("  output/reports/sold_missing_value_report.csv")
print("  output/reports/sold_numeric_distribution.csv")
print("  output/reports/sold_county_summary.csv")
print("=" * 70)
