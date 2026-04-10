import pandas as pd
import glob
import os
import matplotlib
matplotlib.use('Agg')  # Save plots to file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SETUP: Create output folders for plots and reports.
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)


# =============================================================================
# SECTION 1: LOAD ALL LISTING CSV FILES AND COMBINE THEM
# =============================================================================
# Same approach as sold_analysis: glob finds all monthly listing files,
# we read each one and concat them into a single dataframe.
#
# KEY DIFFERENCE FROM SOLD DATA: The listing API fetch script had duplicate
# column names (e.g. PropertyType, DaysOnMarket, LivingArea appeared twice
# in the $select parameter). When pandas reads a CSV with duplicate headers,
# it automatically renames the second occurrence by appending ".1"
# (e.g. PropertyType, PropertyType.1). We drop those ".1" columns right
# after loading because they're just redundant copies of the originals.
# =============================================================================

listing_files = glob.glob("data/CRMLSListing*.csv")
dfs = [pd.read_csv(f, low_memory=False, encoding='latin-1') for f in listing_files]
df_listed = pd.concat(dfs, ignore_index=True)

# Drop the duplicate ".1" columns created by the API extraction script
# A column name ending in ".1" means pandas detected a duplicate header.
# [col for col in ... if col.endswith('.1')] is a list comprehension —
# it builds a list by looping through all column names and keeping only
# the ones that end with ".1". We then drop that list of columns.
dupe_cols = [col for col in df_listed.columns if col.endswith('.1')]
df_listed = df_listed.drop(columns=dupe_cols)


# =============================================================================
# SECTION 2: DATASET UNDERSTANDING
# =============================================================================

rows, cols = df_listed.shape
print("=" * 60)
print("SECTION 1: DATASET UNDERSTANDING")
print("=" * 60)
print(f"Total rows    : {rows:,}")
print(f"Total columns : {cols}")
print(f"Duplicate columns dropped: {dupe_cols}")
print()

print("--- Column Data Types ---")
print(df_listed.dtypes.to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# Metadata vs analysis columns for listing data.
# Listing data has the same agent/office/school metadata as sold data.
# The analysis columns overlap heavily — ListPrice is the main price field
# here since most active listings don't have a ClosePrice yet.
# ─────────────────────────────────────────────────────────────────────────────
metadata_cols = [
    'ListingKey', 'ListingKeyNumeric', 'ListingId',
    'ListAgentEmail', 'ListAgentFirstName', 'ListAgentLastName', 'ListAgentFullName',
    'CoListAgentFirstName', 'CoListAgentLastName',
    'BuyerAgentMlsId', 'BuyerAgentFirstName', 'BuyerAgentLastName',
    'CoBuyerAgentFirstName',
    'ListOfficeName', 'BuyerOfficeName', 'CoListOfficeName',
    'BuyerOfficeAOR',
    'ElementarySchool', 'ElementarySchoolDistrict',
    'MiddleOrJuniorSchool', 'MiddleOrJuniorSchoolDistrict',
    'HighSchool', 'HighSchoolDistrict',
    'SubdivisionName', 'BuilderName', 'MLSAreaMajor',
    'StreetNumberNumeric', 'UnparsedAddress',
    'TaxYear', 'TaxAnnualAmount',
    'BusinessType', 'LotSizeDimensions',
]

analysis_cols = [
    'PropertyType', 'PropertySubType', 'MlsStatus',
    'ListPrice', 'OriginalListPrice', 'ClosePrice',
    'ListingContractDate', 'CloseDate', 'PurchaseContractDate', 'ContractStatusChangeDate',
    'DaysOnMarket',
    'LivingArea', 'AboveGradeFinishedArea', 'BelowGradeFinishedArea', 'BuildingAreaTotal',
    'LotSizeAcres', 'LotSizeArea', 'LotSizeSquareFeet',
    'BedroomsTotal', 'BathroomsTotalInteger',
    'YearBuilt', 'Stories', 'Levels',
    'CountyOrParish', 'City', 'PostalCode', 'StateOrProvince',
    'Latitude', 'Longitude',
    'GarageSpaces', 'CoveredSpaces', 'ParkingTotal',
    'FireplacesTotal', 'FireplaceYN',
    'AssociationFee', 'AssociationFeeFrequency',
    'NewConstructionYN', 'AttachedGarageYN',
    'ViewYN', 'WaterfrontYN', 'BasementYN', 'PoolPrivateYN',
    'Flooring', 'MainLevelBedrooms',
]

# Only show analysis_cols that actually exist after dropping dupes
analysis_cols = [c for c in analysis_cols if c in df_listed.columns]
metadata_cols  = [c for c in metadata_cols  if c in df_listed.columns]

print(f"Metadata columns  ({len(metadata_cols)}): {metadata_cols}")
print()
print(f"Analysis columns  ({len(analysis_cols)}): {analysis_cols}")
print()

# Document all unique property types found
print("--- Unique Property Types ---")
print(df_listed['PropertyType'].value_counts(dropna=False).to_string())
print()

# MlsStatus breakdown — critical for listings: Active means currently for sale,
# Pending/ActiveUnderContract means under contract but not yet closed,
# Closed means sold (these overlap with the sold dataset)
print("--- MLS Status Breakdown ---")
print(df_listed['MlsStatus'].value_counts(dropna=False).to_string())
print()


# =============================================================================
# SECTION 3: MISSING VALUE ANALYSIS
# =============================================================================

print("=" * 60)
print("SECTION 2: MISSING VALUE ANALYSIS")
print("=" * 60)

missing_count = df_listed.isnull().sum()
missing_pct   = (missing_count / len(df_listed) * 100).round(2)

missing_summary = pd.DataFrame({
    'missing_count' : missing_count,
    'missing_pct'   : missing_pct,
}).sort_values('missing_pct', ascending=False)

print("--- Null Count Summary (all columns) ---")
print(missing_summary.to_string())
print()

high_missing = missing_summary[missing_summary['missing_pct'] > 90]
print("--- Columns with >90% Missing Values (candidates to drop) ---")
if high_missing.empty:
    print("None — all columns have at least 10% data present.")
else:
    print(high_missing.to_string())
print()

missing_summary.to_csv("output/reports/listed_missing_value_report.csv")
print("Missing value report saved to output/reports/listed_missing_value_report.csv")
print()


# =============================================================================
# SECTION 4: FILTERING TO RESIDENTIAL ACTIVE/PENDING LISTINGS
# =============================================================================
# For listing analysis, we focus on Residential property type — same reason
# as sold data (commercial and land have completely different price scales).
#
# We keep Active, Pending, and ActiveUnderContract statuses.
# WHY: these represent the current market inventory — homes that were or are
# available to buyers. Closed listings in this dataset are redundant with
# the sold data we already analyzed.
#
# We also require ListPrice > 0 (a listing without a price is invalid).
# =============================================================================

print("=" * 60)
print("SECTION 3: FILTERING")
print("=" * 60)

active_statuses = ['Active', 'Pending', 'ActiveUnderContract', 'ComingSoon']

df_res = df_listed[
    (df_listed['PropertyType'] == 'Residential') &
    (df_listed['MlsStatus'].isin(active_statuses)) &
    (df_listed['ListPrice'].notna()) &
    (df_listed['ListPrice'] > 0)
].copy()

print(f"Rows after filtering (Residential + active statuses + valid ListPrice): {len(df_res):,}")
print()

residential_share = (
    df_listed[df_listed['PropertyType'] == 'Residential'].shape[0]
    / len(df_listed) * 100
)
non_residential = len(df_listed) - df_listed[df_listed['PropertyType'] == 'Residential'].shape[0]
print(f"Q1 — Residential share of all listing records: {residential_share:.1f}%")
print(f"     Non-residential excluded: {non_residential:,} records")
print()

# Status breakdown within residential listings
print("--- Residential listing status breakdown ---")
print(df_listed[df_listed['PropertyType'] == 'Residential']['MlsStatus'].value_counts().to_string())
print()


# =============================================================================
# SECTION 5: NUMERIC DISTRIBUTION REVIEW
# =============================================================================
# Same approach as sold_analysis: percentile summary + histogram + boxplot.
# For listings we swap ClosePrice → ListPrice and OriginalListPrice since
# most active listings don't have a close price.
# =============================================================================

print("=" * 60)
print("SECTION 4: NUMERIC DISTRIBUTION REVIEW")
print("=" * 60)

numeric_fields = [
    'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt',
]
# Only include fields that exist in the dataframe
numeric_fields = [f for f in numeric_fields if f in df_res.columns]

distribution_rows = []

for field in numeric_fields:
    col = df_res[field].dropna()
    if col.empty:
        continue

    row = {
        'field'   : field,
        'count'   : int(col.count()),
        'min'     : col.min(),
        'p25'     : np.percentile(col, 25),
        'median'  : np.percentile(col, 50),
        'mean'    : col.mean(),
        'p75'     : np.percentile(col, 75),
        'p90'     : np.percentile(col, 90),
        'p99'     : np.percentile(col, 99),
        'max'     : col.max(),
    }
    distribution_rows.append(row)

    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(col, bins=50, edgecolor='white', color='darkorange')
    plt.axvline(col.median(), color='red', linestyle='--', label=f'Median: {col.median():,.0f}')
    plt.title(f'Distribution of {field} (Residential Active/Pending)')
    plt.xlabel(field)
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'output/plots/listed_hist_{field}.png', dpi=100)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(col, vert=False, patch_artist=True,
                boxprops=dict(facecolor='moccasin', color='darkorange'),
                medianprops=dict(color='red', linewidth=2))
    plt.title(f'Boxplot of {field} (Residential Active/Pending)')
    plt.xlabel(field)
    plt.tight_layout()
    plt.savefig(f'output/plots/listed_box_{field}.png', dpi=100)
    plt.close()

dist_df = pd.DataFrame(distribution_rows).set_index('field')
print("--- Percentile Summary ---")
print(dist_df.to_string())
print()

dist_df.to_csv("output/reports/listed_numeric_distribution_summary.csv")
print("Numeric distribution summary saved to output/reports/listed_numeric_distribution_summary.csv")
print()

for field in ['ListPrice', 'LivingArea', 'DaysOnMarket']:
    if field in dist_df.index:
        r = dist_df.loc[field]
        print(f"  {field}:")
        print(f"    Min={r['min']:,.0f}  Median={r['median']:,.0f}  Mean={r['mean']:,.0f}  Max={r['max']:,.0f}")
        print()


# =============================================================================
# SECTION 6: BUSINESS QUESTIONS
# =============================================================================

print("=" * 60)
print("SECTION 5: BUSINESS QUESTIONS")
print("=" * 60)

# ── Q2: Median and average list prices ─────────────────────────────────────
median_price = df_res['ListPrice'].median()
mean_price   = df_res['ListPrice'].mean()
print(f"Q2 — List Price (Active/Pending Residential):")
print(f"    Median : ${median_price:,.0f}")
print(f"    Average: ${mean_price:,.0f}")
print()

# ── Q3: Days on Market distribution ────────────────────────────────────────
dom = df_res['DaysOnMarket'].dropna()
print(f"Q3 — Days on Market (Active/Pending Residential):")
print(f"    Median  : {dom.median():.0f} days")
print(f"    Average : {dom.mean():.1f} days")
print(f"    90th pct: {np.percentile(dom, 90):.0f} days")
print(f"    Max     : {dom.max():.0f} days")
print()

# ── Q4: Price reduction analysis ───────────────────────────────────────────
# For listings, the "above vs below list" question from sold data becomes:
# "How many listings have been price-reduced since original listing?"
#
# A price reduction means the seller lowered their ask — usually because
# the home wasn't getting offers at the original price.
#
# price_reduction_pct = (OriginalListPrice - ListPrice) / OriginalListPrice * 100
# Positive = reduced. Negative = raised (rare but happens).
# Zero = no change from original.
#
# We only calculate this where both prices exist and OriginalListPrice > 0.
price_valid = df_res[
    df_res['OriginalListPrice'].notna() &
    df_res['ListPrice'].notna() &
    (df_res['OriginalListPrice'] > 0)
].copy()

price_valid['price_reduction_pct'] = (
    (price_valid['OriginalListPrice'] - price_valid['ListPrice'])
    / price_valid['OriginalListPrice'] * 100
).round(2)

reduced      = (price_valid['price_reduction_pct'] > 0).sum()
not_reduced  = (price_valid['price_reduction_pct'] <= 0).sum()
total_priced = len(price_valid)

print(f"Q4 — Price Reductions ({total_priced:,} listings with valid prices):")
print(f"    Price reduced since original listing : {reduced:,}  ({reduced/total_priced*100:.1f}%)")
print(f"    No reduction (at or above original)  : {not_reduced:,}  ({not_reduced/total_priced*100:.1f}%)")
if reduced > 0:
    avg_reduction = price_valid[price_valid['price_reduction_pct'] > 0]['price_reduction_pct'].mean()
    print(f"    Average reduction among reduced homes : {avg_reduction:.1f}%")
print()

# ── Q5: Date consistency check ─────────────────────────────────────────────
# For listings: is ListingContractDate ever in the future relative to CloseDate?
# We check if any listings have a contract date after their reported close date.
df_res['ListDate_dt']  = pd.to_datetime(df_res['ListingContractDate'], errors='coerce')
df_res['CloseDate_dt'] = pd.to_datetime(df_res['CloseDate'],           errors='coerce')

# Check for listings where CloseDate is before ListingContractDate
date_valid = df_res[df_res['ListDate_dt'].notna() & df_res['CloseDate_dt'].notna()]
bad_dates  = date_valid[date_valid['CloseDate_dt'] < date_valid['ListDate_dt']]

# Also check for future listing dates (listed after today — data error)
today = pd.Timestamp('2026-04-10')
future_listed = df_res[df_res['ListDate_dt'].notna() & (df_res['ListDate_dt'] > today)]

print(f"Q5 — Date Consistency:")
print(f"    CloseDate before ListingContractDate : {len(bad_dates):,} records")
print(f"    ListingContractDate in the future    : {len(future_listed):,} records")
if not bad_dates.empty:
    print("    Sample bad rows (close before list):")
    print(bad_dates[['UnparsedAddress', 'ListingContractDate', 'CloseDate']].head(5).to_string())
print()

# ── Q6: Counties with highest median list prices ────────────────────────────
county_median = (
    df_res.groupby('CountyOrParish')['ListPrice']
    .agg(median_list_price='median', listing_count='count')
    .sort_values('median_list_price', ascending=False)
)
print("Q6 — Top 10 Counties by Median List Price (Active/Pending Residential):")
print(county_median.head(10).to_string())
print()


# =============================================================================
# SECTION 7: DELIVERABLE — SAVE FILTERED DATASET
# =============================================================================

print("=" * 60)
print("SECTION 6: SAVING FILTERED DATASET")
print("=" * 60)

df_res_save = df_res.drop(columns=['ListDate_dt', 'CloseDate_dt'], errors='ignore')
df_res_save.to_csv("data/listed_residential_clean.csv", index=False)

print(f"Filtered dataset saved to data/listed_residential_clean.csv")
print(f"Shape: {df_res_save.shape[0]:,} rows x {df_res_save.shape[1]} columns")
print()

print("=" * 60)
print("DONE. Output files:")
print("  data/listed_residential_clean.csv               — filtered dataset")
print("  output/reports/listed_missing_value_report.csv")
print("  output/reports/listed_numeric_distribution_summary.csv")
print("  output/plots/listed_hist_*.png                  — histograms")
print("  output/plots/listed_box_*.png                   — boxplots")
print("=" * 60)
