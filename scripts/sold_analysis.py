import pandas as pd
import glob
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend so plots save to file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SETUP: Create an output folder for plots and reports.
# os.makedirs(..., exist_ok=True) means: create the folder if it doesn't already
# exist. If it does exist, don't throw an error — just move on.
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)


# =============================================================================
# SECTION 1: LOAD ALL SOLD CSV FILES AND COMBINE THEM
# =============================================================================
# glob.glob finds every file matching the pattern — here, any CSV starting with
# "CRMLSSold" in the data folder. This way we don't have to list every month manually.
#
# pd.read_csv(..., low_memory=False) reads the file. low_memory=False tells
# pandas not to guess column types row-by-row (which causes mixed-type warnings)
# and instead scan the whole column at once.
#
# encoding='latin-1' handles special characters (accented letters, etc.) that
# plain UTF-8 can't read. MLS data often has these in address or agent fields.
#
# pd.concat(..., ignore_index=True) stacks all the monthly dataframes into one.
# ignore_index=True resets the row numbers so they run 0, 1, 2, ... continuously
# instead of repeating 0-999 for each file.
# =============================================================================

sold_files = glob.glob("data/CRMLSSold*.csv")
dfs = [pd.read_csv(f, low_memory=False, encoding='latin-1') for f in sold_files]
df_sold = pd.concat(dfs, ignore_index=True)


# =============================================================================
# SECTION 2: DATASET UNDERSTANDING
# =============================================================================
# .shape returns (rows, columns) as a tuple. We unpack it into two variables.
# f-strings (f"...") let us embed variables directly inside a string with {}.
# =============================================================================

rows, cols = df_sold.shape
print("=" * 60)
print("SECTION 1: DATASET UNDERSTANDING")
print("=" * 60)
print(f"Total rows    : {rows:,}")   # The :, formats large numbers with commas (e.g. 50,000)
print(f"Total columns : {cols}")
print()

# .dtypes gives the data type pandas assigned to each column.
# 'object' means text (or mixed types). float64/int64 are numbers.
# Dates almost always come in as 'object' from CSVs — we'll fix that later.
print("--- Column Data Types ---")
print(df_sold.dtypes.to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# Separate columns into two categories:
#   metadata_cols  = identifiers, agent/office info, school names — not used
#                    in market analysis, but useful for linking records
#   analysis_cols  = the fields we actually analyze: prices, size, dates,
#                    location, property type, condition, etc.
# ─────────────────────────────────────────────────────────────────────────────

metadata_cols = [
    'ListingKey', 'ListingKeyNumeric', 'ListingId', 'MlsStatus',
    'ListAgentEmail', 'ListAgentFirstName', 'ListAgentLastName', 'ListAgentFullName',
    'CoListAgentFirstName', 'CoListAgentLastName',
    'BuyerAgentMlsId', 'BuyerAgentFirstName', 'BuyerAgentLastName',
    'CoBuyerAgentFirstName',
    'ListOfficeName', 'BuyerOfficeName', 'CoListOfficeName',
    'BuyerAgentAOR', 'ListAgentAOR', 'BuyerOfficeAOR',
    'ElementarySchool', 'ElementarySchoolDistrict',
    'MiddleOrJuniorSchool', 'MiddleOrJuniorSchoolDistrict',
    'HighSchool', 'HighSchoolDistrict',
    'OriginatingSystemName', 'OriginatingSystemSubName',
    'SubdivisionName', 'BuilderName', 'MLSAreaMajor',
    'StreetNumberNumeric', 'UnparsedAddress',
    'TaxYear', 'TaxAnnualAmount',
    'BusinessType', 'LotSizeDimensions',
]

analysis_cols = [
    'PropertyType', 'PropertySubType',
    'CloseDate', 'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'ListingContractDate', 'PurchaseContractDate', 'ContractStatusChangeDate',
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

print(f"Metadata columns  ({len(metadata_cols)}): {metadata_cols}")
print()
print(f"Analysis columns  ({len(analysis_cols)}): {analysis_cols}")
print()

# Document every unique PropertyType value in the dataset.
# .value_counts() counts how many times each value appears, sorted highest first.
# This is the first thing you want to know before filtering.
print("--- Unique Property Types ---")
print(df_sold['PropertyType'].value_counts(dropna=False).to_string())
print()


# =============================================================================
# SECTION 3: MISSING VALUE ANALYSIS
# =============================================================================
# .isnull() returns a True/False table — True wherever a value is missing (NaN).
# .sum() counts the Trues (i.e. the number of missing values) per column.
# Dividing by len(df_sold) and multiplying by 100 converts to a percentage.
# We round to 2 decimal places for readability.
# =============================================================================

print("=" * 60)
print("SECTION 2: MISSING VALUE ANALYSIS")
print("=" * 60)

missing_count = df_sold.isnull().sum()
missing_pct   = (missing_count / len(df_sold) * 100).round(2)

# Build a summary DataFrame so we can display it as a clean table.
# pd.DataFrame({...}) creates a table from a dictionary of column_name: series.
# We sort by missing percentage descending so the worst columns are on top.
missing_summary = pd.DataFrame({
    'missing_count'  : missing_count,
    'missing_pct'    : missing_pct,
}).sort_values('missing_pct', ascending=False)

print("--- Null Count Summary (all columns) ---")
print(missing_summary.to_string())
print()

# Flag columns where more than 90% of rows are missing.
# These columns are so empty they're almost useless for analysis.
# The decision: drop them from the analytical dataset, but keep them
# in the raw data in case they matter for edge cases.
high_missing = missing_summary[missing_summary['missing_pct'] > 90]
print("--- Columns with >90% Missing Values (candidates to drop) ---")
if high_missing.empty:
    print("None — all columns have 90% or more data present.")
else:
    print(high_missing.to_string())
print()

# Save the missing value report to a CSV file for your deliverable.
missing_summary.to_csv("output/reports/missing_value_report.csv")
print("Missing value report saved to output/reports/missing_value_report.csv")
print()


# =============================================================================
# SECTION 4: FILTERING TO RESIDENTIAL PROPERTIES
# =============================================================================
# We focus on Residential because that's the core housing market.
# Commercial, CommercialLease, Land etc. have very different price dynamics
# and would distort our distributions and averages.
#
# We also drop rows where ClosePrice is null or zero — you can't analyze a
# sale without knowing what it sold for.
#
# The ~ symbol means NOT. ~df['col'].isin([...]) means "not in this list".
# =============================================================================

print("=" * 60)
print("SECTION 3: FILTERING")
print("=" * 60)

# Show all property types so we can confirm what we're keeping vs dropping
print("Property type distribution before filter:")
print(df_sold['PropertyType'].value_counts(dropna=False).to_string())
print()

# Filter to Residential only
df_res = df_sold[df_sold['PropertyType'] == 'Residential'].copy()

# Drop rows with missing or zero ClosePrice — can't analyze a sale without a price
df_res = df_res[df_res['ClosePrice'].notna() & (df_res['ClosePrice'] > 0)]

print(f"Rows after filtering to Residential + valid ClosePrice: {len(df_res):,}")
print()

# Calculate and print the residential share as a percentage of total records
residential_share = len(df_res) / len(df_sold) * 100
non_residential   = len(df_sold) - len(df_res)
print(f"Residential share : {residential_share:.1f}% of all records")
print(f"Non-residential   : {non_residential:,} records excluded")
print()


# =============================================================================
# SECTION 5: NUMERIC DISTRIBUTION REVIEW
# =============================================================================
# For each key numeric field we'll do three things:
#   1. Percentile summary — min, 25th, median, 75th, 90th, 99th, max
#   2. Histogram — shows the shape of the distribution (skewed? normal? bimodal?)
#   3. Boxplot — shows median, interquartile range, and flags statistical outliers
#      as dots beyond the whiskers (default: 1.5x the IQR from the box edges)
# =============================================================================

print("=" * 60)
print("SECTION 4: NUMERIC DISTRIBUTION REVIEW")
print("=" * 60)

numeric_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt',
]

# np.percentile(series.dropna(), q) computes the value at the qth percentile.
# dropna() skips missing values — np.percentile can't handle NaN.
# We build the summary row by row and collect them into a list of dicts,
# then convert to a DataFrame at the end.

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

    # ── Histogram ──────────────────────────────────────────────────────────
    # plt.figure() starts a new blank figure.
    # plt.hist() draws the histogram. bins=50 splits the range into 50 bars.
    # edgecolor='white' puts a thin white border between bars so they're easier to read.
    # axvline draws a vertical reference line at the median (dashed red).
    # plt.tight_layout() prevents axis labels from getting cut off.
    # plt.savefig() saves the image; plt.close() frees memory.
    # ───────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.hist(col, bins=50, edgecolor='white', color='steelblue')
    plt.axvline(col.median(), color='red', linestyle='--', label=f'Median: {col.median():,.0f}')
    plt.title(f'Distribution of {field}')
    plt.xlabel(field)
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'output/plots/hist_{field}.png', dpi=100)
    plt.close()

    # ── Boxplot ────────────────────────────────────────────────────────────
    # A boxplot shows: the box = 25th to 75th percentile (the "middle 50%")
    # The line inside the box = median. The whiskers extend to 1.5x the IQR.
    # Dots beyond the whiskers are statistical outliers — not necessarily errors,
    # but worth investigating (e.g. a $50M mansion is a real sale but skews stats).
    # ───────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(6, 4))
    plt.boxplot(col, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=2))
    plt.title(f'Boxplot of {field}')
    plt.xlabel(field)
    plt.tight_layout()
    plt.savefig(f'output/plots/box_{field}.png', dpi=100)
    plt.close()

# Print the summary table for ClosePrice, LivingArea, DaysOnMarket
dist_df = pd.DataFrame(distribution_rows).set_index('field')
print("--- Percentile Summary ---")
print(dist_df.to_string())
print()

# Save the full distribution summary for the deliverable
dist_df.to_csv("output/reports/numeric_distribution_summary.csv")
print("Numeric distribution summary saved to output/reports/numeric_distribution_summary.csv")
print()

# Highlight just the three required fields clearly
for field in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
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

# ── Q1: Residential vs other property type share ───────────────────────────
# Already computed above in the filtering section. Reprinting for clarity.
print(f"Q1 — Residential share of all records: {residential_share:.1f}%")
print()

# ── Q2: Median and average close prices ────────────────────────────────────
# .median() returns the middle value — half of sales are above, half below.
# .mean() returns the average — sensitive to outliers (a $10M sale pulls this up).
# In real estate, median is almost always the more useful public stat.
median_price = df_res['ClosePrice'].median()
mean_price   = df_res['ClosePrice'].mean()
print(f"Q2 — Close Price:")
print(f"    Median : ${median_price:,.0f}")
print(f"    Average: ${mean_price:,.0f}")
print()

# ── Q3: Days on Market distribution ────────────────────────────────────────
# We print a few key percentiles to describe the shape:
# p50 = half of homes sold within this many days
# p90 = 90% of homes sold within this many days (flags slow movers above this)
dom = df_res['DaysOnMarket'].dropna()
print(f"Q3 — Days on Market:")
print(f"    Median  : {dom.median():.0f} days")
print(f"    Average : {dom.mean():.1f} days")
print(f"    90th pct: {np.percentile(dom, 90):.0f} days  (10% of homes took longer than this)")
print(f"    Max     : {dom.max():.0f} days")
print()

# ── Q4: Above vs below list price ──────────────────────────────────────────
# We compute the ratio of ClosePrice to ListPrice for each row.
# ratio > 1.0  → sold ABOVE list price (competitive market / bidding war)
# ratio < 1.0  → sold BELOW list price (negotiation, price reduction, slow market)
# ratio == 1.0 → sold exactly at list price
#
# .notna() makes sure we only count rows where both prices are present.
# & is the element-wise AND operator for pandas boolean series.
valid = df_res[df_res['ClosePrice'].notna() & df_res['ListPrice'].notna() & (df_res['ListPrice'] > 0)].copy()
valid['price_ratio'] = valid['ClosePrice'] / valid['ListPrice']

above_list = (valid['price_ratio'] > 1.0).sum()
below_list = (valid['price_ratio'] < 1.0).sum()
at_list    = (valid['price_ratio'] == 1.0).sum()
total_valid = len(valid)

print(f"Q4 — Sold vs List Price ({total_valid:,} valid records):")
print(f"    Above list price : {above_list:,}  ({above_list/total_valid*100:.1f}%)")
print(f"    At list price    : {at_list:,}  ({at_list/total_valid*100:.1f}%)")
print(f"    Below list price : {below_list:,}  ({below_list/total_valid*100:.1f}%)")
print()

# ── Q5: Date consistency check ─────────────────────────────────────────────
# pd.to_datetime() converts a column of date strings into actual datetime objects.
# errors='coerce' means: if a value can't be parsed as a date, make it NaT
# (Not a Time — pandas' equivalent of NaN for dates) instead of crashing.
#
# We then check: does CloseDate ever come BEFORE ListingContractDate?
# That would be physically impossible and indicates a data entry error.
df_res['CloseDate_dt']   = pd.to_datetime(df_res['CloseDate'],           errors='coerce')
df_res['ListDate_dt']    = pd.to_datetime(df_res['ListingContractDate'], errors='coerce')

# Only compare rows where both dates are present
date_valid = df_res[df_res['CloseDate_dt'].notna() & df_res['ListDate_dt'].notna()]
bad_dates  = date_valid[date_valid['CloseDate_dt'] < date_valid['ListDate_dt']]

print(f"Q5 — Date Consistency (CloseDate before ListingContractDate):")
print(f"    Rows with valid dates : {len(date_valid):,}")
print(f"    Bad date rows         : {len(bad_dates):,}  ({len(bad_dates)/len(date_valid)*100:.2f}% of dated records)")
if not bad_dates.empty:
    print("    Sample bad rows:")
    print(bad_dates[['UnparsedAddress', 'ListingContractDate', 'CloseDate']].head(5).to_string())
print()

# ── Q6: Counties with highest median close prices ──────────────────────────
# .groupby('CountyOrParish') splits the data into groups — one per county.
# ['ClosePrice'].median() computes the median price within each group.
# .sort_values(ascending=False) puts the most expensive counties first.
# .head(10) shows only the top 10.
county_median = (
    df_res.groupby('CountyOrParish')['ClosePrice']
    .agg(median_price='median', sales_count='count')
    .sort_values('median_price', ascending=False)
)

print("Q6 — Top 10 Counties by Median Close Price:")
print(county_median.head(10).to_string())
print()


# =============================================================================
# SECTION 7: DELIVERABLE — SAVE FILTERED DATASET
# =============================================================================
# We drop the helper date columns we created temporarily (they're derived,
# not original fields) and save the clean residential-only dataset.
#
# index=False means: don't write the row numbers (0, 1, 2...) as a column
# in the CSV — that's just clutter.
# =============================================================================

print("=" * 60)
print("SECTION 6: SAVING FILTERED DATASET")
print("=" * 60)

df_res_save = df_res.drop(columns=['CloseDate_dt', 'ListDate_dt'], errors='ignore')
df_res_save.to_csv("data/sold_residential_clean.csv", index=False)

print(f"Filtered dataset saved to data/sold_residential_clean.csv")
print(f"Shape: {df_res_save.shape[0]:,} rows × {df_res_save.shape[1]} columns")
print()

# Final summary printout
print("=" * 60)
print("DONE. Output files:")
print("  data/sold_residential_clean.csv      — filtered dataset")
print("  output/reports/missing_value_report.csv")
print("  output/reports/numeric_distribution_summary.csv")
print("  output/plots/hist_*.png              — histograms")
print("  output/plots/box_*.png               — boxplots")
print("=" * 60)
