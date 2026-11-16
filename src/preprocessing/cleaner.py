"""
Data Cleaning and Preprocessing Module
Performs comprehensive data sanitation on the raw Online Retail dataset:
- Removes non-product and adjustment stock codes
- Filters out cancellations (InvoiceNo starting with 'C')
- Handles non-positive quantities and prices
- Imputes missing product descriptions
- Handles outliers gracefully
- Generates structured datetime features
- Produces clean transaction records for downstream modeling.
"""

import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "online_retail.csv")
CLEANED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_transactions.csv")
SUMMARY_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_summary.json")

# Operational / Non-product stock codes to exclude
NON_PRODUCT_CODES = {
    "POST", "D", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK",
    "PADS", "S", "B", "gift_0001_10", "gift_0001_20", "gift_0001_30",
    "gift_0001_40", "gift_0001_50"
}


def clean_retail_data(input_csv: str = RAW_CSV_PATH, output_csv: str = CLEANED_CSV_PATH) -> pd.DataFrame:
    print("=" * 60)
    print("STAGE 3: DATA CLEANING AND PREPROCESSING")
    print("=" * 60)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Raw CSV not found at '{input_csv}'. Run download_data.py first.")

    print(f"[INFO] Reading raw dataset from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    initial_rows = len(df)
    print(f"[INFO] Initial raw record count: {initial_rows:,}")
    print(f"[INFO] Columns present: {list(df.columns)}")

    # Standardize string fields
    for col in ["InvoiceNo", "StockCode", "Description", "Country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Document missing values
    missing_summary = df.isnull().sum().to_dict()
    print(f"[INFO] Missing values per column: {missing_summary}")

    # Step 1: Detect cancellations
    is_cancellation = df["InvoiceNo"].str.upper().str.startswith("C", na=False)
    cancellation_count = int(is_cancellation.sum())
    print(f"[INFO] Detected {cancellation_count:,} cancelled orders ({cancellation_count / initial_rows * 100:.2f}%).")
    df = df[~is_cancellation].copy()

    # Step 2: Filter non-product codes
    clean_stock = df["StockCode"].str.upper()
    is_non_product = clean_stock.isin(NON_PRODUCT_CODES) | clean_stock.str.contains("TEST|SAMPLE|ADJUST", regex=True, na=False)
    non_prod_count = int(is_non_product.sum())
    print(f"[INFO] Filtered {non_prod_count:,} non-product / service adjustment rows.")
    df = df[~is_non_product].copy()

    # Step 3: Numeric validation (Quantity > 0, UnitPrice > 0)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    valid_numeric = (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    invalid_num_count = int((~valid_numeric).sum())
    print(f"[INFO] Filtered {invalid_num_count:,} non-positive quantity or price records.")
    df = df[valid_numeric].copy()

    # Step 4: Outlier capping for extreme single transactions (e.g. 99.9th percentile)
    q_cap = float(df["Quantity"].quantile(0.9995))
    p_cap = float(df["UnitPrice"].quantile(0.9995))
    print(f"[INFO] 99.95th percentile quantity cap: {q_cap:.1f}, unit price cap: {p_cap:.2f}")
    df = df[(df["Quantity"] <= q_cap) & (df["UnitPrice"] <= p_cap)].copy()

    # Step 5: Impute missing descriptions using known StockCode mappings
    stock_desc_map = (
        df[df["Description"].notnull() & (df["Description"] != "nan") & (df["Description"] != "")]
        .groupby("StockCode")["Description"]
        .first()
        .to_dict()
    )
    df["Description"] = df["Description"].replace(["nan", ""], np.nan)
    df["Description"] = df["Description"].fillna(df["StockCode"].map(stock_desc_map)).fillna("Unknown Product")

    # Step 6: Parse InvoiceDate & create temporal features
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[df["InvoiceDate"].notnull()].copy()
    df["Date"] = df["InvoiceDate"].dt.date
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["DayName"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Calculate TotalAmount (Gross Revenue)
    df["TotalAmount"] = (df["Quantity"] * df["UnitPrice"]).round(2)

    # Sort chronologically
    df = df.sort_values("InvoiceDate").reset_index(drop=True)

    final_rows = len(df)
    unique_skus = df["StockCode"].nunique()
    unique_invoices = df["InvoiceNo"].nunique()
    unique_customers = df["CustomerID"].replace(["nan", "None", np.nan], np.nan).dropna().nunique()
    date_min = str(df["Date"].min())
    date_max = str(df["Date"].max())

    print(f"[OK] Cleaned record count: {final_rows:,} (Retention: {final_rows / initial_rows * 100:.2f}%)")
    print(f"[OK] Verified >10,000 requirement: {final_rows >= 10000} ({final_rows:,} records)")
    print(f"[OK] Unique SKUs: {unique_skus:,}")
    print(f"[OK] Date range: {date_min} to {date_max}")

    # Save summary metadata
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Cleaned dataset written to: {output_csv}")

    summary_meta = {
        "dataset_name": "UCI Online Retail Cleaned",
        "raw_record_count": initial_rows,
        "cleaned_record_count": final_rows,
        "meets_10k_requirement": bool(final_rows >= 10000),
        "unique_skus": unique_skus,
        "unique_invoices": unique_invoices,
        "unique_customers": unique_customers,
        "date_range": [date_min, date_max],
        "total_revenue": float(df["TotalAmount"].sum()),
        "total_units_sold": int(df["Quantity"].sum()),
        "average_order_value": float(df.groupby("InvoiceNo")["TotalAmount"].sum().mean()),
        "cancellations_filtered": cancellation_count,
        "non_product_records_filtered": non_prod_count,
        "invalid_numeric_records_filtered": invalid_num_count
    }

    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary_meta, f, indent=4)
    print(f"[OK] Summary metadata written to: {SUMMARY_JSON_PATH}")

    return df


if __name__ == "__main__":
    clean_retail_data()
