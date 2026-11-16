"""
Unit Tests for Data Integrity and Preprocessing Sanitation
"""

import os
import json
import pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_transactions.csv")
RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "online_retail.csv")
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_summary.json")


def test_raw_dataset_exists_and_scale():
    assert os.path.exists(RAW_CSV), f"Raw CSV missing at {RAW_CSV}"
    raw_df = pd.read_csv(RAW_CSV, nrows=100)
    assert len(raw_df) > 0
    with open(SUMMARY_JSON, "r") as f:
        meta = json.load(f)
    assert meta["raw_record_count"] >= 10000, "Raw dataset fails 10,000 record college requirement"
    assert meta["raw_record_count"] > 500000, "Expected >500k raw records from UCI repository"


def test_cleaned_dataset_integrity():
    assert os.path.exists(PROCESSED_CSV), f"Cleaned CSV missing at {PROCESSED_CSV}"
    df = pd.read_csv(PROCESSED_CSV, nrows=5000)

    # 1. Non-negative quantities and prices
    assert (df["Quantity"] > 0).all(), "Found non-positive quantities in cleaned data"
    assert (df["UnitPrice"] > 0).all(), "Found non-positive prices in cleaned data"

    # 2. No cancellations (Invoices starting with 'C')
    assert not df["InvoiceNo"].astype(str).str.startswith("C").any(), "Found cancellation in cleaned data"

    # 3. Required columns exist
    required_cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "TotalAmount", "Date"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}' in cleaned dataset"


def test_10k_requirement_satisfied():
    with open(SUMMARY_JSON, "r") as f:
        meta = json.load(f)
    assert meta["cleaned_record_count"] >= 10000, "Cleaned dataset fails 10,000 record college requirement"
    assert meta["meets_10k_requirement"] is True
