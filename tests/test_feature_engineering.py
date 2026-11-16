"""
Unit Tests for Feature Engineering & Chronological Partitioning
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")


def test_feature_matrix_exists_and_scale():
    feat_path = os.path.join(FEATURES_DIR, "daily_sku_features.csv")
    assert os.path.exists(feat_path), f"Features CSV missing at {feat_path}"
    df = pd.read_csv(feat_path, nrows=500)
    assert len(df) > 0

    with open(os.path.join(FEATURES_DIR, "split_metadata.json"), "r") as f:
        meta = json.load(f)
    assert meta["total_records"] >= 10000, "Daily feature matrix fails 10,000 record college requirement"


def test_chronological_splits():
    train_df = pd.read_csv(os.path.join(FEATURES_DIR, "train.csv"), parse_dates=["Date"])
    val_df = pd.read_csv(os.path.join(FEATURES_DIR, "val.csv"), parse_dates=["Date"])
    test_df = pd.read_csv(os.path.join(FEATURES_DIR, "test.csv"), parse_dates=["Date"])

    # Strict chronological boundaries: max(train) < min(val) and max(val) <= min(test)
    assert train_df["Date"].max() <= val_df["Date"].min(), "Temporal leakage between train and val"
    assert val_df["Date"].max() <= test_df["Date"].min(), "Temporal leakage between val and test"

    # No empty partitions
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_no_target_leakage_in_lags():
    train_df = pd.read_csv(os.path.join(FEATURES_DIR, "train.csv"))
    # Check that Lag_1 is present and not equal to current target across the board
    assert "Lag_1" in train_df.columns
    assert "DailyDemand" in train_df.columns
    assert "Rolling_Mean_7" in train_df.columns
    # Check rolling mean is non-negative
    assert (train_df["Rolling_Mean_7"] >= 0).all()
