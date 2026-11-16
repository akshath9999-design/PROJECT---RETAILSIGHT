"""
Unit Tests for Forecasting Models and Multi-Step Predictions
"""

import os
import json
import joblib
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "forecasting")
PREDS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")


def test_trained_forecaster_artifacts_exist():
    expected_models = [
        "lightgbm_forecaster.joblib",
        "xgboost_forecaster.joblib",
        "catboost_forecaster.joblib",
        "randomforest_forecaster.joblib"
    ]
    for m in expected_models:
        p = os.path.join(MODELS_DIR, m)
        assert os.path.exists(p), f"Model artifact {m} missing at {p}"


def test_forecast_comparison_matrix():
    comp_path = os.path.join(REPORTS_DIR, "forecasting_model_comparison.csv")
    assert os.path.exists(comp_path)
    df = pd.read_csv(comp_path)
    assert len(df) >= 6, "Expected at least 6 models compared (ML + Baselines)"
    assert "Test_MAE" in df.columns
    assert "Test_WAPE" in df.columns

    # Verify best model outperforms Naive benchmark
    naive_mae = df[df["Model"] == "Naive (t-1)"]["Test_MAE"].iloc[0]
    lgbm_mae = df[df["Model"] == "LightGBM"]["Test_MAE"].iloc[0]
    assert lgbm_mae < naive_mae, f"LightGBM MAE ({lgbm_mae}) must be lower than Naive MAE ({naive_mae})"


def test_future_14d_predictions_exist():
    pred_path = os.path.join(PREDS_DIR, "future_demand_forecast_14d.csv")
    assert os.path.exists(pred_path)
    df = pd.read_csv(pred_path)
    assert len(df) > 0
    assert "Forecast_Date" in df.columns
    assert "Predicted_Demand" in df.columns
    assert "Lower_Bound_80" in df.columns
    assert "Upper_Bound_80" in df.columns
    assert (df["Predicted_Demand"] >= 0).all(), "Found negative demand forecasts"
    assert (df["Upper_Bound_80"] >= df["Lower_Bound_80"]).all(), "Upper bound must be >= lower bound"
