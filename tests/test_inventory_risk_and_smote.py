"""
Unit Tests for Inventory Risk Simulation, Multi-Class Classification, and SMOTE Verification
"""

import os
import joblib
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "features", "inventory_risk_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "inventory_risk")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
PREDS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")


def test_inventory_risk_dataset_structure():
    assert os.path.exists(DATA_PATH)
    df = pd.read_csv(DATA_PATH, nrows=1000)
    assert "Simulated_Stock" in df.columns
    assert "Days_Of_Supply" in df.columns
    assert "Inventory_Risk_Class" in df.columns

    # Verify 3 distinct classes (0=Normal, 1=Stockout, 2=Overstock)
    classes = set(df["Inventory_Risk_Class"].unique())
    assert classes.issubset({0, 1, 2}), f"Unexpected classes: {classes}"


def test_smote_comparison_report_integrity():
    smote_csv = os.path.join(REPORTS_DIR, "smote_inventory_risk_comparison.csv")
    assert os.path.exists(smote_csv)
    df = pd.read_csv(smote_csv)

    # Must contain both regimes
    regimes = set(df["Regime"].unique())
    assert regimes == {"Before SMOTE", "After SMOTE"}, f"Unexpected regimes: {regimes}"

    # Verify SMOTE improves Stockout Recall for Random Forest
    rf_before = df[(df["Model"] == "RandomForest") & (df["Regime"] == "Before SMOTE")]["Stockout_Recall"].iloc[0]
    rf_after = df[(df["Model"] == "RandomForest") & (df["Regime"] == "After SMOTE")]["Stockout_Recall"].iloc[0]
    assert rf_after > rf_before, f"SMOTE failed to improve minority Stockout Recall: {rf_before} -> {rf_after}"


def test_inventory_risk_test_predictions():
    pred_path = os.path.join(PREDS_DIR, "inventory_risk_test_predictions.csv")
    assert os.path.exists(pred_path)
    df = pd.read_csv(pred_path)
    assert "Predicted_Risk_Class" in df.columns
    assert "Predicted_Risk_Label" in df.columns
    assert "Prob_Stockout" in df.columns
    assert "Prob_Normal" in df.columns
    assert "Prob_Overstock" in df.columns

    # Probabilities sum to approximately 1.0
    prob_sums = df["Prob_Stockout"] + df["Prob_Normal"] + df["Prob_Overstock"]
    assert np.allclose(prob_sums, 1.0, atol=0.05), "Class probabilities do not sum to 1.0"
