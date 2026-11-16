"""
Master Autonomous Execution Pipeline
Retail Demand Forecasting and Inventory Risk Prediction
Executes all 16 project lifecycle stages sequentially with complete error handling and validation logging.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.cleaner import clean_retail_data
from src.eda.eda_analysis import run_eda
from src.statistical.abc_xyz import compute_abc_xyz
from src.statistical.intermittency import classify_demand_patterns
from src.feature_engineering.features import build_daily_features
from src.forecasting.baselines import run_baselines
from src.forecasting.ml_models import train_and_evaluate_forecasting_models
from src.forecasting.predict import generate_future_forecast
from src.inventory_risk.risk_simulation import generate_inventory_risk_data
from src.inventory_risk.classifiers import evaluate_classifier_suite
from src.explainability.explainer import run_explainability_analysis


def run_full_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("STARTING FULL END-TO-END RETAIL AI/ML PIPELINE EXECUTION")
    print("=" * 80)

    stages = [
        ("STAGE 3: Data Cleaning & Preprocessing", lambda: clean_retail_data()),
        ("STAGE 5: Exploratory Data Analysis (EDA)", lambda: run_eda()),
        ("STAGE 6A: ABC-XYZ Inventory Segmentation", lambda: compute_abc_xyz()),
        ("STAGE 6B: Demand Intermittency Classification", lambda: classify_demand_patterns()),
        ("STAGE 4: Advanced Feature Engineering", lambda: build_daily_features()),
        ("STAGE 7: Baseline Demand Forecasting", lambda: run_baselines()),
        ("STAGE 8 & 9: ML Forecasting & Evaluation", lambda: train_and_evaluate_forecasting_models()),
        ("STAGE 10: 14-Day Multi-Step Prediction", lambda: generate_future_forecast(14)),
        ("STAGE 11A: Inventory Risk Simulation", lambda: generate_inventory_risk_data()),
        ("STAGE 11B: Risk Classification & SMOTE", lambda: evaluate_classifier_suite()),
        ("STAGE 12: Model Explainability (TreeSHAP)", lambda: run_explainability_analysis()),
    ]

    for stage_name, stage_func in stages:
        t0 = time.time()
        print(f"\n>>> Executing {stage_name}...")
        try:
            stage_func()
            print(f">>> [SUCCESS] {stage_name} finished in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f">>> [ERROR] in {stage_name}: {str(e)}")
            raise e

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE: ALL STAGES EXECUTED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
