"""
Baseline Demand Forecasting Models Module
Implements benchmark baseline forecasting models:
1. Naive (t-1)
2. Seasonal Naive (t-7)
3. Moving Average (7-day)
4. Moving Average (14-day)
5. Croston Method
6. Croston SBA (Syntetos-Boylan Approximation)
Evaluates performance against holdout test set to establish benchmark metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.forecasting.croston import CrostonForecaster
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE = sum(|y - y_hat|) / sum(y))"""
    sum_actual = np.sum(y_true)
    if sum_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_actual)


def run_baselines() -> pd.DataFrame:
    print("=" * 60)
    print("STAGE 7: BASELINE FORECASTING MODELS")
    print("=" * 60)

    train_path = os.path.join(FEATURES_DIR, "train.csv")
    test_path = os.path.join(FEATURES_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_test = test_df["DailyDemand"].values

    results = []

    # 1. Naive Forecast (t-1)
    y_naive = test_df["Lag_1"].values
    results.append({
        "Model": "Naive (t-1)",
        "Type": "Baseline",
        "MAE": mean_absolute_error(y_test, y_naive),
        "RMSE": root_mean_squared_error(y_test, y_naive),
        "WAPE": calculate_wape(y_test, y_naive),
        "R2": r2_score(y_test, y_naive)
    })

    # 2. Seasonal Naive (t-7)
    y_snaive = test_df["Lag_7"].values
    results.append({
        "Model": "Seasonal Naive (t-7)",
        "Type": "Baseline",
        "MAE": mean_absolute_error(y_test, y_snaive),
        "RMSE": root_mean_squared_error(y_test, y_snaive),
        "WAPE": calculate_wape(y_test, y_snaive),
        "R2": r2_score(y_test, y_snaive)
    })

    # 3. Moving Average (7-day)
    y_ma7 = test_df["Rolling_Mean_7"].values
    results.append({
        "Model": "Moving Average (7-Day)",
        "Type": "Baseline",
        "MAE": mean_absolute_error(y_test, y_ma7),
        "RMSE": root_mean_squared_error(y_test, y_ma7),
        "WAPE": calculate_wape(y_test, y_ma7),
        "R2": r2_score(y_test, y_ma7)
    })

    # 4. Moving Average (14-day)
    y_ma14 = test_df["Rolling_Mean_14"].values
    results.append({
        "Model": "Moving Average (14-Day)",
        "Type": "Baseline",
        "MAE": mean_absolute_error(y_test, y_ma14),
        "RMSE": root_mean_squared_error(y_test, y_ma14),
        "WAPE": calculate_wape(y_test, y_ma14),
        "R2": r2_score(y_test, y_ma14)
    })

    # 5. Croston Classic
    croston_classic = CrostonForecaster(alpha=0.1, variant="classic")
    croston_classic.fit(train_df)
    y_croston = croston_classic.predict(test_df)
    results.append({
        "Model": "Croston Method (Classic)",
        "Type": "Intermittent Baseline",
        "MAE": mean_absolute_error(y_test, y_croston),
        "RMSE": root_mean_squared_error(y_test, y_croston),
        "WAPE": calculate_wape(y_test, y_croston),
        "R2": r2_score(y_test, y_croston)
    })

    # 6. Croston SBA (Syntetos-Boylan)
    croston_sba = CrostonForecaster(alpha=0.1, variant="sba")
    croston_sba.fit(train_df)
    y_sba = croston_sba.predict(test_df)
    results.append({
        "Model": "Croston SBA (Syntetos-Boylan)",
        "Type": "Intermittent Baseline",
        "MAE": mean_absolute_error(y_test, y_sba),
        "RMSE": root_mean_squared_error(y_test, y_sba),
        "WAPE": calculate_wape(y_test, y_sba),
        "R2": r2_score(y_test, y_sba)
    })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "baseline_evaluation.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved baseline evaluation to: {out_csv}")

    return res_df


if __name__ == "__main__":
    run_baselines()
