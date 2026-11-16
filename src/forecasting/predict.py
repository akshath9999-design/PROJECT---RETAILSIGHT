"""
Multi-Horizon SKU-Level Future Demand Prediction Engine
Generates forward-looking daily/weekly demand forecasts with confidence intervals,
enabling supply chain planning, procurement scheduling, and batch export.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "forecasting")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")


def generate_future_forecast(horizon_days: int = 14, sku_filter: list = None) -> pd.DataFrame:
    print("=" * 60)
    print(f"STAGE 10: MULTI-STEP DEMAND FORECASTING (HORIZON: {horizon_days} DAYS)")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "lightgbm_forecaster.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Forecasting model not found at: {model_path}")

    model = joblib.load(model_path)
    meta_path = os.path.join(MODELS_DIR, "best_forecaster_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]
    rmse_error = meta["metrics"]["Test_RMSE"]

    # Load latest feature state
    daily_features = pd.read_csv(os.path.join(FEATURES_DIR, "daily_sku_features.csv"), parse_dates=["Date"])
    latest_date = daily_features["Date"].max()
    print(f"[INFO] Last historical date: {latest_date.strftime('%Y-%m-%d')}")

    if sku_filter:
        daily_features = daily_features[daily_features["StockCode"].isin(sku_filter)]

    # Forecast forward horizon day-by-day
    future_records = []
    unique_skus = daily_features["StockCode"].unique()

    for sku in unique_skus:
        sku_hist = daily_features[daily_features["StockCode"] == sku].sort_values("Date").tail(60).copy()
        desc = sku_hist["Description"].iloc[-1]
        price = sku_hist["AvgPrice"].iloc[-1]

        # Recent demand series
        recent_demands = list(sku_hist["DailyDemand"].values)
        cur_date = latest_date

        for step in range(1, horizon_days + 1):
            next_date = cur_date + pd.Timedelta(days=step)

            # Build feature vector for step
            lag_1 = recent_demands[-1] if len(recent_demands) >= 1 else 0
            lag_2 = recent_demands[-2] if len(recent_demands) >= 2 else 0
            lag_3 = recent_demands[-3] if len(recent_demands) >= 3 else 0
            lag_7 = recent_demands[-7] if len(recent_demands) >= 7 else lag_1
            lag_14 = recent_demands[-14] if len(recent_demands) >= 14 else lag_1
            lag_28 = recent_demands[-28] if len(recent_demands) >= 28 else lag_1

            roll_7 = np.mean(recent_demands[-7:]) if len(recent_demands) >= 7 else np.mean(recent_demands)
            roll_std_7 = np.std(recent_demands[-7:]) if len(recent_demands) >= 7 else 0.0
            roll_max_7 = np.max(recent_demands[-7:]) if len(recent_demands) >= 7 else lag_1
            roll_min_7 = np.min(recent_demands[-7:]) if len(recent_demands) >= 7 else lag_1

            roll_14 = np.mean(recent_demands[-14:]) if len(recent_demands) >= 14 else roll_7
            roll_std_14 = np.std(recent_demands[-14:]) if len(recent_demands) >= 14 else roll_std_7

            roll_28 = np.mean(recent_demands[-28:]) if len(recent_demands) >= 28 else roll_14
            roll_std_28 = np.std(recent_demands[-28:]) if len(recent_demands) >= 28 else roll_std_14

            # Zero streak
            z_streak = 0
            for d in reversed(recent_demands):
                if d == 0:
                    z_streak += 1
                else:
                    break

            sales_freq_14 = np.mean([1 if d > 0 else 0 for d in recent_demands[-14:]])
            cv_28 = (roll_std_28 / roll_28) if roll_28 > 0 else 0.0

            dow = next_date.dayofweek
            dom = next_date.day
            month = next_date.month
            quarter = (month - 1) // 3 + 1
            is_weekend = 1 if dow in [5, 6] else 0
            doy = next_date.dayofyear
            doy_sin = np.sin(2 * np.pi * doy / 365.25)
            doy_cos = np.cos(2 * np.pi * doy / 365.25)

            feat_dict = {
                "Lag_1": lag_1, "Lag_2": lag_2, "Lag_3": lag_3, "Lag_7": lag_7, "Lag_14": lag_14, "Lag_28": lag_28,
                "Rolling_Mean_7": roll_7, "Rolling_Std_7": roll_std_7, "Rolling_Max_7": roll_max_7, "Rolling_Min_7": roll_min_7,
                "Rolling_Mean_14": roll_14, "Rolling_Std_14": roll_std_14, "Rolling_Mean_28": roll_28, "Rolling_Std_28": roll_std_28,
                "Zero_Demand_Streak": z_streak, "Sales_Frequency_14": sales_freq_14, "Demand_CV_28": cv_28,
                "AvgPrice": price, "DayOfWeek": dow, "DayOfMonth": dom, "Month": month, "Quarter": quarter,
                "IsWeekend": is_weekend, "DayOfYear_Sin": doy_sin, "DayOfYear_Cos": doy_cos
            }

            feat_row = pd.DataFrame([feat_dict])[feature_cols]
            pred_demand = max(0.0, float(model.predict(feat_row)[0]))

            # Append to simulated autoregressive series for multi-step
            recent_demands.append(pred_demand)

            # Confidence bounds (1.96 * RMSE for 95% interval)
            lower_bound = max(0.0, pred_demand - 1.28 * rmse_error)
            upper_bound = pred_demand + 1.28 * rmse_error

            future_records.append({
                "Forecast_Date": next_date.strftime("%Y-%m-%d"),
                "Horizon_Day": step,
                "StockCode": sku,
                "Description": desc,
                "Predicted_Demand": round(pred_demand, 1),
                "Lower_Bound_80": round(lower_bound, 1),
                "Upper_Bound_80": round(upper_bound, 1),
                "Projected_Revenue": round(pred_demand * price, 2)
            })

    forecast_df = pd.DataFrame(future_records)
    out_csv = os.path.join(PREDICTIONS_DIR, f"future_demand_forecast_{horizon_days}d.csv")
    forecast_df.to_csv(out_csv, index=False)
    print(f"[OK] Generated {len(forecast_df):,} future forecast rows across {len(unique_skus)} SKUs.")
    print(f"[OK] Saved to: {out_csv}")

    return forecast_df


if __name__ == "__main__":
    generate_future_forecast(14)
