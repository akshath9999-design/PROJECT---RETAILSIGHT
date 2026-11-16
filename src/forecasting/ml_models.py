"""
Machine Learning Demand Forecasting Module
Develops, trains, tunes, and evaluates gradient boosting and ensemble regressors:
- LightGBM Regressor
- XGBoost Regressor
- CatBoost Regressor
- Random Forest Regressor
Evaluates chronological generalization (Train vs Validation vs Test) and prevents data leakage.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "forecasting")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sum_actual = np.sum(y_true)
    if sum_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_actual)


def train_and_evaluate_forecasting_models() -> tuple:
    print("=" * 60)
    print("STAGE 8 & 9: MACHINE LEARNING FORECASTING DEVELOPMENT & EVALUATION")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(FEATURES_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(FEATURES_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(FEATURES_DIR, "test.csv"))

    feature_cols = [
        "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14", "Lag_28",
        "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Max_7", "Rolling_Min_7",
        "Rolling_Mean_14", "Rolling_Std_14", "Rolling_Mean_28", "Rolling_Std_28",
        "Zero_Demand_Streak", "Sales_Frequency_14", "Demand_CV_28",
        "AvgPrice", "DayOfWeek", "DayOfMonth", "Month", "Quarter",
        "IsWeekend", "DayOfYear_Sin", "DayOfYear_Cos"
    ]

    target_col = "DailyDemand"

    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_val, y_val = val_df[feature_cols], val_df[target_col].values
    X_test, y_test = test_df[feature_cols], test_df[target_col].values

    print(f"[INFO] Train samples: {len(X_train):,}, Val samples: {len(X_val):,}, Test samples: {len(X_test):,}")
    print(f"[INFO] Features ({len(feature_cols)}): {feature_cols}")

    models = {
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=0
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
    }

    eval_results = []
    test_preds = {
        "Date": test_df["Date"],
        "StockCode": test_df["StockCode"],
        "Description": test_df["Description"],
        "ActualDemand": y_test
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train, y_train)

        # Train predictions (check overfitting)
        y_train_pred = np.clip(model.predict(X_train), 0, None)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Validation predictions
        y_val_pred = np.clip(model.predict(X_val), 0, None)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        val_wape = calculate_wape(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        # Test predictions (final holdout)
        y_test_pred = np.clip(model.predict(X_test), 0, None)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_wape = calculate_wape(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        test_preds[f"{name}_Pred"] = np.round(y_test_pred, 2)
        trained_models[name] = model

        # Save model artifact
        model_path = os.path.join(MODELS_DIR, f"{name.lower()}_forecaster.joblib")
        joblib.dump(model, model_path)
        print(f"  [OK] Saved model artifact to: {model_path}")
        print(f"  Val  -> MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, WAPE: {val_wape:.2%}, R2: {val_r2:.3f}")
        print(f"  Test -> MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, WAPE: {test_wape:.2%}, R2: {test_r2:.3f}")

        eval_results.append({
            "Model": name,
            "Type": "Machine Learning",
            "Train_MAE": train_mae,
            "Train_R2": train_r2,
            "Val_MAE": val_mae,
            "Val_RMSE": val_rmse,
            "Val_WAPE": val_wape,
            "Val_R2": val_r2,
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
            "Test_WAPE": test_wape,
            "Test_R2": test_r2
        })

    # Combine with baselines for unified benchmarking
    baseline_path = os.path.join(REPORTS_DIR, "baseline_evaluation.csv")
    if os.path.exists(baseline_path):
        b_df = pd.read_csv(baseline_path)
        for _, row in b_df.iterrows():
            eval_results.append({
                "Model": row["Model"],
                "Type": row["Type"],
                "Train_MAE": np.nan,
                "Train_R2": np.nan,
                "Val_MAE": np.nan,
                "Val_RMSE": np.nan,
                "Val_WAPE": np.nan,
                "Val_R2": np.nan,
                "Test_MAE": row["MAE"],
                "Test_RMSE": row["RMSE"],
                "Test_WAPE": row["WAPE"],
                "Test_R2": row["R2"]
            })

    comparison_df = pd.DataFrame(eval_results).sort_values("Test_MAE")
    comp_csv = os.path.join(REPORTS_DIR, "forecasting_model_comparison.csv")
    comparison_df.to_csv(comp_csv, index=False)
    print(f"\n[OK] Model Comparison Matrix saved to: {comp_csv}")
    print(comparison_df[["Model", "Type", "Test_MAE", "Test_RMSE", "Test_WAPE", "Test_R2"]].to_string(index=False))

    # Identify Best Model by Validation MAE
    ml_only = [r for r in eval_results if r["Type"] == "Machine Learning"]
    best_ml = min(ml_only, key=lambda x: x["Val_MAE"])
    best_name = best_ml["Model"]
    print(f"\n[BEST MODEL SELECTED]: {best_name} (Val MAE: {best_ml['Val_MAE']:.2f}, Test MAE: {best_ml['Test_MAE']:.2f})")

    # Add BestModelPred column to test predictions
    preds_df = pd.DataFrame(test_preds)
    preds_df["BestModelPred"] = preds_df[f"{best_name}_Pred"]
    pred_csv = os.path.join(PREDICTIONS_DIR, "demand_forecast_test_predictions.csv")
    preds_df.to_csv(pred_csv, index=False)
    print(f"[OK] Saved test predictions to: {pred_csv}")

    # Save Best Model metadata
    best_meta = {
        "best_model_name": best_name,
        "feature_columns": feature_cols,
        "metrics": best_ml
    }
    with open(os.path.join(MODELS_DIR, "best_forecaster_meta.json"), "w") as f:
        json.dump(best_meta, f, indent=4)

    return comparison_df, preds_df


if __name__ == "__main__":
    train_and_evaluate_forecasting_models()
