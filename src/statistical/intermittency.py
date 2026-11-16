"""
Demand Intermittency and Lumpiness Analysis Module
Implements the industry-standard Syntetos-Boylan (2005) demand pattern classification:
- Average Demand Interval (ADI)
- Squared Coefficient of Variation of Demand Sizes (CV^2)
Classifies every SKU into Smooth, Intermittent, Erratic, or Lumpy demand.
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CLEANED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_transactions.csv")
CLASSIFICATION_OUTPUT = os.path.join(PROJECT_ROOT, "data", "processed", "demand_classification.csv")

# Standard Syntetos-Boylan cutoffs
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


def classify_demand_patterns(input_csv: str = CLEANED_CSV_PATH) -> pd.DataFrame:
    print("=" * 60)
    print("STAGE 6B: DEMAND INTERMITTENCY CLASSIFICATION (SYNTETOS-BOYLAN)")
    print("=" * 60)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found at: {input_csv}")

    df = pd.read_csv(input_csv, parse_dates=["Date"])

    # Aggregate daily demand per SKU
    daily = df.groupby(["StockCode", "Date"]).agg(
        DailyQuantity=("Quantity", "sum")
    ).reset_index()

    # Get complete calendar date range
    min_date = daily["Date"].min()
    max_date = daily["Date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D")
    total_periods = len(all_dates)

    results = []
    sku_grouped = daily.groupby("StockCode")

    for sku, grp in sku_grouped:
        non_zero_demand = grp[grp["DailyQuantity"] > 0]["DailyQuantity"]
        num_non_zero = len(non_zero_demand)

        if num_non_zero <= 1:
            adi = total_periods
            cv2 = 1.0
        else:
            # ADI: Average time interval between non-zero demand periods
            adi = total_periods / num_non_zero
            mean_demand = non_zero_demand.mean()
            std_demand = non_zero_demand.std()
            cv = (std_demand / mean_demand) if mean_demand > 0 else 0
            cv2 = cv ** 2

        # Classification logic:
        # Smooth: Low intermittency (ADI < 1.32), Low lumpiness (CV2 < 0.49)
        # Intermittent: High intermittency (ADI >= 1.32), Low lumpiness (CV2 < 0.49)
        # Erratic: Low intermittency (ADI < 1.32), High lumpiness (CV2 >= 0.49)
        # Lumpy: High intermittency (ADI >= 1.32), High lumpiness (CV2 >= 0.49)
        if adi < ADI_CUTOFF and cv2 < CV2_CUTOFF:
            category = "Smooth"
            recommended_model = "Standard Machine Learning / Linear Regression / ETS"
        elif adi >= ADI_CUTOFF and cv2 < CV2_CUTOFF:
            category = "Intermittent"
            recommended_model = "Croston Method / SBA (Syntetos-Boylan Approximation)"
        elif adi < ADI_CUTOFF and cv2 >= CV2_CUTOFF:
            category = "Erratic"
            recommended_model = "Gradient Boosted Trees (LightGBM/XGBoost) with Robust Loss"
        else:
            category = "Lumpy"
            recommended_model = "Two-Stage Classifier-Regressor / Specialized Intermittent ML"

        results.append({
            "StockCode": sku,
            "TotalActiveDays": num_non_zero,
            "ADI": round(float(adi), 2),
            "CV2": round(float(cv2), 3),
            "Demand_Category": category,
            "Recommended_Forecasting_Technique": recommended_model
        })

    res_df = pd.DataFrame(results)

    # Attach description and ABC segment if available
    abc_path = os.path.join(PROJECT_ROOT, "data", "processed", "abc_xyz_segmentation.csv")
    if os.path.exists(abc_path):
        abc_df = pd.read_csv(abc_path)
        res_df = pd.merge(res_df, abc_df[["StockCode", "Description", "ABC_Class", "XYZ_Class", "ABC_XYZ_Segment"]], on="StockCode", how="left")

    os.makedirs(os.path.dirname(CLASSIFICATION_OUTPUT), exist_ok=True)
    res_df.to_csv(CLASSIFICATION_OUTPUT, index=False)
    print(f"[OK] Demand classification saved to: {CLASSIFICATION_OUTPUT}")

    print("[INFO] Demand Category Breakdown:")
    print(res_df["Demand_Category"].value_counts())

    return res_df


if __name__ == "__main__":
    classify_demand_patterns()
