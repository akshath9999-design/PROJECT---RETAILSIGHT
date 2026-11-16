"""
ABC-XYZ Inventory Segmentation Analysis Module
Performs advanced Pareto revenue classification (ABC) and demand variability classification (XYZ),
generating the classic 9-box supply chain decision matrix.
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CLEANED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_transactions.csv")
ABC_XYZ_OUTPUT = os.path.join(PROJECT_ROOT, "data", "processed", "abc_xyz_segmentation.csv")


def compute_abc_xyz(input_csv: str = CLEANED_CSV_PATH) -> pd.DataFrame:
    print("=" * 60)
    print("STAGE 6A: ABC-XYZ INVENTORY SEGMENTATION")
    print("=" * 60)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found at: {input_csv}")

    df = pd.read_csv(input_csv)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Date"] = pd.to_datetime(df["Date"])

    # Aggregate SKU total metrics
    sku_summary = df.groupby(["StockCode"]).agg(
        Description=("Description", "first"),
        TotalRevenue=("TotalAmount", "sum"),
        TotalUnits=("Quantity", "sum"),
        ActiveDays=("Date", "nunique"),
        AvgDailyUnits=("Quantity", "mean"),
        StdDailyUnits=("Quantity", "std")
    ).reset_index()

    # --- ABC Analysis (Revenue Pareto) ---
    sku_summary = sku_summary.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)
    total_revenue = sku_summary["TotalRevenue"].sum()
    sku_summary["RevenueShare"] = sku_summary["TotalRevenue"] / total_revenue
    sku_summary["CumRevenueShare"] = sku_summary["RevenueShare"].cumsum()

    # Classification Rules:
    # Class A: Cumulative revenue <= 80%
    # Class B: Cumulative revenue 80% - 95%
    # Class C: Cumulative revenue > 95%
    def assign_abc(cum_share):
        if cum_share <= 0.80:
            return "A"
        elif cum_share <= 0.95:
            return "B"
        else:
            return "C"

    sku_summary["ABC_Class"] = sku_summary["CumRevenueShare"].apply(assign_abc)

    # --- XYZ Analysis (Demand Volatility / Coefficient of Variation) ---
    # Aggregate weekly demand per SKU to compute robust CV
    df_weekly_sku = df.groupby(["StockCode", pd.Grouper(key="InvoiceDate", freq="W")]).agg(
        WeeklyQuantity=("Quantity", "sum")
    ).reset_index()

    cv_stats = df_weekly_sku.groupby("StockCode")["WeeklyQuantity"].agg(
        WeeklyMean="mean",
        WeeklyStd="std"
    ).reset_index()

    cv_stats["CV"] = np.where(cv_stats["WeeklyMean"] > 0, cv_stats["WeeklyStd"] / cv_stats["WeeklyMean"], np.nan)
    cv_stats["CV"] = cv_stats["CV"].fillna(2.0)  # High uncertainty for single-week items

    sku_summary = pd.merge(sku_summary, cv_stats[["StockCode", "CV"]], on="StockCode", how="left")
    sku_summary["CV"] = sku_summary["CV"].fillna(2.0)

    # Classification Rules:
    # Class X: CV <= 0.50 (Constant, high predictability)
    # Class Y: 0.50 < CV <= 1.00 (Fluctuating, moderate predictability)
    # Class Z: CV > 1.00 (High volatility, sporadic / erratic demand)
    def assign_xyz(cv):
        if cv <= 0.50:
            return "X"
        elif cv <= 1.00:
            return "Y"
        else:
            return "Z"

    sku_summary["XYZ_Class"] = sku_summary["CV"].apply(assign_xyz)
    sku_summary["ABC_XYZ_Segment"] = sku_summary["ABC_Class"] + sku_summary["XYZ_Class"]

    # Supply Chain Strategy Recommendation
    strategy_map = {
        "AX": "Automated Continuous Replenishment (High Value, Highly Predictable)",
        "AY": "S&OP Demand Forecasting & Safety Buffer (High Value, Moderate Volatility)",
        "AZ": "Make-to-Order / Strict Consignment (High Value, Erratic / Extreme Risk)",
        "BX": "Vendor Managed Inventory / Periodic Review (Medium Value, Predictable)",
        "BY": "Reorder Point with Dynamic Safety Stock (Medium Value, Volatile)",
        "BZ": "Safety Stock Buffer or Regional Hubbing (Medium Value, Erratic)",
        "CX": "Bulk Two-Bin Ordering / Simple Min-Max (Low Value, Predictable)",
        "CY": "Low-Priority Periodic Review (Low Value, Volatile)",
        "CZ": "On-Demand Only / Minimal Stocking (Low Value, Highly Erratic)"
    }
    sku_summary["Inventory_Strategy"] = sku_summary["ABC_XYZ_Segment"].map(strategy_map)

    # Save outputs
    os.makedirs(os.path.dirname(ABC_XYZ_OUTPUT), exist_ok=True)
    sku_summary.to_csv(ABC_XYZ_OUTPUT, index=False)
    print(f"[OK] Saved ABC-XYZ segmentation to: {ABC_XYZ_OUTPUT}")

    # Summary Crosstab
    ct = pd.crosstab(sku_summary["ABC_Class"], sku_summary["XYZ_Class"], margins=True)
    print("[INFO] ABC-XYZ Segment Distribution (SKU Counts):")
    print(ct)

    return sku_summary


if __name__ == "__main__":
    compute_abc_xyz()
