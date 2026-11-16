"""
Inventory Risk Simulation and Labeling Module
Constructs realistic inventory stock trajectory proxies under standard retail replenishment policies.
Explicitly identifies simulated proxies and documents assumptions.
Generates 3-class inventory risk labels:
  - Class 0: Healthy / Normal Buffer (7 <= Days of Supply <= 30)
  - Class 1: Critical Stockout Risk (Days of Supply < 7)
  - Class 2: Excess / Overstock Risk (Days of Supply > 30)
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
INPUT_FEATURES = os.path.join(FEATURES_DIR, "daily_sku_features.csv")
OUTPUT_RISK_CSV = os.path.join(FEATURES_DIR, "inventory_risk_dataset.csv")


def generate_inventory_risk_data(input_csv: str = INPUT_FEATURES) -> pd.DataFrame:
    print("=" * 60)
    print("STAGE 11A: INVENTORY DYNAMICS SIMULATION & RISK LABELING")
    print("=" * 60)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Features file not found at: {input_csv}")

    df = pd.read_csv(input_csv, parse_dates=["Date"])
    print(f"[INFO] Loaded {len(df):,} daily SKU records for inventory risk modeling.")

    # Sort strictly by SKU and Date
    df = df.sort_values(["StockCode", "Date"]).reset_index(drop=True)

    # Simulation Parameters (Standard Retail Periodic Review Heuristics):
    # - Lead Time: 5 days
    # - Review Period: 7 days
    # - Safety Stock target: 7 days of rolling mean demand
    # - Max Target Stock: 30 days of rolling mean demand

    simulated_records = []

    for sku, grp in df.groupby("StockCode"):
        g = grp.copy().sort_values("Date")
        n_days = len(g)

        # Baseline velocity
        avg_demand = max(1.0, g["DailyDemand"].mean())

        # Initialize opening inventory with reasonable buffer (e.g. 15 days)
        current_stock = avg_demand * 15.0
        pending_orders = []  # list of (arrival_day_index, quantity)

        stock_levels = []
        dos_list = []
        risk_classes = []

        for idx, (_, row) in enumerate(g.iterrows()):
            # 1. Receive pending shipments
            arrived_qty = sum(qty for arr_day, qty in pending_orders if arr_day == idx)
            current_stock += arrived_qty

            # 2. Daily demand satisfies or depletes stock
            demand = row["DailyDemand"]
            units_sold = min(current_stock, demand)
            current_stock = max(0.0, current_stock - demand)

            # 3. Calculate Days of Supply (DOS)
            velocity = max(0.5, row["Rolling_Mean_7"])
            dos = current_stock / velocity

            # 4. Assign Risk Class
            # 0 = Normal / Safe (7 <= DOS <= 30)
            # 1 = Stockout Risk (DOS < 7) -> Imminent shortage
            # 2 = Overstock Risk (DOS > 30) -> Capital stagnation
            if dos < 7.0:
                risk_class = 1  # Stockout Risk
            elif dos > 30.0:
                risk_class = 2  # Overstock Risk
            else:
                risk_class = 0  # Normal / Balanced

            stock_levels.append(round(current_stock, 1))
            dos_list.append(round(dos, 1))
            risk_classes.append(risk_class)

            # 5. Trigger Replenishment Order if stock falls below reorder point (10 days)
            reorder_point = velocity * 10.0
            order_up_to = velocity * 25.0
            # Order periodically or on breach, avoiding duplicate pipeline congestion
            if current_stock <= reorder_point and len(pending_orders) == 0:
                order_qty = max(order_up_to - current_stock, velocity * 7.0)
                lead_time = 5  # 5-day supplier lead time
                pending_orders.append((idx + lead_time, order_qty))

            # Prune completed orders
            pending_orders = [po for po in pending_orders if po[0] > idx]

        g["Simulated_Stock"] = stock_levels
        g["Days_Of_Supply"] = dos_list
        g["Inventory_Risk_Class"] = risk_classes
        simulated_records.append(g)

    risk_df = pd.concat(simulated_records, ignore_index=True)

    # Class Label Distribution
    counts = risk_df["Inventory_Risk_Class"].value_counts()
    percentages = risk_df["Inventory_Risk_Class"].value_counts(normalize=True) * 100

    print("[INFO] Simulated Inventory Risk Class Distribution:")
    class_names = {0: "0: Normal/Healthy Stock", 1: "1: Stockout Risk (Minority)", 2: "2: Overstock Risk (Minority)"}
    for c_id in sorted(counts.keys()):
        print(f"  Class {class_names[c_id]}: {counts[c_id]:,} ({percentages[c_id]:.2f}%)")

    os.makedirs(os.path.dirname(OUTPUT_RISK_CSV), exist_ok=True)
    risk_df.to_csv(OUTPUT_RISK_CSV, index=False)
    print(f"[OK] Inventory risk dataset written to: {OUTPUT_RISK_CSV} ({len(risk_df):,} records)")

    return risk_df


if __name__ == "__main__":
    generate_inventory_risk_data()
