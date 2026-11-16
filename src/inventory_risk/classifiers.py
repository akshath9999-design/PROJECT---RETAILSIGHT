"""
Inventory Risk Classification with SMOTE Imbalance Handling Module
Demonstrates methodologically rigorous machine learning classification for multi-class inventory risk:
1. Class 0: Normal / Healthy Buffer
2. Class 1: Critical Stockout Risk (Severe Minority ~9.8%)
3. Class 2: Excess / Overstock Risk

Rigorous College Criteria Enforced:
- Applied STRICTLY to training partition after chronological split.
- Never applies SMOTE to validation or holdout test sets.
- Direct side-by-side performance benchmarking: BEFORE vs AFTER SMOTE.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "features", "inventory_risk_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "inventory_risk")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
VIZ_DIR = os.path.join(PROJECT_ROOT, "outputs", "visualizations")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")


def evaluate_classifier_suite() -> pd.DataFrame:
    print("=" * 60)
    print("STAGE 11B: INVENTORY RISK CLASSIFICATION & SMOTE EXPERIMENTATION")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    # Chronological Split (Identical to forecasting split)
    unique_dates = np.sort(df["Date"].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * 0.70)
    val_end_idx = int(n_dates * 0.85)

    train_cutoff = unique_dates[train_end_idx]
    val_cutoff = unique_dates[val_end_idx]

    train_df = df[df["Date"] < train_cutoff].copy()
    val_df = df[(df["Date"] >= train_cutoff) & (df["Date"] < val_cutoff)].copy()
    test_df = df[df["Date"] >= val_cutoff].copy()

    feature_cols = [
        "Lag_1", "Lag_2", "Lag_7", "Lag_14",
        "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_14", "Rolling_Mean_28",
        "Zero_Demand_Streak", "Sales_Frequency_14", "Demand_CV_28",
        "AvgPrice", "DayOfWeek", "Month", "IsWeekend",
        "Simulated_Stock"
    ]
    target_col = "Inventory_Risk_Class"

    X_train_raw, y_train_raw = train_df[feature_cols], train_df[target_col].values
    X_val, y_val = val_df[feature_cols], val_df[target_col].values
    X_test, y_test = test_df[feature_cols], test_df[target_col].values

    print(f"[INFO] Train samples: {len(X_train_raw):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print("[INFO] Raw Training Class Distribution:")
    for c, cnt in pd.Series(y_train_raw).value_counts().sort_index().items():
        print(f"  Class {c}: {cnt:,} ({cnt/len(y_train_raw)*100:.1f}%)")

    # Feature Scaling (Fit only on Train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "inventory_scaler.joblib"))

    # Apply SMOTE STRICTLY on Training Set
    print("\n[INFO] Applying SMOTE strictly to Training Set...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_raw)
    print("[OK] Balanced Training Class Distribution after SMOTE:")
    for c, cnt in pd.Series(y_train_smote).value_counts().sort_index().items():
        print(f"  Class {c}: {cnt:,} ({cnt/len(y_train_smote)*100:.1f}%)")

    # Classifiers to benchmark
    def get_models():
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
        }

    comparison_records = []
    trained_artifacts = {}
    best_confusion_matrices = {}

    for regime, X_tr, y_tr in [("Before SMOTE", X_train_scaled, y_train_raw), ("After SMOTE", X_train_smote, y_train_smote)]:
        print(f"\n{'='*30}\nEvaluating Models: {regime}\n{'='*30}")
        models = get_models()

        for name, clf in models.items():
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            macro_rec = recall_score(y_test, y_pred, average="macro")
            macro_prec = precision_score(y_test, y_pred, average="macro")

            # Per-class metrics
            rec_normal = recall_score(y_test, y_pred, labels=[0], average=None)[0]
            rec_stockout = recall_score(y_test, y_pred, labels=[1], average=None)[0]
            rec_overstock = recall_score(y_test, y_pred, labels=[2], average=None)[0]

            f1_normal = f1_score(y_test, y_pred, labels=[0], average=None)[0]
            f1_stockout = f1_score(y_test, y_pred, labels=[1], average=None)[0]
            f1_overstock = f1_score(y_test, y_pred, labels=[2], average=None)[0]

            cm = confusion_matrix(y_test, y_pred)

            comparison_records.append({
                "Model": name,
                "Regime": regime,
                "Accuracy": acc,
                "Balanced_Accuracy": bacc,
                "Macro_F1": macro_f1,
                "Weighted_F1": weighted_f1,
                "Macro_Recall": macro_rec,
                "Macro_Precision": macro_prec,
                "Stockout_Recall": rec_stockout,
                "Stockout_F1": f1_stockout,
                "Overstock_Recall": rec_overstock,
                "Overstock_F1": f1_overstock,
                "Normal_Recall": rec_normal,
                "Normal_F1": f1_normal
            })

            print(f"[{regime}] {name:<18} -> Macro F1: {macro_f1:.3f} | Stockout Recall: {rec_stockout:.3f} | Balanced Acc: {bacc:.3f}")

            if regime == "After SMOTE":
                trained_artifacts[name] = clf
                best_confusion_matrices[name] = cm
                # Save model
                joblib.dump(clf, os.path.join(MODELS_DIR, f"{name.lower()}_smote_classifier.joblib"))
            else:
                best_confusion_matrices[f"{name}_raw"] = cm

    comp_df = pd.DataFrame(comparison_records)
    comp_csv = os.path.join(REPORTS_DIR, "smote_inventory_risk_comparison.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n[OK] SMOTE Comparison Matrix written to: {comp_csv}")

    # Generate Confusion Matrix Visualization (Before vs After SMOTE for Best Model: XGBoost & Random Forest)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    class_labels = ["Normal (0)", "Stockout Risk (1)", "Overstock Risk (2)"]

    for idx, (m_key, title) in enumerate([
        ("RandomForest_raw", "Random Forest (Before SMOTE - Raw)"),
        ("RandomForest", "Random Forest (After SMOTE - Balanced)"),
        ("XGBoost_raw", "XGBoost (Before SMOTE - Raw)"),
        ("XGBoost", "XGBoost (After SMOTE - Balanced)")
    ]):
        ax = axes[idx // 2, idx % 2]
        sns.heatmap(best_confusion_matrices[m_key], annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Inventory Class", fontsize=10)
        ax.set_ylabel("True Inventory Class", fontsize=10)

    fig.tight_layout()
    cm_path = os.path.join(VIZ_DIR, "confusion_matrix_before_after_smote.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved Confusion Matrix visualization to: {cm_path}")

    # Select Best Model (highest Macro F1 in After SMOTE regime)
    smote_only = comp_df[comp_df["Regime"] == "After SMOTE"]
    best_row = smote_only.sort_values("Macro_F1", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_clf = trained_artifacts[best_name]

    print(f"\n[SELECTED BEST RISK CLASSIFIER]: {best_name} (Macro F1: {best_row['Macro_F1']:.3f}, Stockout Recall: {best_row['Stockout_Recall']:.3f})")

    # Generate predictions on Test Set
    test_preds = test_df[["Date", "StockCode", "Description", "Simulated_Stock", "Days_Of_Supply", "Inventory_Risk_Class"]].copy()
    test_preds["Predicted_Risk_Class"] = best_clf.predict(X_test_scaled)
    risk_name_map = {0: "Healthy / Safe", 1: "Critical Stockout Risk", 2: "Excess Overstock Risk"}
    test_preds["Actual_Risk_Label"] = test_preds["Inventory_Risk_Class"].map(risk_name_map)
    test_preds["Predicted_Risk_Label"] = test_preds["Predicted_Risk_Class"].map(risk_name_map)

    # Risk Probabilities
    probs = best_clf.predict_proba(X_test_scaled)
    test_preds["Prob_Normal"] = np.round(probs[:, 0], 3)
    test_preds["Prob_Stockout"] = np.round(probs[:, 1], 3)
    test_preds["Prob_Overstock"] = np.round(probs[:, 2], 3)

    pred_path = os.path.join(PREDICTIONS_DIR, "inventory_risk_test_predictions.csv")
    test_preds.to_csv(pred_path, index=False)
    print(f"[OK] Saved inventory risk predictions to: {pred_path}")

    # Save Best Model Metadata
    meta = {
        "best_classifier_name": best_name,
        "feature_columns": feature_cols,
        "metrics": best_row.to_dict()
    }
    with open(os.path.join(MODELS_DIR, "best_risk_classifier_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    return comp_df


if __name__ == "__main__":
    evaluate_classifier_suite()
