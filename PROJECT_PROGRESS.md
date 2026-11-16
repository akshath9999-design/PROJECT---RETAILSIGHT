# Retail Demand Forecasting and Inventory Risk Prediction
## Project Progress Tracking (16-Stage Autonomous Lifecycle)

| Stage | Title | Status | Completion Details |
|---|---|---|---|
| Stage 1 | Problem Definition & Architecture Design | COMPLETED | System scope, retail business case, dual-objective framing (Forecasting + Risk Classification), and modular architecture documented in `docs/ARCHITECTURE.md`. |
| Stage 2 | Dataset Inspection & Ingestion | COMPLETED | UCI Online Retail dataset downloaded and extracted from archive. 541,909 raw records verified (Dataset ID: 352). |
| Stage 3 | Data Cleaning & Preprocessing | COMPLETED | Filtered 9,288 cancellations ('C'), 2,190 non-product codes, 2,501 invalid numeric entries. Capped bulk outliers (99.95th pct). 527,447 cleaned records retained (97.33% retention). |
| Stage 4 | Advanced Feature Engineering | COMPLETED | 51,900 daily SKU records created across top 150 SKUs. Engineered 25 predictors (Lags 1, 2, 3, 7, 14, 28; Rolling Means/Stds; Zero Streak; Frequency; Cyclical DoY). Strict chronological 70/15/15 split. |
| Stage 5 | Exploratory Data Analysis (EDA) | COMPLETED | Generated and saved high-res figures: `sales_trend_over_time.png`, `top_15_products_revenue.png`, `day_of_week_demand.png`, `monthly_demand_seasonality.png`. Published `outputs/reports/EDA_REPORT.md`. |
| Stage 6 | Advanced Statistical Analysis | COMPLETED | ABC Analysis (Pareto turnover), XYZ Analysis (demand volatility CV), and Syntetos-Boylan demand categorization (Smooth, Intermittent, Erratic, Lumpy) executed across all 3,903 catalog SKUs. |
| Stage 7 | Baseline Forecasting Models | COMPLETED | Benchmarked Naive (t-1), Seasonal Naive (t-7), Moving Average (7d, 14d), Croston Classic, and Croston SBA on holdout test set. Saved `outputs/reports/baseline_evaluation.csv`. |
| Stage 8 | Machine Learning Development | COMPLETED | Trained LightGBM, CatBoost, XGBoost, and Random Forest Regressors using chronological train and validation splits. Saved models in `models/forecasting/`. |
| Stage 9 | Model Evaluation & Benchmarking | COMPLETED | CatBoost (MAE: 26.91, WAPE: 70.16%) and LightGBM (MAE: 26.95, WAPE: 70.25%) beat all baselines (Naive MAE: 41.60, WAPE: 108.46%). Saved `outputs/reports/forecasting_model_comparison.csv`. |
| Stage 10 | SKU-Level Demand Forecasting Module | COMPLETED | Built `src/forecasting/predict.py`. Generated 2,100 multi-step future demand forecasts (14-day horizon) with 80% confidence prediction intervals in `outputs/predictions/future_demand_forecast_14d.csv`. |
| Stage 11 | Inventory Risk Prediction (Classification + SMOTE) | COMPLETED | Simulated $(R, S)$ dynamic periodic review policy. 3 classes: Normal (0), Stockout Risk (1; 9.8% minority), Overstock Risk (2). SMOTE applied strictly on train set. Random Forest Stockout Recall gained +11.9% (81.0% -> 92.9%), Logistic Regression achieved 99.8%. Saved comparison matrix & confusion matrices. |
| Stage 12 | Model Explainability (TreeSHAP) | COMPLETED | Generated TreeSHAP global beeswarm plot (`shap_summary_forecasting.png`) and mean absolute feature importance ranking (`shap_feature_importance_bar.png`). Published `outputs/reports/EXPLAINABILITY_REPORT.md`. |
| Stage 13 | Interactive Streamlit Application | COMPLETED | Developed full-stack Streamlit app in `app/dashboard.py` featuring 10 interactive modules, Plotly time-series graphics, What-If simulator, and export hub. Verified clean compilation. |
| Stage 14 | Complete System Integration | COMPLETED | Integrated master execution pipeline `run_pipeline.py` linking all data loading, preprocessing, feature engineering, modeling, simulation, and explainability stages. |
| Stage 15 | Testing & Debugging | COMPLETED | Authored and executed 12 comprehensive unit tests using pytest (`test_data_integrity.py`, `test_feature_engineering.py`, `test_forecasting_models.py`, `test_inventory_risk_and_smote.py`). 100% pass rate (12 passed). |
| Stage 16 | Deployment & College Viva Documentation | COMPLETED | Created comprehensive `docs/PROJECT_REPORT.md`, `docs/VIVA_QUESTIONS_AND_ANSWERS.md` (20 in-depth Q&A pairs), `docs/DATA_DICTIONARY.md`, and complete `README.md`. |
