# System Architecture
## Retail Demand Forecasting and Inventory Risk Prediction

### 1. Executive Summary & Problem Framing
In retail supply chains, two opposing operational risks directly impact profitability:
1. **Stockouts (Understocking)**: Resulting in lost revenue, diminished customer goodwill, and expedited freight costs.
2. **Overstocking (Excess Inventory)**: Resulting in bloated working capital, high warehouse holding costs, obsolescence, and margin-eroding markdowns.

Traditional retail approaches rely either on static min-max heuristics or naive moving averages, failing to capture complex cross-temporal patterns, demand intermittency, seasonal spikes, and non-linear risk factors.

This project delivers an enterprise-grade, end-to-end AI/ML solution combining:
- **Continuous Demand Forecasting (Regression & Intermittent Time Series)**: Multi-horizon SKU-level sales forecasting using advanced gradient boosting (LightGBM, XGBoost, CatBoost) and classical intermittent methods (Croston, Syntetos-Boylan).
- **Multi-Class Inventory Risk Prediction (Classification with Imbalance Handling)**: Predicting operational inventory states (Critical Stockout Risk, Safe Stock, Overstock Risk) using synthetic minority oversampling (SMOTE) strictly on training partitions.
- **Supply Chain Segmentation**: Automated ABC-XYZ classification and demand lumpiness diagnostics.
- **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations) values to surface key operational demand drivers.
- **Interactive Decision Support Application**: A production-ready Streamlit analytical suite for inventory managers and retail executives.

---

### 2. High-Level Modular Architecture

```
                          [ Raw Retail Data (UCI Online Retail 541k+ rows) ]
                                                │
                                                ▼
                                    [ 1. Data Cleaning Engine ]
                                    - Cancellation filtering (Invoice 'C')
                                    - Non-product code removal (POST, D, etc.)
                                    - Missing CustomerID handling
                                    - Price/Quantity sanity checks
                                                │
                                                ▼
                                  [ Cleaned Transaction Store ]
                                                │
                      ┌─────────────────────────┴────────────────────────┐
                      ▼                                                  ▼
          [ Time-Series Aggregator ]                            [ SKU Profiler & Segmentation ]
          - Daily SKU-level demand grids                        - ABC Analysis (Pareto Value)
          - Zero-demand imputation                              - XYZ Analysis (Volatility CV)
          - Calendar & holiday alignment                        - Syntetos-Boylan Intermittency
                      │                                                  │
                      ▼                                                  │
        [ 2. Feature Engineering Pipeline ]                              │
        - Lag Features (t-1, t-7, t-14, t-28)                            │
        - Rolling Statistics (Mean, Std, Min, Max)                       │
        - Calendar/Temporal (DoW, Month, Day, IsWeekend)                 │
        - Intermittency & Frequency Indicators                           │
        - Strict Chronological Train/Val/Test Split                      │
                      │                                                  │
                      ├──────────────────────────┐                       │
                      ▼                          ▼                       ▼
          [ 3. Forecasting Engine ]    [ 4. Inventory Risk Engine ] ◄────┘
          - LightGBM Regressor          - Feature Enrichment (Runout, Days Cover)
          - XGBoost Regressor           - Class Imbalance Handling (SMOTE on Train)
          - CatBoost Regressor          - Multi-Class Classifiers (RF, XGBoost, etc.)
          - Croston / SBA Benchmarks    - Risk Scoring & Thresholding
                      │                          │
                      ▼                          ▼
          [ 5. Model Evaluation ]       [ 6. Explainability Module (SHAP) ]
          - MAE, RMSE, WAPE, R²         - Global Summary & Beeswarm
          - Over/Underfitting checks    - Local SKU Waterfall Drivers
                      │                          │
                      └──────────────┬───────────┘
                                     ▼
                    [ 7. Interactive Streamlit Dashboard ]
                    - Executive KPI Command Center
                    - SKU Deep-Dive & Forecast Visualization
                    - Inventory Risk & Reorder Recommendation Matrix
                    - Downloadable Action Reports (CSV / PDF)
```

---

### 3. Chronological Data Partitioning & Leakage Prevention
To ensure methodological validity in retail forecasting:
- **No Random K-Fold Splits**: Random splits cause future information leakage across time series.
- **Chronological Split**:
  - **Training Set**: Earliest 70% of timeline.
  - **Validation Set**: Subsequent 15% of timeline (used for hyperparameter tuning & early stopping).
  - **Test Set**: Final 15% out-of-time horizon (strictly hold-out for unbiased benchmarking).
- **Feature Origin Principle**: All lag and rolling calculations strictly reference observations prior to or at forecast cutoff $t$.
- **SMOTE Boundary**: SMOTE is applied strictly on the training partition of the classification dataset to prevent synthetic data bleed into validation or evaluation sets.
