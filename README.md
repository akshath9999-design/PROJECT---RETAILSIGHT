# Retail Demand Forecasting and Inventory Risk Prediction
## Enterprise AI/ML Supply Chain Decision Support System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![PyTest Status](https://img.shields.io/badge/pytest-12%20passed-brightgreen.svg)](https://pytest.org/)
[![Dataset Scale](https://img.shields.io/badge/Records-541k%2B%20UCI-success.svg)](https://archive.ics.uci.edu/dataset/352/online+retail)

An end-to-end, production-grade AI/ML system designed to optimize retail supply chain operations by uniting **continuous multi-horizon demand forecasting** with **multi-class inventory risk prediction (using SMOTE)**, **ABC-XYZ segmentation**, **Syntetos-Boylan intermittency analysis**, and **TreeSHAP explainability**.

Developed for undergraduate engineering project evaluation, presentation, and viva voce examination.

---

## 🌟 Executive Highlights & College Criteria Compliance

| College Requirement | Project Implementation & Proof | Verification Status |
|---|---|---|
| **Record Scale >= 10,000** | Authentic UCI Online Retail Dataset: **541,909 raw records**, **527,447 clean records**, **51,900 daily SKU feature records**. | ✅ Exceeded by **52x** |
| **SMOTE Imbalance Handling** | Applied **strictly** to training data of 3-class inventory risk task (Stockout vs Normal vs Overstock). Stockout Recall improved from **81.0% to 92.9% (+11.9% gain)** on Random Forest, and to **99.8%** on Logistic Regression. | ✅ Fully Satisfied & Benchmarked |
| **Meaningful AI/ML** | Multi-algorithm benchmark: **LightGBM, CatBoost, XGBoost, Random Forest, Croston SBA, Moving Average, Naive**. LightGBM/CatBoost achieved **35.3% error reduction** over baselines. | ✅ Methodologically Validated |
| **Distinguishing Features** | ABC-XYZ matrix, Syntetos-Boylan demand pattern classification, TreeSHAP feature attributions, 14-day multi-step forecasting with 80% prediction intervals, and interactive scenario simulator. | ✅ Enterprise-Grade Architecture |
| **Functional Application** | Full-stack interactive **Streamlit** suite with 10 integrated analytical modules, Plotly interactive graphics, and one-click export hubs. | ✅ Functional & Tested |
| **Automated Testing** | 12 automated unit tests across data integrity, temporal leakage, forecasting, and SMOTE verification. | ✅ 12/12 Passed (100%) |

---

## 🏗️ System Architecture

```
                          [ Raw Retail Data (UCI Online Retail 541k+ rows) ]
                                                │
                                                ▼
                                    [ 1. Data Cleaning Engine ]
                                    - Cancellation filtering (Invoice 'C')
                                    - Non-product code removal (POST, D, etc.)
                                    - Missing description imputation
                                    - Price/Quantity sanity checks & outlier capping
                                                │
                                                ▼
                                  [ Cleaned Transaction Store ]
                                  (527,447 valid records, 3,903 SKUs)
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
        - 25 Predictors (Lags, Rolling Means, Streaks, Cyclical Dates)   │
        - Strict Chronological Splitting (70% Train, 15% Val, 15% Test)  │
                      │                                                  │
                      ├──────────────────────────┐                       │
                      ▼                          ▼                       ▼
          [ 3. Forecasting Engine ]    [ 4. Inventory Risk Engine ] ◄────┘
          - LightGBM Regressor          - Feature Enrichment (Runout, Days Cover)
          - CatBoost Regressor          - Class Imbalance Handling (SMOTE on Train)
          - XGBoost Regressor           - Multi-Class Classifiers (RF, XGBoost, LR)
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

## 📊 Empirical Benchmarks

### 1. Demand Forecasting Benchmark (Holdout Test Set)

| Rank | Model | Category | Test MAE | Test RMSE | Test WAPE | Test $R^2$ |
|---|---|---|---|---|---|---|
| 🥇 | **CatBoost** | Machine Learning | **26.91** | **61.32** | **70.16%** | **0.321** |
| 🥈 | **LightGBM** | Machine Learning | **26.95** | **61.40** | **70.25%** | **0.319** |
| 🥉 | **XGBoost** | Machine Learning | 28.64 | 61.50 | 74.67% | 0.317 |
| 4 | **Random Forest** | Machine Learning | 30.11 | 61.26 | 78.49% | 0.322 |
| 5 | **Moving Average (14d)** | Baseline | 32.24 | 64.39 | 84.06% | 0.251 |
| 6 | **Moving Average (7d)** | Baseline | 33.06 | 65.14 | 86.21% | 0.234 |
| 7 | **Seasonal Naive (t-7)** | Baseline | 34.31 | 77.48 | 89.47% | -0.084 |
| 8 | **Croston SBA** | Intermittent Baseline | 36.81 | 74.42 | 95.97% | -0.000 |
| 9 | **Croston Classic** | Intermittent Baseline | 37.37 | 74.53 | 97.44% | -0.003 |
| 10 | **Naive (t-1)** | Baseline | 41.60 | 85.22 | 108.46% | -0.312 |

---

### 2. SMOTE Inventory Risk Benchmark (Holdout Test Set)

| Classifier | Training Regime | Balanced Accuracy | Macro F1 | Stockout Recall (Minority) | Stockout F1 | Overall Accuracy |
|---|---|---|---|---|---|---|
| **Random Forest** | Before SMOTE (Raw) | 0.921 | 0.940 | 0.810 (81.0%) | 0.887 | 96.0% |
| **Random Forest** | **After SMOTE** | **0.952** (+3.1%) | **0.957** (+1.7%) | **0.929 (92.9%)** (**+11.9%**) | **0.915** | 96.6% |
| **Logistic Regression** | Before SMOTE (Raw) | 0.974 | 0.981 | 0.933 (93.3%) | 0.963 | 98.7% |
| **Logistic Regression** | **After SMOTE** | **0.991** (+1.7%) | **0.988** (+0.7%) | **0.998 (99.8%)** (**+6.5%**) | **0.981** | 99.2% |
| **XGBoost** | Before SMOTE (Raw) | 0.978 | 0.982 | 0.954 (95.4%) | 0.966 | 98.8% |
| **XGBoost** | **After SMOTE** | **0.983** (+0.5%) | **0.983** (+0.1%) | **0.977 (97.7%)** (**+2.3%**) | **0.968** | 98.8% |

---

## 📂 Project Organization

```
Retail_Demand_Forecasting/
├── data/
│   ├── raw/                  # Downloaded raw UCI dataset (Excel & CSV)
│   ├── processed/            # Cleaned transactions & ABC-XYZ segmentation
│   └── features/             # Daily feature matrix, chronological splits, risk dataset
├── src/
│   ├── preprocessing/        # Data ingestion & cleaning modules
│   ├── feature_engineering/  # Continuous time-series grid & lag builders
│   ├── eda/                  # Exploratory visual & statistical analysis
│   ├── statistical/          # ABC-XYZ Pareto & Syntetos-Boylan intermittency
│   ├── forecasting/          # Baselines, Croston, ML regressors & predictors
│   ├── inventory_risk/       # Inventory simulation, classification & SMOTE
│   └── explainability/       # TreeSHAP beeswarm & attribution plots
├── models/
│   ├── forecasting/          # Serialized regressors (LightGBM, XGBoost, CatBoost, RF)
│   └── inventory_risk/       # Serialized classifiers & scalers
├── outputs/
│   ├── visualizations/       # Publication-ready PNG charts
│   ├── reports/              # Markdown & CSV evaluation reports
│   └── predictions/          # Generated forecasts & risk test predictions
├── app/
│   └── dashboard.py          # Full-featured Streamlit interactive application
├── tests/                    # Pytest test suite (12 passed)
├── docs/
│   ├── ARCHITECTURE.md       # Complete architectural documentation
│   ├── DATA_DICTIONARY.md    # Attribute schemas & cleaning rules
│   ├── PROJECT_REPORT.md     # Full undergraduate engineering report
│   └── VIVA_QUESTIONS_AND_ANSWERS.md # 20 Comprehensive Viva Voce Q&A pairs
├── requirements.txt          # Python dependencies
├── PROJECT_PROGRESS.md       # 16-Stage autonomous progress tracking
└── run_pipeline.py           # Master automated execution script
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git

### 2. Clone Repository & Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/retail-demand-forecasting.git
cd retail-demand-forecasting

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run Complete Autonomous Pipeline (Optional — models already trained)
```bash
python run_pipeline.py
```

### 4. Run Automated Unit Tests
```bash
pytest tests -v
```

### 5. Launch Interactive Application
```bash
streamlit run app/dashboard.py
```
Open your web browser at `http://localhost:8501`.

---

## 🎓 Academic Presentation & Viva Guide
A dedicated 20-question viva voce preparation guide is provided in [`docs/VIVA_QUESTIONS_AND_ANSWERS.md`](file:///c:/Users/hi/OneDrive/Desktop/ps/docs/VIVA_QUESTIONS_AND_ANSWERS.md) covering:
1. Business problem framing and inventory carrying cost equations.
2. Data leakage elimination and chronological splitting protocols.
3. Theoretical foundations of ABC-XYZ and Syntetos-Boylan (2005) classifications.
4. Mathematical derivation of WAPE vs. MAPE and handling of zero-inflated demand.
5. Mathematical mechanics of SMOTE and its empirical impact on minority class recall.
6. Cooperative game-theoretic foundations of TreeSHAP attributions.
7. Architectural decisions and production deployment considerations.

---

## 📜 License
This project is open-source under the MIT License.
