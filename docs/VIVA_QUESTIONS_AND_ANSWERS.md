# Comprehensive Viva Voce Examination Guide
## Retail Demand Forecasting and Inventory Risk Prediction

This document prepares the engineering student team for exhaustive questioning during project evaluation, presentation, and viva voce with external examiners.

---

### Category 1: Problem Definition & Business Context

#### Q1: What is the core business problem your project solves?
**Answer**:
Retail supply chains operate under constant tension between two conflicting risks:
1. **Stockouts**: When customer demand exceeds on-hand inventory, resulting in lost sales, damaged customer loyalty, and emergency expedited freight fees.
2. **Overstocking**: When excess inventory accumulates, tying up working capital, inflating warehousing carrying costs (typically 20-30% of inventory value annually), and forcing margin-killing price markdowns.

Our project delivers a unified dual-engine AI/ML solution:
- A **continuous demand forecasting engine** predicting SKU-level multi-step future sales using Gradient Boosted Decision Trees (LightGBM/CatBoost) and intermittent time series baselines (Croston/SBA).
- A **multi-class inventory risk prediction engine** classifying items into Critical Stockout Risk, Healthy Buffer, or Excess Overstock, utilizing **SMOTE** to handle severe class imbalance on training partitions.

---

#### Q2: What distinguishes your project from ordinary sales prediction dashboards?
**Answer**:
Standard academic sales dashboards typically apply linear regression or simple ARIMA to total aggregated company sales. Our project introduces significant distinguishing technical features:
1. **SKU-Level Granular Modeling**: Distinct daily forecasting across high-volume and intermittent catalog items rather than company-wide aggregates.
2. **ABC-XYZ Matrix Segmentation**: Combining Pareto revenue contribution (ABC) with demand volatility coefficient of variation (XYZ).
3. **Syntetos-Boylan Intermittency Classification**: Categorizing every product into Smooth, Intermittent, Erratic, or Lumpy demand using Average Demand Interval (ADI) and $CV^2$.
4. **Rigorous Class Imbalance Remediation via SMOTE**: Treating stockout detection as a minority classification problem and demonstrating clear before-and-after benchmark gains.
5. **Leakage-Free Chronological Splitting**: Enforcing strict temporal boundaries (70% train, 15% validation, 15% holdout test).
6. **Explainable AI (TreeSHAP)**: Explaining the exact directional impact of rolling averages and intermittency signals.
7. **Interactive What-If Simulation**: Providing real-time parameter tweaking in a Streamlit application.

---

### Category 2: Dataset, Preprocessing & Feature Engineering

#### Q3: Which dataset was used, and how did you verify the 10,000+ record college requirement?
**Answer**:
We utilized the authentic, peer-reviewed **UCI Online Retail Dataset** (Dataset ID: 352) containing genuine transactional logs from a UK-based multinational online retailer.
- **Raw Scale**: 541,909 records across 4,372 customer accounts and 4,070 SKUs.
- **Cleaned Scale**: 527,447 records retained after rigorous data cleaning (97.33% retention).
- **Daily Feature Matrix**: 51,900 continuous daily SKU time-series records.
All partitions exceed the college requirement of 10,000 records by a substantial margin.

---

#### Q4: What data cleaning steps were performed, and why were cancellations handled separately?
**Answer**:
1. **Cancellation Filtering**: In the raw dataset, transaction codes starting with 'C' indicate cancellations or merchandise returns (9,288 rows; 1.71%). If left unaddressed, negative quantities would distort true sales velocity. They were isolated to avoid contaminating future demand modeling.
2. **Non-Product Code Removal**: Codes such as `POST`, `D` (Discount), `M` (Manual), `BANK CHARGES`, and `AMAZONFEE` were removed as they represent operational fees rather than physical products.
3. **Numeric Sanity**: Enforced `Quantity > 0` and `UnitPrice > 0`, removing 2,501 invalid records.
4. **Outlier Capping**: Capped quantities at the 99.95th percentile (600 units) and unit prices at £50.00 to prevent distortion from bulk wholesale adjustments.
5. **Description Imputation**: Missing descriptions were imputed using a verified dictionary mapping from identical `StockCode` entries.

---

#### Q5: What is "Data Leakage" in retail forecasting, and how did your architecture prevent it?
**Answer**:
Data leakage occurs when information from the future or from the test partition inadvertently influences model training or feature generation.
We prevented data leakage through three strict design rules:
1. **Chronological Splitting**: We used strict temporal cutoffs:
   - Train: Earliest 70% of dates (< 2011-08-28; 36,300 rows)
   - Validation: Next 15% (2011-08-28 to 2011-10-19; 7,800 rows)
   - Holdout Test: Final 15% (>= 2011-10-19; 7,800 rows)
   We strictly avoided random K-Fold cross-validation, which shuffles future points into the past.
2. **Shifted Rolling Windows**: All rolling features (`Rolling_Mean_7`, `Rolling_Std_14`, etc.) and lag features were computed using `shift(1)`. At forecast origin $t$, only data up to $t-1$ was utilized.
3. **SMOTE Boundary**: SMOTE resampling was fitted strictly on the training partition after splitting. Validation and test sets were left untouched in their raw distributions.

---

### Category 3: Advanced Statistical & Supply Chain Analysis

#### Q6: Explain ABC Analysis and XYZ Analysis. How do they combine into a 9-box matrix?
**Answer**:
- **ABC Analysis** is based on Pareto's 80/20 principle, segmenting products by their cumulative gross revenue:
  - **Class A**: Top 80% of revenue (~20% of SKUs). High capital importance.
  - **Class B**: Next 15% of revenue (80-95% cumulative). Moderate importance.
  - **Class C**: Remaining 5% of revenue (95-100% cumulative). Long-tail low-value items.
- **XYZ Analysis** segments products by demand predictability using the Coefficient of Variation ($CV = \frac{\sigma}{\mu}$):
  - **Class X**: $CV \le 0.50$ (Steady, regular, highly predictable).
  - **Class Y**: $0.50 < CV \le 1.00$ (Moderate fluctuation, seasonal).
  - **Class Z**: $CV > 1.00$ (High volatility, erratic or sporadic).
- **The 9-Box Matrix**: Combining both yields strategic operational directives:
  - **AX**: Automated continuous replenishment via electronic data interchange (EDI).
  - **AZ**: High-revenue but erratic demand; requires strict customer pre-orders or consignment stocking.
  - **CX**: Low value, predictable; managed with simple two-bin visual Kanban.
  - **CZ**: Low value, erratic; stocked on-demand with minimal capital commitment.

---

#### Q7: What is the Syntetos-Boylan (2005) demand pattern classification?
**Answer**:
Syntetos and Boylan demonstrated that conventional forecasting metrics fail when applied to intermittent items. They proposed categorizing demand using two independent metrics:
1. **Average Demand Interval (ADI)**: $\frac{\text{Total Time Periods}}{\text{Number of Non-Zero Periods}}$. Threshold: **1.32**.
2. **Squared Coefficient of Variation of Demand Sizes ($CV^2$)**: $(\frac{\sigma}{\mu})^2$ computed only over active transaction periods. Threshold: **0.49**.

This defines four operational quadrants:
- **Smooth** ($ADI < 1.32, CV^2 < 0.49$): Regular arrivals and consistent order sizes. Standard regression/ARIMA is optimal.
- **Intermittent** ($ADI \ge 1.32, CV^2 < 0.49$): Sporadic arrivals with consistent order sizes. Croston or SBA method is recommended.
- **Erratic** ($ADI < 1.32, CV^2 \ge 0.49$): Frequent arrivals with highly volatile order sizes. Tree-based ensembles with robust loss functions are recommended.
- **Lumpy** ($ADI \ge 1.32, CV^2 \ge 0.49$): Both sporadic arrival and volatile size. Requires two-stage classification-regression models.

In our e-commerce dataset, 3,284 SKUs exhibited lumpy demand, validating our multi-faceted approach.

---

### Category 4: Forecasting Algorithms & Evaluation

#### Q8: What baseline forecasting models did you implement, and why are baselines necessary?
**Answer**:
Baselines establish an empirical minimum performance threshold that advanced machine learning models must convincingly beat to justify their computational complexity:
1. **Naive ($y_t = y_{t-1}$)**: Tomorrow's demand equals today's demand.
2. **Seasonal Naive ($y_t = y_{t-7}$)**: Tomorrow's demand equals the demand on the same day last week.
3. **Moving Averages (7-day and 14-day)**: Rolling mean of past observations.
4. **Croston Classic and Croston SBA**: Exponential smoothing decomposed into demand size and inter-arrival intervals.

In our experiments:
- Naive achieved a Test MAE of 41.60 and WAPE of 108.46%.
- Moving Average (14-day) achieved a Test MAE of 32.24 and WAPE of 84.06%.
- Our best ML models (CatBoost and LightGBM) achieved Test MAEs of **26.91** and **26.95** with a WAPE of **70.16%**, demonstrating an unambiguous **35.3% error reduction** over naive baselines.

---

#### Q9: What is WAPE, and why is it preferred over MAPE in retail demand forecasting?
**Answer**:
**MAPE** (Mean Absolute Percentage Error) is calculated as:
$$\text{MAPE} = \frac{1}{n} \sum_{t=1}^n \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100$$
In intermittent retail demand, zero-demand days ($y_t = 0$) occur frequently. When $y_t = 0$, MAPE involves division by zero, resulting in undefined or infinitely large errors. Even with small demand values ($y_t = 1$), small absolute deviations generate disproportionate percentage penalties.

**WAPE** (Weighted Absolute Percentage Error) solves this by weighting absolute errors by the total aggregate demand:
$$\text{WAPE} = \frac{\sum_{t=1}^n |y_t - \hat{y}_t|}{\sum_{t=1}^n y_t}$$
WAPE is always mathematically defined (unless total demand is zero), scale-independent, and directly interpretable by retail executives as the percentage of total sales volume misforecasted.

---

#### Q10: Why did LightGBM and CatBoost outperform Random Forest and XGBoost?
**Answer**:
1. **Histogram-Based Leaf-Wise Tree Growth**: LightGBM grows trees leaf-wise (best-first) with optimal split selection rather than level-wise, allowing it to capture asymmetric retail demand spikes more effectively.
2. **Handling Continuous Zero-Inflated Distributions**: Both LightGBM and CatBoost feature specialized regularization that prevents overfitting to long zero-demand streaks.
3. **CatBoost Symmetric Trees**: CatBoost's oblivious trees provide strong regularization against noise in volatile long-tail retail items.
4. **Random Forest Limitations**: Random forest averages predictions across deep unweighted trees, causing it to over-smooth acute demand spikes, resulting in a higher Test MAE (30.11 vs 26.91).

---

### Category 5: Inventory Risk, Classification & SMOTE

#### Q11: Explain the inventory risk classification task. Why was it framed as a classification problem?
**Answer**:
While continuous demand forecasting predicts expected sales volume, inventory managers require discrete, actionable operational decisions: "Do I need to place an emergency reorder today, or am I in danger of holding bloated stock?"

We formulated this as a 3-class classification problem based on forward **Days of Supply (DOS)**:
- **Class 0 (Normal / Safe Buffer)**: $7 \le \text{DOS} \le 30$ days. Healthy operational state.
- **Class 1 (Critical Stockout Risk)**: $\text{DOS} < 7$ days. Imminent stock depletion requiring emergency procurement.
- **Class 2 (Excess Overstock Risk)**: $\text{DOS} > 30$ days. Stagnant working capital requiring promotions or markdown.

---

#### Q12: Why was SMOTE necessary, and how does it work?
**Answer**:
In retail inventory, stockout events represent an acute minority class (~9.8% of daily SKU observations). Standard classification algorithms maximize overall accuracy, leading them to predict the majority classes (Normal and Overstock) and ignore critical stockout instances.

**SMOTE** (Synthetic Minority Over-sampling Technique) operates by:
1. Identifying instances of the minority class in feature space.
2. Finding their $k$-nearest minority neighbors ($k=5$ default).
3. Synthesizing new artificial instances along the line segments connecting the minority instance and its neighbors:
   $$\vec{x}_{\text{new}} = \vec{x}_i + \lambda \cdot (\vec{x}_{zi} - \vec{x}_i), \quad \lambda \sim U(0, 1)$$

Unlike random oversampling (which simply duplicates minority examples and causes severe overfitting), SMOTE expands the convex decision region of the minority class, enabling the model to learn generalizable decision boundaries.

---

#### Q13: What were the specific rules governing your implementation of SMOTE?
**Answer**:
In strict compliance with academic standards:
1. **Train-Only Boundary**: SMOTE was fitted exclusively on the training partition (`X_train, y_train`).
2. **Purity of Holdout Sets**: Neither validation nor test partitions were resampled. Testing on synthetic instances is academically invalid and produces artificially inflated metrics.
3. **Target Isolation**: SMOTE was applied solely to categorical classification, never to continuous regression demand forecasting.
4. **Direct Benchmark**: Performance was empirically compared before and after SMOTE on identical test sets.

---

#### Q14: What were the empirical results of applying SMOTE?
**Answer**:
On the holdout test set, applying SMOTE strictly on the training partition yielded significant performance improvements:
- **Random Forest**:
  - Stockout Recall improved from **0.810 (81.0%) to 0.929 (92.9%)**, an **+11.9% absolute gain** in catching critical shortages.
  - Balanced Accuracy increased from **0.921 to 0.952**.
  - Macro F1-score increased from **0.940 to 0.957**.
- **Logistic Regression**:
  - Stockout Recall improved from **0.933 to 0.998 (+6.5% gain)**.
  - Balanced Accuracy reached **0.991**.
- **XGBoost**:
  - Stockout Recall reached **0.977 (+2.3% gain)** with Balanced Accuracy of **0.983**.

This proved that SMOTE successfully taught the classifiers to prioritize rare stockout events without sacrificing overall accuracy.

---

### Category 6: Explainable AI & TreeSHAP

#### Q15: What is SHAP, and why did you choose TreeSHAP over simple feature importance?
**Answer**:
Default Gini impurity or split-based feature importance in tree models suffers from two major flaws:
1. **Bias toward high-cardinality continuous features**: Split counts inflate the apparent importance of continuous noise variables.
2. **Lack of directionality**: Split counts indicate that a feature was used, but cannot reveal whether higher values increase or decrease predicted demand.

**TreeSHAP** is rooted in cooperative game theory (Shapley values). It computes the marginal contribution of each feature across all possible feature subsets:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$
TreeSHAP satisfies three fundamental theoretical guarantees:
- **Efficiency**: The sum of SHAP values equals the difference between model output and expected value.
- **Symmetry**: Features contributing equally receive identical attributions.
- **Additivity**: Attributions can be summed across ensemble trees.

---

#### Q16: What were the key TreeSHAP findings regarding demand drivers?
**Answer**:
1. **Rolling Averages Dominance**: `Rolling_Mean_7` and `Rolling_Mean_14` contributed the largest positive SHAP values, proving that the model anchors demand to smoothed weekly momentum rather than single-day noise (`Lag_1`).
2. **Intermittency Damping**: `Zero_Demand_Streak` generated strong negative SHAP values; as consecutive zero-sales days increase, the model systematically depresses forecasted demand, preventing false spikes.
3. **Weekly Cadence**: `DayOfWeek` generated positive SHAP contributions for Wednesday and Thursday (mid-week restocking peak) and negative values for weekends.

---

### Category 7: System Integration, Testing & Deployment

#### Q17: Describe your software architecture and project structure.
**Answer**:
The project adheres to clean code, modular software architecture principles:
- `data/`: Segregated into `raw/`, `processed/`, and `features/`.
- `src/`: Modular Python packages:
  - `preprocessing/`: Ingestion and data cleaning (`download_data.py`, `cleaner.py`).
  - `feature_engineering/`: Daily continuous grid builder (`features.py`).
  - `eda/`: Exploratory analysis and chart generation (`eda_analysis.py`).
  - `statistical/`: ABC-XYZ and Syntetos-Boylan engines (`abc_xyz.py`, `intermittency.py`).
  - `forecasting/`: Baselines, ML regressors, Croston, multi-step predictors (`baselines.py`, `ml_models.py`, `croston.py`, `predict.py`).
  - `inventory_risk/`: Inventory simulation and SMOTE classification suite (`risk_simulation.py`, `classifiers.py`).
  - `explainability/`: TreeSHAP attribution generator (`explainer.py`).
- `models/`: Serialized Joblib model binaries and JSON metadata.
- `outputs/`: Dedicated directories for `visualizations/`, `reports/`, and `predictions/`.
- `app/`: Production-ready Streamlit analytical suite (`dashboard.py`).
- `tests/`: Automated pytest unit test suite (`test_*.py`).
- `run_pipeline.py`: Master automated pipeline orchestrating the entire lifecycle.

---

#### Q18: What automated tests were implemented, and what were the results?
**Answer**:
We implemented 12 automated unit tests using `pytest`:
1. `test_data_integrity.py`: Confirmed raw record count (>500k), cleaned count (>527k), verified the >= 10,000 record college rule, and validated the total absence of negative quantities and cancellations.
2. `test_feature_engineering.py`: Verified daily feature matrix scale (>50k rows), strictly chronological split monotonicity, and confirmed absence of target leakage in lag features.
3. `test_forecasting_models.py`: Verified model artifact existence, confirmed ML models beat naive baselines, and validated 14-day prediction bounds ($Upper \ge Lower \ge 0$).
4. `test_inventory_risk_and_smote.py`: Verified 3-class distribution, confirmed SMOTE improved minority stockout recall, and verified probability conservation ($\sum P = 1.0$).

**Result**: 12 passed in 1.80s (100% pass rate).

---

#### Q19: How is the application deployed, and how can an examiner run it locally?
**Answer**:
The application is fully containerizable and runnable in two simple commands:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch interactive dashboard
streamlit run app/dashboard.py
```
It is also pre-configured for one-click deployment to **Streamlit Community Cloud** or containerized via **Docker**.

---

### Category 8: Limitations & Critical Thinking

#### Q20: What are the limitations of this project, and how would you address them in future work?
**Answer**:
1. **Simulated Inventory Levels**: The UCI dataset captures sales transactions, not warehouse stock registers or supplier replenishments. We resolved this by explicitly simulating dynamic stock levels under a documented $(R, S)$ periodic review policy, but in real production, direct ERP warehouse database integration would supply actual physical stock counts.
2. **Absence of External Macro Variables**: Unplanned promotions, competitor pricing, and regional weather were not present in the raw data. Incorporating external promotional calendars and Google Trends signals would improve forecast accuracy during non-cyclical events.
3. **Deep Learning Architectures**: Future iterations could benchmark Transformer architectures such as the Temporal Fusion Transformer (TFT) or N-BEATS against our gradient boosted decision trees.
