# Model Explainability & Feature Attribution Report
## Retail Demand Forecasting and Inventory Risk Prediction

### 1. Executive Summary
Machine learning models are frequently criticized as opaque "black boxes." In enterprise retail operations, inventory planners and category managers must understand **why** an algorithm predicts a demand spike or flags an impending stockout before committing millions in working capital.

To provide algorithmic transparency, this project utilizes **TreeSHAP (SHapley Additive exPlanations)**, rooted in cooperative game theory, ensuring equitable and additive feature attribution.

---

### 2. Primary Demand Drivers (Forecasting Module)
The top three primary drivers of SKU-level demand identified by TreeSHAP are:
1. **`Lag_7`**: Represents short-term momentum and immediate prior velocity. Higher values exert strong positive pressure on forecasted sales.
2. **`AvgPrice`**: Captures intermediate moving baseline run-rate, dampening transient outliers while preserving persistent volume trends.
3. **`Rolling_Mean_28`**: Accounts for weekly cyclicality and same-day purchasing cadence.

#### Key SHAP Observations:
- **Rolling Averages vs. Single Lags**: The model attributes significantly higher structural weight to smoothed rolling statistics (`Rolling_Mean_7`, `Rolling_Mean_14`) than to noisy single-day lags (`Lag_1`, `Lag_2`), demonstrating that it has learned to ignore high-frequency noise and anchor predictions to sustainable velocity.
- **Intermittency Signals**: `Zero_Demand_Streak` and `Sales_Frequency_14` act as crucial negative moderators; as the streak of zero-demand days increases, the model systematically deflates forecasted demand.
- **Calendar & Seasonality**: `DayOfWeek` and `Month` introduce moderate cyclical shifts, capturing weekend lulls and Q4 holiday shopping acceleration.

---

### 3. Inventory Risk Attribution (Classification Module)
For the 3-class inventory risk prediction (Normal vs. Stockout vs. Overstock):
- **Stockout Risk (Class 1)** is primarily driven by:
  - Low `Simulated_Stock` relative to elevated `Rolling_Mean_7` velocity.
  - High `Sales_Frequency_14` indicating persistent daily drain.
- **Overstock Risk (Class 2)** is primarily driven by:
  - High `Simulated_Stock` coupled with long `Zero_Demand_Streak` and declining velocity.
- **SMOTE Impact**: By synthesizing minority instances, the classifier learns sharp decision boundaries around the critical stockout horizon (Days of Supply < 7) without getting blinded by the overwhelming majority of safe and overstocked items.

---

### 4. Correlation vs. Causation Warning
- High lag sales correlate strongly with tomorrow's demand, but do not *cause* consumer purchases.
- Price adjustments (`AvgPrice`) reflect both price elasticity and catalog positioning; caution is advised when using price features as prescriptive causal levers without controlled A/B testing.

---

### 5. Generated Artifacts
- `outputs/visualizations/shap_summary_forecasting.png`
- `outputs/visualizations/shap_feature_importance_bar.png`
