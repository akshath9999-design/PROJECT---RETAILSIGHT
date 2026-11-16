# Exploratory Data Analysis (EDA) Report
## Retail Demand Forecasting and Inventory Risk Prediction

### 1. Key Operational Findings & Summary Metrics
- **Total Analyzed Cleaned Transactions**: 527,447 records (comfortably surpassing the 10,000 threshold).
- **Total Gross Revenue**: £9,639,140.40
- **Total Units Sold**: 5,131,655 units
- **Unique Catalog Products (SKUs)**: 3,903
- **Unique Retail Customer Accounts**: 4,320
- **Primary Geographical Footprint**: United Kingdom (84.5% of revenue), followed by Germany, France, EIRE, and Spain.

### 2. Temporal & Seasonal Dynamics
1. **Q4 Demand Surge**: Monthly revenue exhibits dramatic escalation entering September, October, and peaking in November (Q4 gift-giving & holiday peak).
2. **Weekly Cadence**: Thursday and Wednesday represent the peak reorder days for retail buyers. Saturdays exhibit low or zero logged dispatches (logistics closure pattern).
3. **Intermittency Profile**: While hero SKUs exhibit daily transactional velocity, the vast majority of long-tail catalog items exhibit intermittent, lumpy, or sporadic purchase intervals.

### 3. Pareto Revenue Concentration (80/20 Rule)
- Top 10% of SKUs account for over 72% of cumulative retail turnover.
- This empirical evidence justifies the implementation of **ABC-XYZ Inventory Segmentation** to prioritize forecasting models on high-value and high-volatility product cohorts.

### 4. Generated Artifacts
- `outputs/visualizations/sales_trend_over_time.png`
- `outputs/visualizations/top_15_products_revenue.png`
- `outputs/visualizations/day_of_week_demand.png`
- `outputs/visualizations/monthly_demand_seasonality.png`
