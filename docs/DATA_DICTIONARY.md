# Data Dictionary
## Retail Demand Forecasting and Inventory Risk Prediction

### 1. Raw Dataset Source: UCI Online Retail
- **Source**: UCI Machine Learning Repository (Dataset ID: 352)
- **URL**: `https://archive.ics.uci.edu/dataset/352/online+retail`
- **Domain**: Non-store online retail (registered UK multinational retailer)
- **Coverage**: Transactions occurring between 01/12/2010 and 09/12/2011 (1 year)
- **Scale**: 541,909 raw records, 4,372 unique customer accounts, 4,070 unique stock items.

### 2. Raw Schema Attributes

| Column Name | Data Type | Description | Cleaning / Validation Rule |
|---|---|---|---|
| `InvoiceNo` | Nominal / String | A 6-digit integral number uniquely assigned to each transaction. If code starts with 'C', it indicates a cancellation. | Filter out cancellations (`InvoiceNo.str.startswith('C')`) for demand forecasting; track cancellation rate separately. |
| `StockCode` | Nominal / String | A 5-digit integral number uniquely assigned to each distinct product. | Remove non-product service charges: 'POST', 'D', 'DOT', 'M', 'BANK CHARGES', 'AMAZONFEE'. |
| `Description` | Nominal / String | Product name. | Standardize whitespace, uppercase; fill missing descriptions from known StockCode lookup. |
| `Quantity` | Numeric / Integer | The quantities of each product per transaction. | Filter `Quantity > 0` and cap extreme bulk anomalies using 99.9th percentile or IQR. |
| `InvoiceDate` | Datetime | The day and time when each transaction was generated. | Parse format `%d/%m/%Y %H:%M` or ISO standard; extract Date, Year, Month, Day, DayOfWeek, Hour. |
| `UnitPrice` | Numeric / Float | Product price per unit in sterling (£). | Filter `UnitPrice > 0`; identify high-value vs low-value items. |
| `CustomerID` | Nominal / Integer | A 5-digit integral number uniquely assigned to each customer. | Missing values (~25%) represent guest checkouts; imputed with surrogate or retained for aggregate demand. |
| `Country` | Nominal / String | The name of the country where each customer resides. | Categorize into UK (primary market, >90%) vs International. |

---

### 3. Derived SKU-Daily Time-Series Features

| Feature Name | Type | Formula / Origin | Business Interpretation |
|---|---|---|---|
| `DailyDemand` (Target) | Integer | $\sum \text{Quantity}_{\text{SKU}, d}$ | Aggregate daily units sold per SKU. Target for regression. |
| `Revenue` | Float | $\sum (\text{Quantity} \times \text{UnitPrice})$ | Total daily gross revenue generated. |
| `Lag_1` | Integer | $\text{Demand}_{t-1}$ | Immediate previous day sales volume. |
| `Lag_7` | Integer | $\text{Demand}_{t-7}$ | Same-day-last-week sales volume (weekly seasonality). |
| `Lag_14` | Integer | $\text{Demand}_{t-14}$ | Fortnightly sales benchmark. |
| `Lag_28` | Integer | $\text{Demand}_{t-28}$ | 4-week sales benchmark. |
| `Rolling_Mean_7` | Float | $\frac{1}{7} \sum_{i=1}^7 \text{Demand}_{t-i}$ | Short-term smoothed sales run rate. |
| `Rolling_Std_7` | Float | $\sigma(\text{Demand}_{t-1 \dots t-7})$ | Short-term demand volatility. |
| `Rolling_Mean_14` | Float | $\frac{1}{14} \sum_{i=1}^{14} \text{Demand}_{t-i}$ | Medium-term baseline velocity. |
| `Rolling_Mean_28` | Float | $\frac{1}{28} \sum_{i=1}^{28} \text{Demand}_{t-i}$ | Long-term baseline velocity. |
| `DayOfWeek` | Categorical | $0 \dots 6$ (Monday to Sunday) | Day of week seasonality indicator. |
| `IsWeekend` | Binary | $1 \text{ if Saturday/Sunday else } 0$ | Weekend shopping pattern flag. |
| `Month` | Categorical | $1 \dots 12$ | Monthly / quarterly seasonal shifts. |
| `Zero_Demand_Streak` | Integer | Consecutive zero-demand days up to $t-1$ | Indicator of stock depletion or intermittent demand. |
| `Sales_Frequency_14` | Float | $\frac{\text{Count}(\text{Demand}_{t-i} > 0)}{14}$ | Proportion of active selling days in past 2 weeks. |

---

### 4. Inventory Risk Attributes & Classes

| Target / Metric | Type | Definition & Thresholds | Purpose |
|---|---|---|---|
| `Simulated_Stock_Level` | Float | Dynamic stock tracking under replenishment policy: $S_t = \max(0, S_{t-1} - D_t + R_t)$ | Transparent, documented inventory simulation proxy. |
| `Days_Of_Supply` (DOS) | Float | $\frac{S_t}{\max(\text{Rolling\_Mean\_7}, 0.1)}$ | Estimated days until complete stockout under current run rate. |
| `Inventory_Risk_Class` | Categorical | **0: Normal / Healthy** ($7 \le \text{DOS} \le 30$ days)<br>**1: Stockout Risk (Critical)** ($\text{DOS} < 7$ days)<br>**2: Overstock Risk (Excess)** ($\text{DOS} > 30$ days) | 3-Class operational classification target for SMOTE balancing and decision support. |
