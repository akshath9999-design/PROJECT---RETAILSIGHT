"""
Croston and Syntetos-Boylan Approximation (SBA) Forecasting Module
Specialized classical intermittent demand forecasting methods for sporadic and lumpy retail items.
"""

import numpy as np
import pandas as pd


class CrostonForecaster:
    """
    Croston's Method and Syntetos-Boylan Approximation (SBA) for intermittent demand.
    Separate exponential smoothing for demand size (z) and inter-arrival interval (p).
    """

    def __init__(self, alpha: float = 0.1, variant: str = "sba"):
        self.alpha = alpha
        self.variant = variant.lower()  # 'classic' or 'sba'
        self.fitted_models = {}

    def fit_series(self, series: pd.Series) -> tuple:
        y = np.array(series, dtype=float)
        non_zero_indices = np.where(y > 0)[0]

        if len(non_zero_indices) == 0:
            return 0.0, 1.0

        # Initialization
        z = y[non_zero_indices[0]]  # First observed demand size
        if len(non_zero_indices) > 1:
            p = float(np.mean(np.diff(non_zero_indices)))
        else:
            p = float(len(y))

        q = 1  # Time elapsed since last non-zero demand
        for t in range(len(y)):
            if y[t] > 0:
                z = self.alpha * y[t] + (1 - self.alpha) * z
                p = self.alpha * q + (1 - self.alpha) * p
                q = 1
            else:
                q += 1

        return z, p

    def fit(self, train_df: pd.DataFrame, sku_col: str = "StockCode", target_col: str = "DailyDemand"):
        for sku, grp in train_df.groupby(sku_col):
            z, p = self.fit_series(grp[target_col])
            self.fitted_models[sku] = (z, p)
        return self

    def predict_sku(self, sku: str) -> float:
        if sku not in self.fitted_models:
            return 0.0
        z, p = self.fitted_models[sku]
        if p <= 0:
            return 0.0
        forecast = z / p
        if self.variant == "sba":
            # Deflate bias via Syntetos-Boylan correction factor
            forecast *= (1.0 - (self.alpha / 2.0))
        return max(0.0, float(forecast))

    def predict(self, test_df: pd.DataFrame, sku_col: str = "StockCode") -> np.ndarray:
        return np.array([self.predict_sku(sku) for sku in test_df[sku_col]])
