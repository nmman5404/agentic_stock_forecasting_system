from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from utils.logger import get_logger

logger = get_logger("ModelTrainer")


def load_config() -> Dict[str, Any]:
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class QuantileLightGBM:
    """LightGBM quantile forecaster that predicts future returns, then maps them to prices."""

    def __init__(self, target_col: str = "close", model_params: Optional[Dict[str, Any]] = None):
        self.target_col = target_col
        self.config = model_params.copy() if model_params is not None else load_config().get("lightgbm_params", {})
        self.quantiles = [0.025, 0.1, 0.5, 0.9, 0.975]
        self.features: List[str] = []

    def prepare_data(self, df: pd.DataFrame, step: int = 1):
        """Create supervised data for direct multi-step return forecasting."""
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' is missing from training data.")

        ordered = df.copy()
        if isinstance(ordered.index, pd.DatetimeIndex):
            ordered = ordered.sort_index()

        excluded = {"ticker", "date", "target"}
        candidate_features = [col for col in ordered.columns if col not in excluded]
        self.features = [
            col for col in candidate_features if pd.api.types.is_numeric_dtype(ordered[col])
        ]

        supervised = ordered[self.features].copy()
        supervised["target"] = (
            ordered[self.target_col].shift(-step) - ordered[self.target_col]
        ) / ordered[self.target_col]
        supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])
        supervised = supervised.dropna(axis=0)
        return supervised[self.features], supervised["target"]

    def train_and_predict_step(self, df: pd.DataFrame, x_last: pd.DataFrame, step: int) -> Dict[str, float]:
        X, y = self.prepare_data(df, step=step)
        if X.empty or y.empty:
            raise ValueError(f"Insufficient training rows for forecast step {step}.")

        current_close = float(df[self.target_col].iloc[-1])
        x_last = x_last.reindex(columns=self.features).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        step_forecast: Dict[str, float] = {"step": int(step)}
        for quantile in self.quantiles:
            params = self.config.copy()
            params["objective"] = "quantile"
            params["alpha"] = float(quantile)

            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)

            predicted_return = float(model.predict(x_last)[0])
            step_forecast[f"q_{quantile}"] = current_close * (1.0 + predicted_return)

        return step_forecast
