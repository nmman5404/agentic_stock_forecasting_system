from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.modeling.trainer import QuantileLightGBM
from src.modeling.validation import run_walk_forward_validation
from utils.logger import get_logger

logger = get_logger("Predictor")


def generate_7_day_forecast(
    df: pd.DataFrame,
    ticker: str, 
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a 7-day direct multi-step quantile forecast.

    Walk-forward validation is the only model-evaluation source of truth.
    """
    logger.info("Forecast generation started | ticker=%s | horizon_days=7", ticker)

    trainer = QuantileLightGBM(model_params=model_params)
    validation_metrics = run_walk_forward_validation(
        df,
        target_col=trainer.target_col,
        model_params=trainer.config,
    )

    last_row = df.iloc[[-1]]
    x_last = last_row[[col for col in df.columns if col not in {"ticker", "date", "target"}]]

    forecasts = []
    logger.info("Quantile forecast training started | ticker=%s | steps=7 | quantiles=5", ticker)
    for step in range(1, 8):
        forecasts.append(trainer.train_and_predict_step(df, x_last, step))

    logger.info("Forecast generation completed | ticker=%s | horizon_days=7", ticker)
    return {
        "ticker": ticker,
        "current_price": float(df["close"].iloc[-1]) if "close" in df.columns and not df.empty else None,
        "as_of_date": df.index[-1].strftime("%Y-%m-%d") if len(df.index) else None,
        "evaluation_method": "walk_forward",
        "validation_metrics": validation_metrics,
        "forecasts": forecasts,
    }