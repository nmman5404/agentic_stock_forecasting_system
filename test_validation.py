import json

import numpy as np
import pandas as pd

from src.modeling.validation import WalkForwardConfig, walk_forward_validate


def build_sample_frame(rows: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    returns = rng.normal(loc=0.001, scale=0.015, size=rows)
    close = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame(index=index)
    df["ticker"] = "TEST"
    df["close"] = close
    df["high"] = close * (1 + np.abs(rng.normal(0.004, 0.002, rows)))
    df["low"] = close * (1 - np.abs(rng.normal(0.004, 0.002, rows)))
    df["volume"] = rng.integers(500_000, 1_500_000, rows)
    df["daily_return"] = pd.Series(close, index=index).pct_change().fillna(0)
    df["vol_change"] = pd.Series(df["volume"], index=index).pct_change().fillna(0)
    df["ma_7"] = pd.Series(close, index=index).rolling(7, min_periods=1).mean()
    df["ma_14"] = pd.Series(close, index=index).rolling(14, min_periods=1).mean()
    df["volatility_7"] = df["daily_return"].rolling(7, min_periods=1).std().fillna(0)
    df["return_lag_1"] = df["daily_return"].shift(1).fillna(0)
    df["return_lag_3"] = df["daily_return"].shift(3).fillna(0)
    return df


if __name__ == "__main__":
    config = WalkForwardConfig(
        initial_train_size=70,
        validation_window=10,
        step_size=10,
        max_windows=2,
        horizon=1,
    )
    report = walk_forward_validate(build_sample_frame(), config=config)

    assert report.status == "PASS"
    assert report.fold_count == 2
    assert report.metrics is not None
    assert report.metrics.mae >= 0
    assert 0 <= report.metrics.directional_accuracy <= 1
    assert 0 <= report.metrics.interval_95_coverage <= 1
    json.dumps(report.to_dict())
    print("Walk-forward validation smoke check passed.")
