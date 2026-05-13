import numpy as np
import pandas as pd

from src.monitoring.drift_detector import detect_drift
from src.monitoring.regime_detector import detect_regime
from src.risk.risk_engine import calculate_risk_report


def build_frame(rows: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    returns = rng.normal(0.001, 0.012, rows)
    close = 100 * np.cumprod(1 + returns)
    volume = rng.integers(700_000, 1_400_000, rows)
    df = pd.DataFrame(index=index)
    df["ticker"] = "TEST"
    df["close"] = close
    df["high"] = close * 1.01
    df["low"] = close * 0.99
    df["volume"] = volume
    df["daily_return"] = df["close"].pct_change().fillna(0)
    df["vol_change"] = df["volume"].pct_change().fillna(0)
    df["ma_7"] = df["close"].rolling(7, min_periods=1).mean()
    df["ma_14"] = df["close"].rolling(14, min_periods=1).mean()
    df["volatility_7"] = df["daily_return"].rolling(7, min_periods=1).std().fillna(0)
    return df


if __name__ == "__main__":
    frame = build_frame()
    regime = detect_regime(frame)
    drift = detect_drift(frame.iloc[:-40], frame.iloc[-40:])
    forecast_data = {
        "current_price": 100.0,
        "forecasts": [
            {"step": 1, "q_0.025": 97.0, "q_0.1": 98.0, "q_0.5": 101.0, "q_0.9": 103.0, "q_0.975": 104.0},
            {"step": 7, "q_0.025": 94.0, "q_0.1": 96.0, "q_0.5": 104.0, "q_0.9": 108.0, "q_0.975": 112.0},
        ],
    }
    validation = {"metrics": {"directional_accuracy": 0.56, "interval_95_coverage": 0.78, "mape": 0.03, "rmse": 2.0}}
    risk = calculate_risk_report(forecast_data, validation, regime, drift)

    assert regime["volatility_regime"] in {"LOW_VOLATILITY", "NORMAL_VOLATILITY", "HIGH_VOLATILITY", "EXTREME_VOLATILITY"}
    assert drift["severity"] in {"LOW", "MEDIUM", "HIGH"}
    assert risk["preliminary_signal"] in {"BUY", "SELL", "HOLD", "WATCH", "MANUAL_REVIEW"}
    assert 0 <= risk["signal_confidence"] <= 1
    print("Monitoring and risk smoke check passed.")
