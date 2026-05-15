from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("RegimeDetector")


def detect_regime(processed_df: pd.DataFrame) -> Dict[str, Any]:
    _validate_regime_input(processed_df)

    df = processed_df.sort_index().copy()
    notes: List[str] = []
    warnings: List[str] = []

    volatility_regime, vol_metrics = _volatility_regime(df)
    trend_regime, trend_metrics, trend_warnings = _trend_regime(df)
    volume_regime, volume_metrics = _volume_regime(df)
    warnings.extend(trend_warnings)

    final_regime_label = f"{volatility_regime}__{trend_regime}__{volume_regime}"
    metrics = {
        **vol_metrics,
        **trend_metrics,
        **volume_metrics,
    }
    notes.append(
        "Regime components computed independently: volatility=%s, trend=%s, volume=%s."
        % (volatility_regime, trend_regime, volume_regime)
    )

    report = {
        "volatility_regime": volatility_regime,
        "trend_regime": trend_regime,
        "volume_regime": volume_regime,
        "final_regime_label": final_regime_label,
        "metrics": metrics,
        "warnings": warnings,
        "regime_notes": notes,
        "liquidity_regime": volume_regime,
    }
    logger.info(
        "Regime report generated | volatility=%s | trend=%s | volume=%s | label=%s",
        volatility_regime,
        trend_regime,
        volume_regime,
        final_regime_label,
    )
    return report


def _validate_regime_input(processed_df: pd.DataFrame) -> None:
    if not isinstance(processed_df, pd.DataFrame):
        raise TypeError("Regime detection requires a pandas DataFrame input.")
    if processed_df.empty:
        raise ValueError("Regime detection requires a non-empty processed dataset.")
    if "close" not in processed_df.columns:
        raise ValueError("Regime detection requires a close column.")
    close = processed_df["close"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(close) < 22:
        raise ValueError("Regime detection requires at least 22 valid close observations.")


def _volatility_regime(df: pd.DataFrame) -> tuple[str, Dict[str, Optional[float]]]:
    returns = df["daily_return"] if "daily_return" in df.columns else df["close"].pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 20:
        raise ValueError("Volatility regime requires at least 20 return observations.")

    current_vol = float(returns.tail(20).std() or 0.0)
    rolling_vol = returns.rolling(20).std().dropna()
    vol_percentile = _percentile_rank(rolling_vol, current_vol)

    if vol_percentile < 0.25:
        label = "LOW_VOLATILITY"
    elif vol_percentile < 0.75:
        label = "NORMAL_VOLATILITY"
    elif vol_percentile < 0.92:
        label = "HIGH_VOLATILITY"
    else:
        label = "EXTREME_VOLATILITY"

    return label, {
        "vol_percentile": round(vol_percentile, 6),
        "current_vol_20d": round(current_vol, 6),
    }


def _trend_regime(df: pd.DataFrame) -> tuple[str, Dict[str, float], List[str]]:
    ma7 = df["close"].rolling(7).mean()
    ma21 = df["close"].rolling(21).mean()
    latest_ma21 = float(ma21.iloc[-1]) if not pd.isna(ma21.iloc[-1]) else 0.0
    if latest_ma21 <= 0:
        raise ValueError("Trend regime requires a positive 21-day moving average.")

    ma_gap = float((ma7.iloc[-1] - ma21.iloc[-1]) / latest_ma21)
    return_20d = _safe_float(df["close"].pct_change(20).iloc[-1])
    return_5d = _safe_float(df["close"].pct_change(5).iloc[-1])

    if ma_gap > 0.015 and return_20d > 0.03:
        label = "UPTREND"
    elif ma_gap < -0.015 and return_20d < -0.03:
        label = "DOWNTREND"
    elif abs(ma_gap) <= 0.015 and abs(return_20d) <= 0.04:
        label = "SIDEWAYS"
    else:
        label = "MIXED_TREND"

    warnings: List[str] = []
    if label == "UPTREND" and return_5d < -0.02:
        warnings.append("SHORT_TERM_PULLBACK")
    if label == "DOWNTREND" and return_5d > 0.02:
        warnings.append("SHORT_TERM_REBOUND")

    return label, {
        "ma_gap": round(ma_gap, 6),
        "return_20d": round(return_20d, 6),
        "return_5d": round(return_5d, 6),
    }, warnings


def _volume_regime(df: pd.DataFrame) -> tuple[str, Dict[str, Optional[float]]]:
    if "volume" not in df.columns:
        return "NORMAL_VOLUME", {"volume_zscore": None, "recent_volume_ratio": None}

    volume = df["volume"].replace([np.inf, -np.inf], np.nan).dropna()
    if volume.empty:
        return "NORMAL_VOLUME", {"volume_zscore": None, "recent_volume_ratio": None}

    trailing = volume.tail(60)
    volume_mean = float(trailing.mean() or 0.0)
    volume_std = float(trailing.std() or 0.0)
    volume_zscore = (float(volume.iloc[-1]) - volume_mean) / volume_std if volume_std > 0 else 0.0
    recent_volume_ratio = float(volume.tail(5).mean() / volume_mean) if volume_mean > 0 else 1.0

    if volume_zscore >= 2.5 or recent_volume_ratio >= 1.8:
        label = "VOLUME_SPIKE"
    elif recent_volume_ratio <= 0.45:
        label = "LOW_VOLUME"
    else:
        label = "NORMAL_VOLUME"

    return label, {
        "volume_zscore": round(volume_zscore, 6),
        "recent_volume_ratio": round(recent_volume_ratio, 6),
    }


def _percentile_rank(values: pd.Series, current_value: float) -> float:
    if values.empty:
        raise ValueError("Volatility percentile requires rolling volatility observations.")
    return float((values <= current_value).mean())


def _safe_float(value: Any) -> float:
    if pd.isna(value):
        raise ValueError("Regime metric calculation produced NaN.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Regime metric calculation produced non-numeric value: {value}") from exc
