from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("RegimeDetector")


def detect_regime(processed_df: pd.DataFrame) -> Dict[str, Any]:
    if processed_df.empty or "close" not in processed_df.columns:
        return {
            "volatility_regime": "NORMAL_VOLATILITY",
            "trend_regime": "SIDEWAYS",
            "liquidity_regime": "NORMAL_LIQUIDITY",
            "regime_confidence": 0.0,
            "regime_notes": ["Insufficient data for regime detection."],
        }

    df = processed_df.sort_index().copy()
    returns = df["daily_return"] if "daily_return" in df.columns else df["close"].pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    notes: List[str] = []

    current_vol = float(returns.tail(20).std() or 0.0)
    rolling_vol = returns.rolling(20).std().dropna()
    vol_percentile = _percentile_rank(rolling_vol, current_vol)
    volatility_regime = _volatility_label(vol_percentile)
    notes.append(f"20-day volatility percentile={vol_percentile:.2f}.")

    ma7 = df["close"].rolling(7).mean()
    ma21 = df["close"].rolling(21).mean()
    ma_gap = float((ma7.iloc[-1] - ma21.iloc[-1]) / ma21.iloc[-1]) if ma21.iloc[-1] else 0.0
    return_20d = float(df["close"].pct_change(20).iloc[-1] or 0.0)
    return_5d = float(df["close"].pct_change(5).iloc[-1] or 0.0)
    trend_regime = _trend_label(ma_gap, return_20d, return_5d)
    notes.append(f"MA7/MA21 gap={ma_gap:.4f}; 20-day return={return_20d:.4f}; 5-day return={return_5d:.4f}.")

    liquidity_regime = "NORMAL_LIQUIDITY"
    if "volume" in df.columns:
        volume = df["volume"].replace([np.inf, -np.inf], np.nan).dropna()
        volume_mean = float(volume.tail(60).mean() or 0.0)
        volume_std = float(volume.tail(60).std() or 0.0)
        volume_zscore = (float(volume.iloc[-1]) - volume_mean) / volume_std if volume_std > 0 else 0.0
        recent_volume_ratio = float(volume.tail(5).mean() / volume_mean) if volume_mean > 0 else 1.0
        if volume_zscore >= 2.5 or recent_volume_ratio >= 1.8:
            liquidity_regime = "VOLUME_SPIKE"
        elif recent_volume_ratio <= 0.45:
            liquidity_regime = "LOW_LIQUIDITY"
        notes.append(f"Volume z-score={volume_zscore:.2f}; recent volume ratio={recent_volume_ratio:.2f}.")

    confidence = _confidence(len(df), volatility_regime, trend_regime, liquidity_regime)
    report = {
        "volatility_regime": volatility_regime,
        "trend_regime": trend_regime,
        "liquidity_regime": liquidity_regime,
        "regime_confidence": confidence,
        "regime_notes": notes,
    }
    logger.info(
        "Regime report generated | volatility=%s | trend=%s | liquidity=%s | confidence=%.2f",
        volatility_regime,
        trend_regime,
        liquidity_regime,
        confidence,
    )
    return report


def _percentile_rank(values: pd.Series, current_value: float) -> float:
    if values.empty:
        return 0.5
    return float((values <= current_value).mean())


def _volatility_label(percentile: float) -> str:
    if percentile < 0.25:
        return "LOW_VOLATILITY"
    if percentile < 0.75:
        return "NORMAL_VOLATILITY"
    if percentile < 0.92:
        return "HIGH_VOLATILITY"
    return "EXTREME_VOLATILITY"


def _trend_label(ma_gap: float, return_20d: float, return_5d: float) -> str:
    if ma_gap > 0.015 and return_20d > 0.03:
        return "REVERSAL_RISK" if return_5d < -0.02 else "UPTREND"
    if ma_gap < -0.015 and return_20d < -0.03:
        return "REVERSAL_RISK" if return_5d > 0.02 else "DOWNTREND"
    if abs(ma_gap) <= 0.015 and abs(return_20d) <= 0.04:
        return "SIDEWAYS"
    return "REVERSAL_RISK"


def _confidence(row_count: int, volatility_regime: str, trend_regime: str, liquidity_regime: str) -> float:
    confidence = 0.55 if row_count >= 80 else 0.35
    if volatility_regime in {"HIGH_VOLATILITY", "EXTREME_VOLATILITY"}:
        confidence += 0.08
    if trend_regime in {"UPTREND", "DOWNTREND"}:
        confidence += 0.08
    if liquidity_regime != "NORMAL_LIQUIDITY":
        confidence += 0.05
    return round(min(confidence, 0.9), 2)
