from __future__ import annotations

from typing import Any, Dict, List, Optional
from utils.helpers import safe_float

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("DriftDetector")
LEVEL_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    validation_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(reference_df, pd.DataFrame) or not isinstance(current_df, pd.DataFrame):
        raise TypeError("Drift detection requires pandas DataFrame inputs.")
    if reference_df.empty or current_df.empty:
        raise ValueError("Drift detection requires non-empty reference and current datasets.")

    f_level = _feature_drift(reference_df, current_df)
    t_level = _target_drift(reference_df, current_df)
    c_level = _concept_drift(validation_metrics)

    overall_level = _max_level([f_level, t_level, c_level])
    final_label = f"FEATURE_{f_level}__TARGET_{t_level}__CONCEPT_{c_level}"

    report = {
        "feature_drift_level": f_level,
        "target_drift_level": t_level,
        "concept_drift_level": c_level,
        "overall_drift_level": overall_level,
        "final_drift_label": final_label,
    }
    
    logger.info("Drift report generated | label=%s", final_label)
    return report


def _feature_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> str:
    num_cols = [c for c in ref_df.columns if c in cur_df.columns and pd.api.types.is_numeric_dtype(ref_df[c])]
    if not num_cols:
        return "NONE"

    FEATURES_TO_MONITOR = [
        "close", "volume", "daily_return", "vol_change", 
        "ma_7", "ma_14", "volatility_7", "rsi_14", 
        "macd", "atr_14", "roc_7", 
        "vn30_return", "vn30f_return",
        "return_lag_1", "return_lag_3", "return_lag_7"
    ]
    
    priorities = [c for c in FEATURES_TO_MONITOR if c in num_cols]

    feature_levels = []
    for col in priorities:
        ref, cur = ref_df[col].dropna(), cur_df[col].dropna()
        if len(ref) < 20 or len(cur) < 5: 
            continue

        ref_std = float(ref.std() or 0.0)
        ms_z = abs(float(cur.mean() - ref.mean())) / ref_std if ref_std > 0 else 0.0
        sr = float(cur.std() / ref_std) if ref_std > 0 else 1.0
        psi_val = _psi(ref, cur)

        l_ms = "NONE" if ms_z < 1.0 else "LOW" if ms_z < 2.0 else "MEDIUM" if ms_z < 3.0 else "HIGH"
        l_psi = "NONE" if psi_val < 0.1 else "LOW" if psi_val < 0.2 else "MEDIUM" if psi_val < 0.35 else "HIGH"
        l_sr = "NONE" if 0.8 <= sr <= 1.2 else "LOW" if 0.6 <= sr <= 1.5 else "MEDIUM" if 0.5 <= sr <= 2.0 else "HIGH"

        feature_levels.append(_max_level([l_ms, l_sr, l_psi]))

    if not feature_levels:
        return "NONE"

    return _max_level(feature_levels)


def _target_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> str:
    if "close" not in ref_df.columns or "close" not in cur_df.columns:
        return "NONE"

    ref_ret, cur_ret = ref_df["close"].pct_change().dropna(), cur_df["close"].pct_change().dropna()
    if len(ref_ret) < 20 or len(cur_ret) < 5:
        return "NONE"

    ref_std = float(ref_ret.std() or 0.0)
    ms_z = abs(float(cur_ret.mean() - ref_ret.mean())) / ref_std if ref_std > 0 else 0.0
    vr = float(cur_ret.std() / ref_std) if ref_std > 0 else 1.0

    l_ms = "NONE" if ms_z < 0.75 else "LOW" if ms_z < 1.5 else "MEDIUM" if ms_z < 2.5 else "HIGH"
    l_vr = "NONE" if 0.8 <= vr <= 1.2 else "LOW" if 0.6 <= vr <= 1.5 else "MEDIUM" if 0.5 <= vr <= 1.8 else "HIGH"

    return _max_level([l_ms, l_vr])


def _concept_drift(metrics_dict: Optional[Dict[str, Any]]) -> str:
    metrics = (metrics_dict or {}).get("metrics", {})
    if not metrics:
        return "NONE"

    mape = safe_float(metrics.get("mape"))
    da = safe_float(metrics.get("directional_accuracy"))
    cov = safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage")))
    
    levels = []
    if mape is not None:
        levels.append("NONE" if mape < 0.03 else "LOW" if mape < 0.05 else "MEDIUM" if mape < 0.08 else "HIGH")
    if da is not None:
        levels.append("NONE" if da >= 0.60 else "LOW" if da >= 0.55 else "MEDIUM" if da >= 0.48 else "HIGH")
    if cov is not None:
        levels.append("NONE" if cov >= 0.80 else "LOW" if cov >= 0.70 else "MEDIUM" if cov >= 0.55 else "HIGH")

    if not levels: 
        return "NONE"
        
    return _max_level(levels)


def _psi(ref: pd.Series, cur: pd.Series, buckets: int = 10) -> float:
    edges = np.unique(np.quantile(ref.to_numpy(dtype=float), np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3: 
        return 0.0
    ref_pct = np.maximum(np.histogram(ref, bins=edges)[0] / max(len(ref), 1), 1e-6)
    cur_pct = np.maximum(np.histogram(cur, bins=edges)[0] / max(len(cur), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _max_level(levels: List[str]) -> str:
    return max([str(l).upper() for l in levels] + ["NONE"], key=lambda l: LEVEL_ORDER.get(l, 0))