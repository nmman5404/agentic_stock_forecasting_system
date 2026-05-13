from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("DriftDetector")


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    validation_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if reference_df.empty or current_df.empty:
        return {
            "feature_drift_detected": False,
            "target_drift_detected": False,
            "concept_drift_detected": False,
            "drifted_features": [],
            "severity": "LOW",
            "recommended_action": "MONITOR",
            "drift_notes": ["Insufficient data for drift detection."],
        }

    numeric_cols = [
        col
        for col in reference_df.columns
        if col in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[col])
    ]
    priority_cols = _priority_features(numeric_cols)
    drifted_features: List[Dict[str, Any]] = []
    notes: List[str] = []

    for col in priority_cols:
        ref = reference_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        cur = current_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref) < 20 or len(cur) < 5:
            continue
        ref_mean = float(ref.mean())
        cur_mean = float(cur.mean())
        pooled_std = float(ref.std() or 0.0)
        mean_shift_z = abs(cur_mean - ref_mean) / pooled_std if pooled_std > 0 else 0.0
        std_ratio = float(cur.std() / pooled_std) if pooled_std > 0 else 1.0
        psi = _psi(ref, cur)
        if mean_shift_z >= 2.0 or std_ratio >= 2.0 or std_ratio <= 0.5 or psi >= 0.20:
            drifted_features.append(
                {
                    "feature": col,
                    "mean_shift_z": round(mean_shift_z, 3),
                    "std_ratio": round(std_ratio, 3),
                    "psi": round(psi, 3),
                }
            )

    feature_drift_detected = bool(drifted_features)
    target_drift_detected = _target_drift(reference_df, current_df, notes)
    concept_drift_detected = _concept_drift(validation_metrics, notes)
    severity = _severity(feature_drift_detected, target_drift_detected, concept_drift_detected, len(drifted_features))
    recommended_action = _recommended_action(severity, concept_drift_detected)

    if drifted_features:
        notes.append(f"Feature drift detected in {len(drifted_features)} monitored features.")
    else:
        notes.append("No material feature drift detected in monitored features.")

    report = {
        "feature_drift_detected": feature_drift_detected,
        "target_drift_detected": target_drift_detected,
        "concept_drift_detected": concept_drift_detected,
        "drifted_features": drifted_features,
        "severity": severity,
        "recommended_action": recommended_action,
        "drift_notes": notes,
    }
    logger.info(
        "Drift report generated | severity=%s | feature_drift=%s | target_drift=%s | concept_drift=%s",
        severity,
        feature_drift_detected,
        target_drift_detected,
        concept_drift_detected,
    )
    return report


def _priority_features(numeric_cols: List[str]) -> List[str]:
    preferred = [
        "close",
        "volume",
        "daily_return",
        "vol_change",
        "ma_7",
        "ma_14",
        "volatility_7",
        "rsi_14",
        "macd",
        "atr_14",
        "roc_7",
        "vn30_return",
        "vn30f_return",
    ]
    ordered = [col for col in preferred if col in numeric_cols]
    ordered.extend([col for col in numeric_cols if col not in ordered][:8])
    return ordered[:16]


def _target_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, notes: List[str]) -> bool:
    if "close" not in reference_df.columns or "close" not in current_df.columns:
        return False
    ref_returns = reference_df["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    cur_returns = current_df["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref_returns) < 20 or len(cur_returns) < 5:
        return False
    ref_std = float(ref_returns.std() or 0.0)
    mean_shift = abs(float(cur_returns.mean() - ref_returns.mean())) / ref_std if ref_std > 0 else 0.0
    vol_ratio = float(cur_returns.std() / ref_std) if ref_std > 0 else 1.0
    drifted = mean_shift >= 1.5 or vol_ratio >= 1.8 or vol_ratio <= 0.55
    notes.append(f"Target return mean_shift_z={mean_shift:.2f}; volatility_ratio={vol_ratio:.2f}.")
    return drifted


def _concept_drift(validation_metrics: Optional[Dict[str, Any]], notes: List[str]) -> bool:
    metrics = validation_metrics.get("metrics") if isinstance(validation_metrics, dict) else {}
    if not isinstance(metrics, dict):
        return False
    mape = float(metrics.get("mape", metrics.get("MAPE", 0.0)) or 0.0)
    directional_accuracy = float(metrics.get("directional_accuracy", 1.0) or 1.0)
    interval_coverage = float(metrics.get("interval_95_coverage", metrics.get("interval_coverage", 1.0)) or 1.0)
    notes.append(
        f"Validation snapshot: MAPE={mape:.4f}; directional_accuracy={directional_accuracy:.4f}; interval_95_coverage={interval_coverage:.4f}."
    )
    return mape >= 0.05 or directional_accuracy < 0.48 or interval_coverage < 0.55


def _severity(feature_drift: bool, target_drift: bool, concept_drift: bool, drifted_count: int) -> str:
    if concept_drift and (target_drift or drifted_count >= 3):
        return "HIGH"
    if drifted_count >= 5:
        return "HIGH"
    if concept_drift or target_drift or drifted_count >= 2:
        return "MEDIUM"
    return "LOW"


def _recommended_action(severity: str, concept_drift: bool) -> str:
    if severity == "HIGH":
        return "MANUAL_REVIEW"
    if severity == "MEDIUM" or concept_drift:
        return "MONITOR_AND_REVALIDATE"
    return "MONITOR"


def _psi(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(reference.to_numpy(dtype=float), quantiles))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 0.0001)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 0.0001)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
