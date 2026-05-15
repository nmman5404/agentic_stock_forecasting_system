from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("DriftDetector")

EVIDENCE_SCORE = {
    "NONE": 0,
    "WEAK": 1,
    "MODERATE": 2,
    "STRONG": 3,
}


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    validation_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if reference_df.empty or current_df.empty:
        return _empty_report("Insufficient data for drift detection.")

    feature_drift = _feature_drift(reference_df, current_df)
    target_drift = _target_drift(reference_df, current_df)
    concept_drift = _concept_drift(validation_metrics)

    feature_score = int(feature_drift["score"])
    target_score = int(target_drift["score"])
    concept_score = int(concept_drift["score"])
    total_score = feature_score + target_score + concept_score

    severity = _severity(total_score)
    recommended_action = _recommended_action(severity)
    evidence_summary = _evidence_summary(feature_drift, target_drift, concept_drift)
    drift_notes = evidence_summary or ["No material drift evidence detected."]

    report = {
        "feature_drift_detected": feature_score > 0,
        "target_drift_detected": target_score > 0,
        "concept_drift_detected": concept_score > 0,
        "feature_drift": feature_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "feature_score": feature_score,
        "target_score": target_score,
        "concept_score": concept_score,
        "total_score": total_score,
        "severity": severity,
        "recommended_action": recommended_action,
        "evidence_summary": evidence_summary,
        "drift_notes": drift_notes,
        "drifted_features": feature_drift["features"],
    }
    logger.info(
        "Drift report generated | severity=%s | total_score=%s | feature=%s | target=%s | concept=%s",
        severity,
        total_score,
        feature_score,
        target_score,
        concept_score,
    )
    return report


def _empty_report(reason: str) -> Dict[str, Any]:
    return {
        "feature_drift_detected": False,
        "target_drift_detected": False,
        "concept_drift_detected": False,
        "feature_drift": {"detected": False, "score": 0, "features": []},
        "target_drift": {"detected": False, "score": 0, "metrics": {}},
        "concept_drift": {"detected": False, "score": 0, "metrics": {}},
        "feature_score": 0,
        "target_score": 0,
        "concept_score": 0,
        "total_score": 0,
        "severity": "LOW",
        "recommended_action": "MONITOR",
        "evidence_summary": [],
        "drift_notes": [reason],
        "drifted_features": [],
    }


def _feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = [
        col
        for col in reference_df.columns
        if col in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[col])
    ]

    features: List[Dict[str, Any]] = []
    for col in _priority_features(numeric_cols):
        ref = reference_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        cur = current_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref) < 20 or len(cur) < 5:
            continue

        ref_std = float(ref.std() or 0.0)
        mean_shift_z = abs(float(cur.mean() - ref.mean())) / ref_std if ref_std > 0 else 0.0
        std_ratio = float(cur.std() / ref_std) if ref_std > 0 else 1.0
        psi = _psi(ref, cur)

        mean_evidence = _mean_shift_evidence(mean_shift_z)
        std_evidence = _std_ratio_evidence(std_ratio)
        psi_evidence = _psi_evidence(psi)
        feature_score = (
            EVIDENCE_SCORE[mean_evidence]
            + EVIDENCE_SCORE[std_evidence]
            + EVIDENCE_SCORE[psi_evidence]
        )

        if feature_score > 0:
            features.append(
                {
                    "feature": col,
                    "mean_shift_z": round(mean_shift_z, 6),
                    "mean_shift_evidence": mean_evidence,
                    "std_ratio": round(std_ratio, 6),
                    "std_ratio_evidence": std_evidence,
                    "psi": round(psi, 6),
                    "psi_evidence": psi_evidence,
                    "feature_score": feature_score,
                }
            )

    score = sum(item["feature_score"] for item in features)
    return {"detected": score > 0, "score": score, "features": features}


def _target_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
    if "close" not in reference_df.columns or "close" not in current_df.columns:
        return {"detected": False, "score": 0, "metrics": {}}

    ref_returns = reference_df["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    cur_returns = current_df["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref_returns) < 20 or len(cur_returns) < 5:
        return {"detected": False, "score": 0, "metrics": {}}

    ref_std = float(ref_returns.std() or 0.0)
    mean_shift_z = abs(float(cur_returns.mean() - ref_returns.mean())) / ref_std if ref_std > 0 else 0.0
    volatility_ratio = float(cur_returns.std() / ref_std) if ref_std > 0 else 1.0
    mean_evidence = _target_mean_evidence(mean_shift_z)
    vol_evidence = _target_vol_evidence(volatility_ratio)
    score = EVIDENCE_SCORE[mean_evidence] + EVIDENCE_SCORE[vol_evidence]
    return {
        "detected": score > 0,
        "score": score,
        "metrics": {
            "return_mean_shift_z": round(mean_shift_z, 6),
            "return_mean_shift_evidence": mean_evidence,
            "return_volatility_ratio": round(volatility_ratio, 6),
            "return_volatility_evidence": vol_evidence,
        },
    }


def _concept_drift(validation_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = validation_metrics.get("metrics") if isinstance(validation_metrics, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        return {"detected": False, "score": 0, "metrics": {}}

    mape = _safe_float(metrics.get("mape"))
    directional_accuracy = _safe_float(metrics.get("directional_accuracy"), default=1.0)
    interval_95_coverage = _safe_float(
        metrics.get("interval_95_coverage", metrics.get("interval_coverage")),
        default=1.0,
    )
    mape_evidence = _concept_mape_evidence(mape)
    da_evidence = _directional_accuracy_evidence(directional_accuracy)
    coverage_evidence = _interval_coverage_evidence(interval_95_coverage)
    score = (
        EVIDENCE_SCORE[mape_evidence]
        + EVIDENCE_SCORE[da_evidence]
        + EVIDENCE_SCORE[coverage_evidence]
    )
    return {
        "detected": score > 0,
        "score": score,
        "metrics": {
            "mape": round(mape, 6),
            "mape_evidence": mape_evidence,
            "directional_accuracy": round(directional_accuracy, 6),
            "directional_accuracy_evidence": da_evidence,
            "interval_95_coverage": round(interval_95_coverage, 6),
            "interval_coverage_evidence": coverage_evidence,
        },
    }


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


def _mean_shift_evidence(value: float) -> str:
    if value < 1.0:
        return "NONE"
    if value < 2.0:
        return "WEAK"
    if value < 3.0:
        return "MODERATE"
    return "STRONG"


def _std_ratio_evidence(value: float) -> str:
    if 0.8 <= value <= 1.2:
        return "NONE"
    if 0.6 <= value < 0.8 or 1.2 < value <= 1.5:
        return "WEAK"
    if 0.5 <= value < 0.6 or 1.5 < value < 2.0:
        return "MODERATE"
    return "STRONG"


def _psi_evidence(value: float) -> str:
    if value < 0.10:
        return "NONE"
    if value < 0.20:
        return "WEAK"
    if value < 0.35:
        return "MODERATE"
    return "STRONG"


def _target_mean_evidence(value: float) -> str:
    if value < 0.75:
        return "NONE"
    if value < 1.5:
        return "WEAK"
    if value < 2.5:
        return "MODERATE"
    return "STRONG"


def _target_vol_evidence(value: float) -> str:
    if 0.8 <= value <= 1.2:
        return "NONE"
    if 0.6 <= value < 0.8 or 1.2 < value <= 1.5:
        return "WEAK"
    if 0.5 <= value < 0.6 or 1.5 < value < 1.8:
        return "MODERATE"
    return "STRONG"


def _concept_mape_evidence(value: float) -> str:
    if value < 0.03:
        return "NONE"
    if value < 0.05:
        return "WEAK"
    if value < 0.08:
        return "MODERATE"
    return "STRONG"


def _directional_accuracy_evidence(value: float) -> str:
    if value >= 0.60:
        return "NONE"
    if value >= 0.55:
        return "WEAK"
    if value >= 0.48:
        return "MODERATE"
    return "STRONG"


def _interval_coverage_evidence(value: float) -> str:
    if value >= 0.80:
        return "NONE"
    if value >= 0.70:
        return "WEAK"
    if value >= 0.55:
        return "MODERATE"
    return "STRONG"


def _severity(total_score: int) -> str:
    if total_score >= 8:
        return "HIGH"
    if total_score >= 4:
        return "MEDIUM"
    return "LOW"


def _recommended_action(severity: str) -> str:
    if severity == "HIGH":
        return "MANUAL_REVIEW"
    if severity == "MEDIUM":
        return "REVALIDATE"
    return "MONITOR"


def _evidence_summary(
    feature_drift: Dict[str, Any],
    target_drift: Dict[str, Any],
    concept_drift: Dict[str, Any],
) -> List[str]:
    summary: List[str] = []
    if feature_drift["detected"]:
        summary.append(f"Feature drift score={feature_drift['score']} across {len(feature_drift['features'])} features.")
    if target_drift["detected"]:
        summary.append(f"Target drift score={target_drift['score']}.")
    if concept_drift["detected"]:
        summary.append(f"Concept drift score={concept_drift['score']}.")
    return summary


def _psi(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    edges = np.unique(np.quantile(reference.to_numpy(dtype=float), np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    eps = 1e-6
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), eps)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
