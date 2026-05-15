from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("DriftDetector")

DRIFT_LEVELS = ("NONE", "LOW", "MEDIUM", "HIGH")
LEVEL_ORDER = {level: idx for idx, level in enumerate(DRIFT_LEVELS)}
EVIDENCE_TO_LEVEL = {
    "NONE": "NONE",
    "WEAK": "LOW",
    "MODERATE": "MEDIUM",
    "STRONG": "HIGH",
}


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    validation_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(reference_df, pd.DataFrame) or not isinstance(current_df, pd.DataFrame):
        raise TypeError("Drift detection requires pandas DataFrame inputs.")
    if reference_df.empty or current_df.empty:
        raise ValueError("Drift detection requires non-empty reference and current datasets.")

    drift_notes: List[str] = []
    feature_drift = _feature_drift(reference_df, current_df, drift_notes)
    target_drift = _target_drift(reference_df, current_df, drift_notes)
    concept_drift = _concept_drift(validation_metrics, drift_notes)

    feature_level = feature_drift["level"]
    target_level = target_drift["level"]
    concept_level = concept_drift["level"]
    overall_level = _max_level([feature_level, target_level, concept_level])
    final_drift_label = f"FEATURE_{feature_level}__TARGET_{target_level}__CONCEPT_{concept_level}"

    evidence_summary = _evidence_summary(feature_drift, target_drift, concept_drift)
    if not drift_notes:
        drift_notes.extend(evidence_summary or ["No material drift evidence detected."])

    report = {
        "feature_drift_level": feature_level,
        "target_drift_level": target_level,
        "concept_drift_level": concept_level,
        "overall_drift_level": overall_level,
        "final_drift_label": final_drift_label,
        "feature_drift_detected": feature_level != "NONE",
        "target_drift_detected": target_level != "NONE",
        "concept_drift_detected": concept_level != "NONE",
        "feature_drift": feature_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "drifted_features": feature_drift.get("drifted_features", []),
        "evidence_summary": evidence_summary,
        "drift_notes": drift_notes,
    }
    logger.info(
        "Drift report generated | label=%s | feature=%s | target=%s | concept=%s",
        final_drift_label,
        feature_level,
        target_level,
        concept_level,
    )
    return report


def _feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_notes: List[str],
) -> Dict[str, Any]:
    numeric_cols = [
        col
        for col in reference_df.columns
        if col in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[col])
    ]
    if not numeric_cols:
        drift_notes.append("Feature drift not evaluated because no shared numeric columns were available.")
        return {
            "level": "NONE",
            "detected": False,
            "features": [],
            "drifted_features": [],
            "evaluated_feature_count": 0,
        }

    features: List[Dict[str, Any]] = []
    skipped_features = 0
    for col in _priority_features(numeric_cols):
        ref = reference_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        cur = current_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref) < 20 or len(cur) < 5:
            skipped_features += 1
            continue

        ref_std = float(ref.std() or 0.0)
        mean_shift_z = abs(float(cur.mean() - ref.mean())) / ref_std if ref_std > 0 else 0.0
        std_ratio = float(cur.std() / ref_std) if ref_std > 0 else 1.0
        psi = _psi(ref, cur)

        mean_evidence = _mean_shift_evidence(mean_shift_z)
        std_evidence = _std_ratio_evidence(std_ratio)
        psi_evidence = _psi_evidence(psi)
        feature_level = _max_level(
            [
                _evidence_to_level(mean_evidence),
                _evidence_to_level(std_evidence),
                _evidence_to_level(psi_evidence),
            ]
        )

        features.append(
            {
                "feature": col,
                "mean_shift_z": round(mean_shift_z, 6),
                "mean_shift_evidence": mean_evidence,
                "std_ratio": round(std_ratio, 6),
                "std_ratio_evidence": std_evidence,
                "psi": round(psi, 6),
                "psi_evidence": psi_evidence,
                "feature_drift_level": feature_level,
            }
        )

    if not features:
        drift_notes.append("Feature drift not evaluated because shared numeric columns had insufficient samples.")

    drifted_features = [item for item in features if item["feature_drift_level"] != "NONE"]
    level = _max_level([item["feature_drift_level"] for item in features])
    return {
        "level": level,
        "detected": level != "NONE",
        "features": features,
        "drifted_features": drifted_features,
        "evaluated_feature_count": len(features),
        "skipped_feature_count": skipped_features,
    }


def _target_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_notes: List[str],
) -> Dict[str, Any]:
    if "close" not in reference_df.columns or "close" not in current_df.columns:
        drift_notes.append("Target drift not evaluated because close price is unavailable.")
        return {"level": "NONE", "detected": False, "metrics": {}}

    ref_returns = reference_df["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    cur_returns = current_df["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref_returns) < 20 or len(cur_returns) < 5:
        drift_notes.append("Target drift not evaluated because return samples are insufficient.")
        return {"level": "NONE", "detected": False, "metrics": {}}

    ref_std = float(ref_returns.std() or 0.0)
    mean_shift_z = abs(float(cur_returns.mean() - ref_returns.mean())) / ref_std if ref_std > 0 else 0.0
    volatility_ratio = float(cur_returns.std() / ref_std) if ref_std > 0 else 1.0
    mean_evidence = _target_mean_evidence(mean_shift_z)
    vol_evidence = _target_vol_evidence(volatility_ratio)
    level = _max_level([_evidence_to_level(mean_evidence), _evidence_to_level(vol_evidence)])
    return {
        "level": level,
        "detected": level != "NONE",
        "metrics": {
            "return_mean_shift_z": round(mean_shift_z, 6),
            "return_mean_shift_evidence": mean_evidence,
            "return_volatility_ratio": round(volatility_ratio, 6),
            "return_volatility_evidence": vol_evidence,
        },
    }


def _concept_drift(
    validation_metrics: Optional[Dict[str, Any]],
    drift_notes: List[str],
) -> Dict[str, Any]:
    metrics = validation_metrics.get("metrics") if isinstance(validation_metrics, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        drift_notes.append("Concept drift not evaluated because validation metrics are unavailable.")
        return {"level": "NONE", "detected": False, "metrics": {}}

    mape = _safe_float(metrics.get("mape"))
    directional_accuracy = _safe_float(metrics.get("directional_accuracy"))
    interval_95_coverage = _safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage")))

    metric_payload: Dict[str, Any] = {}
    levels: List[str] = []
    if mape is not None:
        mape_evidence = _concept_mape_evidence(mape)
        metric_payload.update({"mape": round(mape, 6), "mape_evidence": mape_evidence})
        levels.append(_evidence_to_level(mape_evidence))
    if directional_accuracy is not None:
        da_evidence = _directional_accuracy_evidence(directional_accuracy)
        metric_payload.update(
            {
                "directional_accuracy": round(directional_accuracy, 6),
                "directional_accuracy_evidence": da_evidence,
            }
        )
        levels.append(_evidence_to_level(da_evidence))
    if interval_95_coverage is not None:
        coverage_evidence = _interval_coverage_evidence(interval_95_coverage)
        metric_payload.update(
            {
                "interval_95_coverage": round(interval_95_coverage, 6),
                "interval_coverage_evidence": coverage_evidence,
            }
        )
        levels.append(_evidence_to_level(coverage_evidence))

    if not levels:
        drift_notes.append("Concept drift not evaluated because validation metrics were present but unusable.")

    level = _max_level(levels)
    return {"level": level, "detected": level != "NONE", "metrics": metric_payload}


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


def _evidence_summary(
    feature_drift: Dict[str, Any],
    target_drift: Dict[str, Any],
    concept_drift: Dict[str, Any],
) -> List[str]:
    summary: List[str] = []
    if feature_drift["detected"]:
        summary.append(
            f"Feature drift level={feature_drift['level']} across "
            f"{len(feature_drift.get('drifted_features', []))} drifted features."
        )
    if target_drift["detected"]:
        summary.append(f"Target drift level={target_drift['level']}.")
    if concept_drift["detected"]:
        summary.append(f"Concept drift level={concept_drift['level']}.")
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


def _evidence_to_level(evidence: str) -> str:
    return EVIDENCE_TO_LEVEL.get(str(evidence).upper(), "NONE")


def _max_level(levels: List[str]) -> str:
    if not levels:
        return "NONE"
    return max((str(level).upper() for level in levels), key=lambda level: LEVEL_ORDER.get(level, 0))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
