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

    drift_notes: List[str] = []
    feature_drift = _feature_drift(reference_df, current_df, drift_notes)
    target_drift = _target_drift(reference_df, current_df, drift_notes)
    concept_drift = _concept_drift(validation_metrics, drift_notes)

    overall_level = _max_level([feature_drift["level"], target_drift["level"], concept_drift["level"]])
    final_label = f"FEATURE_{feature_drift['level']}__TARGET_{target_drift['level']}__CONCEPT_{concept_drift['level']}"

    summary = _evidence_summary(feature_drift, target_drift, concept_drift)
    if not drift_notes:
        drift_notes.extend(summary or ["No material drift evidence detected."])

    report = {
        "feature_drift_level": feature_drift["level"],
        "target_drift_level": target_drift["level"],
        "concept_drift_level": concept_drift["level"],
        "overall_drift_level": overall_level,
        "final_drift_label": final_label,
        "feature_drift_detected": feature_drift["level"] != "NONE",
        "target_drift_detected": target_drift["level"] != "NONE",
        "concept_drift_detected": concept_drift["level"] != "NONE",
        "feature_drift": feature_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "drifted_features": feature_drift.get("drifted_features", []),
        "evidence_summary": summary,
        "drift_notes": drift_notes,
    }
    logger.info("Drift report generated | label=%s", final_label)
    return report


def _feature_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, notes: List[str]) -> Dict[str, Any]:
    num_cols = [c for c in ref_df.columns if c in cur_df.columns and pd.api.types.is_numeric_dtype(ref_df[c])]
    if not num_cols:
        notes.append("Feature drift not evaluated (no shared numeric columns).")
        return {"level": "NONE", "detected": False, "features": [], "drifted_features": []}

    features = []
    priorities = [c for c in ["close", "volume", "daily_return", "vol_change", "ma_7", "ma_14", "volatility_7", "rsi_14", "macd", "atr_14", "roc_7", "vn30_return", "vn30f_return"] if c in num_cols]
    priorities += [c for c in num_cols if c not in priorities][:8]
    priorities = priorities[:16]

    for col in priorities:
        ref, cur = ref_df[col].dropna(), cur_df[col].dropna()
        if len(ref) < 20 or len(cur) < 5: 
            continue

        ref_std = float(ref.std() or 0.0)
        ms_z = abs(float(cur.mean() - ref.mean())) / ref_std if ref_std > 0 else 0.0
        sr = float(cur.std() / ref_std) if ref_std > 0 else 1.0
        psi_val = _psi(ref, cur)

        # Rút gọn logic if/else trực tiếp map ra LOW/MEDIUM/HIGH
        l_ms = "NONE" if ms_z < 1.0 else "LOW" if ms_z < 2.0 else "MEDIUM" if ms_z < 3.0 else "HIGH"
        l_psi = "NONE" if psi_val < 0.1 else "LOW" if psi_val < 0.2 else "MEDIUM" if psi_val < 0.35 else "HIGH"
        l_sr = "NONE" if 0.8 <= sr <= 1.2 else "LOW" if 0.6 <= sr <= 1.5 else "MEDIUM" if 0.5 <= sr <= 2.0 else "HIGH"

        f_lvl = _max_level([l_ms, l_sr, l_psi])
        features.append({
            "feature": col, "mean_shift_z": round(ms_z, 6), "std_ratio": round(sr, 6),
            "psi": round(psi_val, 6), "feature_drift_level": f_lvl
        })

    if not features:
        notes.append("Feature drift not evaluated (insufficient samples).")

    drifted = [f for f in features if f["feature_drift_level"] != "NONE"]
    level = _max_level([f["feature_drift_level"] for f in features])
    return {"level": level, "detected": level != "NONE", "features": features, "drifted_features": drifted}


def _target_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, notes: List[str]) -> Dict[str, Any]:
    if "close" not in ref_df.columns or "close" not in cur_df.columns:
        return {"level": "NONE", "detected": False, "metrics": {}}

    ref_ret, cur_ret = ref_df["close"].pct_change().dropna(), cur_df["close"].pct_change().dropna()
    if len(ref_ret) < 20 or len(cur_ret) < 5:
        return {"level": "NONE", "detected": False, "metrics": {}}

    ref_std = float(ref_ret.std() or 0.0)
    ms_z = abs(float(cur_ret.mean() - ref_ret.mean())) / ref_std if ref_std > 0 else 0.0
    vr = float(cur_ret.std() / ref_std) if ref_std > 0 else 1.0

    l_ms = "NONE" if ms_z < 0.75 else "LOW" if ms_z < 1.5 else "MEDIUM" if ms_z < 2.5 else "HIGH"
    l_vr = "NONE" if 0.8 <= vr <= 1.2 else "LOW" if 0.6 <= vr <= 1.5 else "MEDIUM" if 0.5 <= vr <= 1.8 else "HIGH"

    lvl = _max_level([l_ms, l_vr])
    return {
        "level": lvl, "detected": lvl != "NONE", 
        "metrics": {"return_mean_shift_z": round(ms_z, 6), "return_volatility_ratio": round(vr, 6)}
    }


def _concept_drift(metrics_dict: Optional[Dict[str, Any]], notes: List[str]) -> Dict[str, Any]:
    metrics = (metrics_dict or {}).get("metrics", {})
    if not metrics:
        notes.append("Concept drift not evaluated (missing metrics).")
        return {"level": "NONE", "detected": False, "metrics": {}}

    mape = safe_float(metrics.get("mape"))
    da = safe_float(metrics.get("directional_accuracy"))
    cov = safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage")))
    
    levels, payload = [], {}

    if mape is not None:
        lvl = "NONE" if mape < 0.03 else "LOW" if mape < 0.05 else "MEDIUM" if mape < 0.08 else "HIGH"
        levels.append(lvl); payload["mape"] = round(mape, 6)
    if da is not None:
        lvl = "NONE" if da >= 0.60 else "LOW" if da >= 0.55 else "MEDIUM" if da >= 0.48 else "HIGH"
        levels.append(lvl); payload["directional_accuracy"] = round(da, 6)
    if cov is not None:
        lvl = "NONE" if cov >= 0.80 else "LOW" if cov >= 0.70 else "MEDIUM" if cov >= 0.55 else "HIGH"
        levels.append(lvl); payload["interval_95_coverage"] = round(cov, 6)

    if not levels: 
        notes.append("Concept drift not evaluated (usable metrics missing).")
        
    level = _max_level(levels)
    return {"level": level, "detected": level != "NONE", "metrics": payload}


def _psi(ref: pd.Series, cur: pd.Series, buckets: int = 10) -> float:
    edges = np.unique(np.quantile(ref.to_numpy(dtype=float), np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3: 
        return 0.0
    ref_pct = np.maximum(np.histogram(ref, bins=edges)[0] / max(len(ref), 1), 1e-6)
    cur_pct = np.maximum(np.histogram(cur, bins=edges)[0] / max(len(cur), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _max_level(levels: List[str]) -> str:
    return max([str(l).upper() for l in levels] + ["NONE"], key=lambda l: LEVEL_ORDER.get(l, 0))


def _evidence_summary(f_drift: Dict, t_drift: Dict, c_drift: Dict) -> List[str]:
    sum_list = []
    if f_drift["detected"]: sum_list.append(f"Feature drift={f_drift['level']} ({len(f_drift.get('drifted_features', []))} features).")
    if t_drift["detected"]: sum_list.append(f"Target drift={t_drift['level']}.")
    if c_drift["detected"]: sum_list.append(f"Concept drift={c_drift['level']}.")
    return sum_list