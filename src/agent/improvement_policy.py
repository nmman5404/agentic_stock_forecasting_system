from __future__ import annotations

from typing import Any, Dict, List, Optional


def assess_retrain_need(diagnostics: dict, improvement_config: dict) -> dict:
    """Deterministic retrain gate and strategy hint.

    This function does not choose model parameters. The agent proposes a patch and the
    validator decides whether it is safe to train.
    """
    trigger_rules = _as_dict(improvement_config.get("trigger_rules"))
    if bool(improvement_config.get("disable_retraining", trigger_rules.get("disable_retraining", False))):
        return _result(False, ["Retraining disabled by config."], "LOW", "NO_ACTION")

    if str(diagnostics.get("data_quality_status", "OK")).upper() == "FAIL":
        return _result(False, ["Data quality status is FAIL."], "HIGH", "NO_ACTION")
    if not diagnostics.get("forecast_available", True):
        return _result(False, ["Forecast data is missing."], "HIGH", "NO_ACTION")

    governance = _as_dict(diagnostics.get("governance_history"))
    retry_count = _as_int(governance.get("retry_count"), 0)
    max_retries = _as_int(governance.get("max_retries"), None)
    if max_retries is not None and retry_count >= max_retries:
        return _result(False, ["Maximum retrain attempts reached."], "MEDIUM", "NO_ACTION")

    walk_forward = _as_dict(diagnostics.get("walk_forward"))
    drift = _as_dict(diagnostics.get("drift"))
    risk = _as_dict(diagnostics.get("risk"))

    mape_threshold = _as_float(trigger_rules.get("wf_mape_threshold"), None)
    if mape_threshold is None:
        mape_threshold = _as_float(trigger_rules.get("mape_threshold"), 0.03)
    min_directional_accuracy = _as_float(trigger_rules.get("min_directional_accuracy"), 0.50)
    min_interval_95_coverage = _as_float(trigger_rules.get("min_interval_95_coverage"), 0.85)
    max_abs_prediction_bias_pct = _as_float(trigger_rules.get("max_abs_prediction_bias_pct"), None)

    wf_mape = _as_float(walk_forward.get("mape"), None)
    directional_accuracy = _as_float(walk_forward.get("directional_accuracy"), None)
    interval_95_coverage = _as_float(walk_forward.get("interval_95_coverage"), None)
    prediction_bias_pct = _as_float(walk_forward.get("prediction_bias_pct"), None)
    drift_severity = str(drift.get("severity", "")).upper()
    concept_score = _as_float(drift.get("concept_score"), 0.0) or 0.0
    concept_drift_detected = bool(drift.get("concept_drift_detected", False))
    risk_level = str(risk.get("risk_level", "")).upper()

    reasons: List[str] = []
    severity = "LOW"
    strategy_votes: List[str] = []
    strong_conditions = 0

    if wf_mape is not None and mape_threshold is not None and wf_mape > mape_threshold:
        reasons.append(f"Walk-forward MAPE {wf_mape:.4f} exceeded threshold {mape_threshold:.4f}.")
        severity = _max_severity(severity, "HIGH")
        strategy_votes.append("STABILIZE_MODEL")
        strong_conditions += 1

    if directional_accuracy is not None and directional_accuracy < min_directional_accuracy:
        reasons.append(
            f"Directional accuracy {directional_accuracy:.4f} below minimum {min_directional_accuracy:.4f}."
        )
        severity = _max_severity(severity, "MEDIUM")
        strategy_votes.append("RETRAIN_RECENT_WINDOW")
        strong_conditions += 1

    if interval_95_coverage is not None and interval_95_coverage < min_interval_95_coverage:
        reasons.append(
            f"95% interval coverage {interval_95_coverage:.4f} below minimum {min_interval_95_coverage:.4f}."
        )
        severity = _max_severity(severity, "MEDIUM")
        strategy_votes.append("WIDEN_INTERVAL")
        strong_conditions += 1

    if drift_severity == "HIGH" and concept_score > 0 and bool(trigger_rules.get("trigger_on_high_drift", True)):
        reasons.append("High drift severity with concept-drift evidence.")
        severity = _max_severity(severity, "HIGH")
        strategy_votes.append("RETRAIN_RECENT_WINDOW")
        strong_conditions += 1
    elif concept_drift_detected and concept_score > 0:
        reasons.append("Concept drift detected.")
        severity = _max_severity(severity, "MEDIUM")
        strategy_votes.append("RETRAIN_RECENT_WINDOW")

    if (
        prediction_bias_pct is not None
        and max_abs_prediction_bias_pct is not None
        and abs(prediction_bias_pct) > max_abs_prediction_bias_pct
    ):
        reasons.append(
            f"Absolute prediction bias pct {abs(prediction_bias_pct):.4f} exceeded maximum {max_abs_prediction_bias_pct:.4f}."
        )
        severity = _max_severity(severity, "MEDIUM")
        strategy_votes.append("STABILIZE_MODEL")

    if (
        risk_level in {"EXTREME", "EXTREME_RISK"}
        and drift_severity in {"MEDIUM", "HIGH"}
        and bool(trigger_rules.get("trigger_on_extreme_risk", True))
    ):
        reasons.append(f"Risk level is {risk_level} with drift severity {drift_severity}.")
        severity = _max_severity(severity, "HIGH")
        strategy_votes.append("RETRAIN_RECENT_WINDOW")
        strong_conditions += 1

    should_plan = bool(reasons) and (strong_conditions > 0 or severity == "HIGH")
    if not should_plan:
        return _result(False, reasons, severity, "NO_ACTION")

    return _result(True, reasons, severity, _choose_strategy(strategy_votes))


def should_train_challenger(diagnostics: dict, improvement_config: dict) -> dict:
    """Compatibility wrapper for older imports."""
    result = assess_retrain_need(diagnostics, improvement_config)
    return {
        "should_train": result["should_plan_retrain"],
        "reasons": result["reasons"],
        "severity": result["severity"],
        "recommended_strategy": result["recommended_strategy"],
        "requires_agent_patch": result["requires_agent_patch"],
    }


def _result(should_plan: bool, reasons: list[str], severity: str, recommended_strategy: str) -> dict:
    return {
        "should_plan_retrain": should_plan,
        "reasons": reasons,
        "severity": severity,
        "recommended_strategy": recommended_strategy,
        "requires_agent_patch": bool(should_plan),
    }


def _choose_strategy(votes: list[str]) -> str:
    for strategy in ("RETRAIN_RECENT_WINDOW", "WIDEN_INTERVAL", "STABILIZE_MODEL", "REDUCE_OVERFIT"):
        if strategy in votes:
            return strategy
    return "NO_ACTION"


def _max_severity(left: str, right: str) -> str:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    return right if order.get(right, 0) > order.get(left, 0) else left


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
