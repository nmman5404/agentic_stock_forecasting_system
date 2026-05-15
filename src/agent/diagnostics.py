from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def build_model_diagnostics(state: dict, candidate: str = "champion") -> dict:
    candidate_state = _as_dict(state.get(candidate))
    forecast_data = _as_dict(candidate_state.get("forecast_data"))
    validation_report = _as_dict(candidate_state.get("validation_metrics") or forecast_data.get("validation_metrics"))
    walk_forward = _validation_summary(validation_report)
    risk = _as_dict(candidate_state.get("risk_report"))
    regime = _as_dict(candidate_state.get("regime_report"))
    drift = _as_dict(candidate_state.get("drift_report"))
    workflow = _as_dict(state.get("workflow"))
    news = _as_dict(state.get("news"))
    governance = _as_dict(state.get("governance"))
    trigger_rules = _configured_trigger_rules()

    statuses = _quality_statuses(
        wf_mape=_safe_float(walk_forward.get("mape")),
        wf_mape_threshold=_safe_float(trigger_rules.get("wf_mape_threshold"))
        or _safe_float(trigger_rules.get("mape_threshold"))
        or 0.03,
        directional_accuracy=_safe_float(walk_forward.get("directional_accuracy")),
        min_directional_accuracy=_safe_float(trigger_rules.get("min_directional_accuracy")) or 0.50,
        interval_95_coverage=_safe_float(walk_forward.get("interval_95_coverage")),
        min_interval_95_coverage=_safe_float(trigger_rules.get("min_interval_95_coverage")) or 0.85,
        drift=drift,
        risk=risk,
    )

    return {
        "ticker": state.get("ticker"),
        "run_id": state.get("run_id"),
        "candidate": candidate,
        "forecast_available": bool(forecast_data.get("forecasts")),
        "data_quality_status": workflow.get("data_quality_status", "OK"),
        "statuses": statuses,
        **statuses,
        "walk_forward": {
            "mape": _safe_float(walk_forward.get("mape")),
            "rmse": _safe_float(walk_forward.get("rmse")),
            "mae": _safe_float(walk_forward.get("mae")),
            "smape": _safe_float(walk_forward.get("smape")),
            "directional_accuracy": _safe_float(walk_forward.get("directional_accuracy")),
            "interval_80_coverage": _safe_float(walk_forward.get("interval_80_coverage")),
            "interval_95_coverage": _safe_float(walk_forward.get("interval_95_coverage")),
            "pinball_loss": _safe_float(walk_forward.get("pinball_loss")),
            "prediction_bias": _safe_float(walk_forward.get("prediction_bias")),
            "prediction_bias_pct": _safe_float(walk_forward.get("prediction_bias_pct")),
            "quantile_crossing_rate": _safe_float(walk_forward.get("quantile_crossing_rate")),
            "fold_count": validation_report.get("fold_count"),
            "evaluation_method": validation_report.get("evaluation_method", "walk_forward"),
        },
        "forecast_quality": {
            "quantiles_fixed": candidate_state.get("quantiles_fixed"),
            "expected_return": _safe_float(risk.get("expected_return")),
            "downside_risk_95": _safe_float(risk.get("downside_risk_95")),
            "upside_potential_95": _safe_float(risk.get("upside_potential_95")),
            "risk_reward_ratio": _safe_float(risk.get("risk_reward_ratio")),
        },
        "regime": regime,
        "drift": drift,
        "risk": risk,
        "news": {
            "found": news.get("found", False),
            "status": news.get("status"),
            "items_count": news.get("items_count"),
            "evidence_level": news.get("evidence_level", "NONE"),
            "shock_type": news.get("shock_type", "NO_NEWS"),
            "summary": news.get("context", "NO_NEWS"),
            "sources": news.get("sources", []),
            "errors": news.get("errors", []),
        },
        "governance_history": {
            "retry_count": workflow.get("retrain_count", 0),
            "max_retries": workflow.get("max_retries", _configured_max_retries()),
            "retrain_attempted": workflow.get("retrain_attempted", False),
            "previous_governance_decision": governance.get("decision"),
        },
        "retrain_policy": _as_dict(_as_dict(state.get("improvement")).get("retrain_policy")),
    }


def _validation_summary(validation_report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = validation_report.get("metrics") if isinstance(validation_report, dict) else {}
    if not isinstance(metrics, dict):
        return {}
    return {
        "mape": metrics.get("mape"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "smape": metrics.get("smape"),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "interval_80_coverage": metrics.get("interval_80_coverage"),
        "interval_95_coverage": metrics.get("interval_95_coverage", metrics.get("interval_coverage")),
        "pinball_loss": metrics.get("pinball_loss"),
        "prediction_bias": metrics.get("prediction_bias"),
        "prediction_bias_pct": metrics.get("prediction_bias_pct"),
        "quantile_crossing_rate": metrics.get("quantile_crossing_rate"),
    }


def _configured_trigger_rules() -> Dict[str, Any]:
    improvement_config = _load_yaml(Path("configs/improvement_config.yaml"))
    trigger_rules = _as_dict(improvement_config.get("trigger_rules"))
    if trigger_rules:
        return trigger_rules

    agent_config = _load_yaml(Path("configs/agent_config.yaml"))
    max_mape = _as_dict(agent_config.get("thresholds")).get("max_mape")
    return {"mape_threshold": max_mape}


def _configured_max_retries() -> Optional[int]:
    agent_config = _load_yaml(Path("configs/agent_config.yaml"))
    value = _as_dict(agent_config.get("thresholds")).get("max_retries")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _quality_statuses(
    *,
    wf_mape: Optional[float],
    wf_mape_threshold: float,
    directional_accuracy: Optional[float],
    min_directional_accuracy: float,
    interval_95_coverage: Optional[float],
    min_interval_95_coverage: float,
    drift: Dict[str, Any],
    risk: Dict[str, Any],
) -> Dict[str, str]:
    wf_breached = wf_mape is not None and wf_mape > wf_mape_threshold
    weak_direction = directional_accuracy is not None and directional_accuracy < min_directional_accuracy
    under_covered = interval_95_coverage is not None and interval_95_coverage < min_interval_95_coverage

    drift_status = _drift_status(drift)
    risk_clearance_status = _risk_clearance_status(risk)
    accuracy_check_status = "ACCURACY_DEGRADED" if wf_breached else "ACCURACY_OK"

    if wf_breached or (weak_direction and under_covered):
        walk_forward_reliability_status = "DEGRADED"
    elif weak_direction or under_covered:
        walk_forward_reliability_status = "WEAK"
    else:
        walk_forward_reliability_status = "RELIABLE"

    if interval_95_coverage is None:
        uncertainty_calibration_status = "UNKNOWN"
    elif under_covered:
        uncertainty_calibration_status = "UNDER_COVERED"
    else:
        uncertainty_calibration_status = "CALIBRATED"

    if risk_clearance_status == "MANUAL_REVIEW_REQUIRED" or drift_status in {"HIGH_DRIFT", "CONCEPT_DRIFT"}:
        overall_trust_status = "REQUIRES_MANUAL_REVIEW"
    elif accuracy_check_status == "ACCURACY_DEGRADED" or walk_forward_reliability_status == "DEGRADED":
        overall_trust_status = "REQUIRES_REVALIDATION"
    elif walk_forward_reliability_status == "WEAK" or drift_status != "NO_DRIFT":
        overall_trust_status = "WATCH"
    else:
        overall_trust_status = "CLEARED_FOR_RESEARCH_USE"

    return {
        "accuracy_check_status": accuracy_check_status,
        "walk_forward_reliability_status": walk_forward_reliability_status,
        "uncertainty_calibration_status": uncertainty_calibration_status,
        "drift_status": drift_status,
        "risk_clearance_status": risk_clearance_status,
        "overall_trust_status": overall_trust_status,
    }


def _drift_status(drift: Dict[str, Any]) -> str:
    if str(drift.get("severity", "")).upper() == "HIGH":
        return "HIGH_DRIFT"
    if bool(drift.get("concept_drift_detected", False)):
        return "CONCEPT_DRIFT"
    if bool(drift.get("target_drift_detected", False)):
        return "TARGET_DRIFT"
    if bool(drift.get("feature_drift_detected", False)):
        return "FEATURE_DRIFT"
    return "NO_DRIFT"


def _risk_clearance_status(risk: Dict[str, Any]) -> str:
    risk_level = str(risk.get("risk_level", "")).upper()
    if risk_level in {"EXTREME_RISK", "EXTREME"}:
        return "MANUAL_REVIEW_REQUIRED"
    if risk_level in {"HIGH_RISK", "HIGH", "MEDIUM_RISK", "MEDIUM"}:
        return "WATCH"
    return "CLEARED"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
