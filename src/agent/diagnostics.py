from __future__ import annotations

from typing import Any, Dict


def build_model_diagnostics(state: dict, candidate: str = "champion") -> dict:
    """Build the compact context Gemini needs for model improvement reasoning."""
    candidate_state = _as_dict(state.get(candidate))
    forecast = _as_dict(candidate_state.get("forecast_data"))
    validation = _as_dict(candidate_state.get("validation_metrics") or forecast.get("validation_metrics"))
    metrics = _as_dict(validation.get("metrics"))
    drift = _as_dict(candidate_state.get("drift_report"))
    regime = _as_dict(candidate_state.get("regime_report"))
    risk = _as_dict(candidate_state.get("risk_report"))
    news = _as_dict(state.get("news"))

    return {
        "ticker": state.get("ticker"),
        "run_id": state.get("run_id"),
        "candidate": candidate,
        "walk_forward": {
            "mape": _safe_float(metrics.get("mape")),
            "rmse": _safe_float(metrics.get("rmse")),
            "mae": _safe_float(metrics.get("mae")),
            "directional_accuracy": _safe_float(metrics.get("directional_accuracy")),
            "interval_95_coverage": _safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage"))),
            "pinball_loss": _safe_float(metrics.get("pinball_loss")),
            "prediction_bias_pct": _safe_float(metrics.get("prediction_bias_pct")),
        },
        "drift": {
            "feature_drift_level": drift.get("feature_drift_level"),
            "target_drift_level": drift.get("target_drift_level"),
            "concept_drift_level": drift.get("concept_drift_level"),
            "final_drift_label": drift.get("final_drift_label"),
        },
        "regime": {
            "volatility_regime": regime.get("volatility_regime"),
            "trend_regime": regime.get("trend_regime"),
            "volume_regime": regime.get("volume_regime"),
            "final_regime_label": regime.get("final_regime_label"),
        },
        "risk": {
            "risk_level": risk.get("risk_level"),
            "expected_return": _safe_float(risk.get("expected_return")),
            "downside_risk_95": _safe_float(risk.get("downside_risk_95")),
            "upside_potential_95": _safe_float(risk.get("upside_potential_95")),
            "risk_reward_ratio": _safe_float(risk.get("risk_reward_ratio")),
        },
        "news": {
            "status": news.get("status", "NO_NEWS"),
            "evidence_level": news.get("evidence_level", "NONE"),
            "shock_type": news.get("shock_type", "NO_NEWS"),
            "context": news.get("context", "NO_NEWS"),
        },
    }


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
