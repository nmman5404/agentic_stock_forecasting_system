from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("RiskEngine")


def calculate_risk_report(
    forecast_data: Dict[str, Any],
    validation_metrics: Optional[Dict[str, Any]] = None,
    regime_report: Optional[Dict[str, Any]] = None,
    drift_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure forecast risk from validated quantile forecasts.

    Monitoring context is attached for downstream reasoning, but it does not
    drive the risk label. The agent layer decides actions.
    """
    _ = validation_metrics
    forecasts = _validated_forecasts(forecast_data)
    current_price = _required_positive_float(forecast_data.get("current_price"), "current_price")

    horizon_forecast = forecasts[-1]
    expected_price = _required_positive_float(horizon_forecast.get("q_0.5"), "q_0.5")
    lower_95 = _required_positive_float(horizon_forecast.get("q_0.025"), "q_0.025")
    upper_95 = _required_positive_float(horizon_forecast.get("q_0.975"), "q_0.975")
    if lower_95 > expected_price or expected_price > upper_95:
        raise ValueError("Risk calculation requires ordered 95% forecast quantiles.")

    expected_return = (expected_price - current_price) / current_price
    downside_risk_95 = min(0.0, (lower_95 - current_price) / current_price)
    upside_potential_95 = max(0.0, (upper_95 - current_price) / current_price)
    var_95 = max(0.0, (current_price - lower_95) / current_price)
    expected_shortfall = var_95 * 1.15
    risk_reward_ratio = upside_potential_95 / abs(downside_risk_95) if downside_risk_95 < 0 else 0.0

    risk_notes: List[str] = []
    risk_level = _risk_level(var_95=var_95, expected_shortfall=expected_shortfall, risk_notes=risk_notes)
    volume_regime = None
    if regime_report:
        volume_regime = regime_report.get("volume_regime") or regime_report.get("liquidity_regime")

    report = {
        "expected_return": round(expected_return, 6),
        "expected_return_7d": round(expected_return, 6),
        "downside_risk_95": round(downside_risk_95, 6),
        "upside_potential_95": round(upside_potential_95, 6),
        "risk_reward_ratio": round(risk_reward_ratio, 4),
        "var_95": round(var_95, 6),
        "expected_shortfall": round(expected_shortfall, 6),
        "risk_level": risk_level,
        "risk_inputs": {
            "current_price": current_price,
            "expected_price": expected_price,
            "lower_95": lower_95,
            "upper_95": upper_95,
        },
        "risk_context": {
            "drift_label": drift_report.get("final_drift_label") if drift_report else None,
            "feature_drift_level": drift_report.get("feature_drift_level") if drift_report else None,
            "target_drift_level": drift_report.get("target_drift_level") if drift_report else None,
            "concept_drift_level": drift_report.get("concept_drift_level") if drift_report else None,
            "overall_drift_level": drift_report.get("overall_drift_level") if drift_report else None,
            "volatility_regime": regime_report.get("volatility_regime") if regime_report else None,
            "trend_regime": regime_report.get("trend_regime") if regime_report else None,
            "volume_regime": volume_regime,
        },
        "risk_notes": risk_notes,
    }
    logger.info(
        "Risk report generated | expected_return=%.4f | risk_level=%s | var_95=%.4f",
        expected_return,
        risk_level,
        var_95,
    )
    return report


def _validated_forecasts(forecast_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(forecast_data, dict):
        raise TypeError("Risk calculation requires forecast_data as a dict.")
    forecasts = forecast_data.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        raise ValueError("Risk calculation requires non-empty forecast quantiles.")
    if not all(isinstance(item, dict) for item in forecasts):
        raise ValueError("Risk calculation requires forecast rows as dictionaries.")
    return forecasts


def _risk_level(var_95: float, expected_shortfall: float, risk_notes: List[str]) -> str:
    if var_95 >= 0.20 or expected_shortfall >= 0.25:
        risk_notes.append("Tail loss estimate is extreme.")
        return "EXTREME_RISK"
    if var_95 >= 0.12 or expected_shortfall >= 0.15:
        risk_notes.append("Tail loss estimate is elevated.")
        return "HIGH_RISK"
    if var_95 >= 0.07:
        risk_notes.append("Tail loss estimate is moderate.")
        return "MEDIUM_RISK"
    risk_notes.append("Tail risk is within normal monitoring range.")
    return "LOW_RISK"


def _required_positive_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Risk calculation requires {field_name}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Risk calculation requires numeric {field_name}.") from exc
    if numeric <= 0:
        raise ValueError(f"Risk calculation requires positive {field_name}.")
    return numeric
