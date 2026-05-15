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
    """Measure forecast risk only.

    Trading and research actions are decided downstream by the recommendation node.
    """
    forecasts = forecast_data.get("forecasts", []) if isinstance(forecast_data, dict) else []
    if not forecasts:
        return _manual_fallback("No forecast quantiles available.")

    current_price = _safe_float(forecast_data.get("current_price") or forecasts[0].get("q_0.5"))
    if current_price <= 0:
        return _manual_fallback("Invalid current price for risk calculation.")

    horizon_forecast = forecasts[-1]
    expected_price = _safe_float(horizon_forecast.get("q_0.5"), current_price)
    lower_95 = _safe_float(horizon_forecast.get("q_0.025"), expected_price)
    upper_95 = _safe_float(horizon_forecast.get("q_0.975"), expected_price)

    expected_return = (expected_price - current_price) / current_price
    downside_risk_95 = min(0.0, (lower_95 - current_price) / current_price)
    upside_potential_95 = max(0.0, (upper_95 - current_price) / current_price)
    var_95 = max(0.0, (current_price - lower_95) / current_price)
    expected_shortfall = var_95 * 1.15
    risk_reward_ratio = (
        upside_potential_95 / abs(downside_risk_95)
        if downside_risk_95 < 0
        else 0.0
    )

    risk_notes: List[str] = []
    risk_level = _risk_level(
        var_95=var_95,
        expected_shortfall=expected_shortfall,
        regime_report=regime_report,
        drift_report=drift_report,
        risk_notes=risk_notes,
    )
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
            "drift_severity": drift_report.get("severity") if drift_report else None,
            "volatility_regime": regime_report.get("volatility_regime") if regime_report else None,
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


def _manual_fallback(reason: str) -> Dict[str, Any]:
    logger.warning("Risk report forced to EXTREME_RISK | reason=%s", reason)
    return {
        "expected_return": 0.0,
        "expected_return_7d": 0.0,
        "downside_risk_95": 0.0,
        "upside_potential_95": 0.0,
        "risk_reward_ratio": 0.0,
        "var_95": 0.0,
        "expected_shortfall": 0.0,
        "risk_level": "EXTREME_RISK",
        "risk_inputs": {},
        "risk_context": {},
        "risk_notes": [reason],
    }


def _risk_level(
    var_95: float,
    expected_shortfall: float,
    regime_report: Optional[Dict[str, Any]],
    drift_report: Optional[Dict[str, Any]],
    risk_notes: List[str],
) -> str:
    volume_regime = None
    if regime_report:
        volume_regime = regime_report.get("volume_regime") or regime_report.get("liquidity_regime")

    if drift_report and drift_report.get("severity") == "HIGH":
        risk_notes.append("High drift severity detected.")
        return "EXTREME_RISK"
    if regime_report and regime_report.get("volatility_regime") == "EXTREME_VOLATILITY":
        risk_notes.append("Extreme volatility regime detected.")
        return "EXTREME_RISK"
    if volume_regime in {"LOW_VOLUME", "LOW_LIQUIDITY"}:
        risk_notes.append("Low volume regime detected.")
        return "HIGH_RISK"
    if var_95 >= 0.12 or expected_shortfall >= 0.15:
        risk_notes.append("Tail loss estimate is elevated.")
        return "HIGH_RISK"
    if var_95 >= 0.07:
        risk_notes.append("Tail loss estimate is moderate.")
        return "MEDIUM_RISK"
    risk_notes.append("Tail risk is within normal monitoring range.")
    return "LOW_RISK"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
