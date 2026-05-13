from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("RiskEngine")


def calculate_risk_report(
    forecast_data: Dict[str, Any],
    validation_metrics: Optional[Dict[str, Any]],
    regime_report: Optional[Dict[str, Any]] = None,
    drift_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    forecasts = forecast_data.get("forecasts", [])
    if not forecasts:
        return _manual_review("No forecast quantiles available.")

    current_price = float(forecast_data.get("current_price") or forecasts[0].get("q_0.5", 0.0))
    if current_price <= 0:
        return _manual_review("Invalid current price for risk calculation.")

    horizon_forecast = forecasts[-1]
    expected_price = float(horizon_forecast.get("q_0.5", current_price))
    lower_95 = float(horizon_forecast.get("q_0.025", expected_price))
    upper_95 = float(horizon_forecast.get("q_0.975", expected_price))

    expected_return_7d = (expected_price - current_price) / current_price
    downside_risk_95 = min(0.0, (lower_95 - current_price) / current_price)
    upside_potential_95 = max(0.0, (upper_95 - current_price) / current_price)
    var_95 = max(0.0, (current_price - lower_95) / current_price)
    expected_shortfall = var_95 * 1.15
    risk_reward_ratio = upside_potential_95 / abs(downside_risk_95) if downside_risk_95 < 0 else 0.0

    validation_summary = _extract_validation_metrics(validation_metrics)
    risk_notes: List[str] = []
    risk_level = _risk_level(var_95, expected_shortfall, regime_report, drift_report, risk_notes)
    preliminary_signal = _preliminary_signal(
        expected_return_7d=expected_return_7d,
        risk_reward_ratio=risk_reward_ratio,
        var_95=var_95,
        risk_level=risk_level,
        drift_report=drift_report,
        regime_report=regime_report,
        validation_summary=validation_summary,
        risk_notes=risk_notes,
    )
    signal_confidence = _signal_confidence(
        preliminary_signal=preliminary_signal,
        validation_summary=validation_summary,
        risk_level=risk_level,
        drift_report=drift_report,
    )

    report = {
        "expected_return_7d": round(expected_return_7d, 6),
        "downside_risk_95": round(downside_risk_95, 6),
        "upside_potential_95": round(upside_potential_95, 6),
        "risk_reward_ratio": round(risk_reward_ratio, 4),
        "var_95": round(var_95, 6),
        "expected_shortfall": round(expected_shortfall, 6),
        "risk_level": risk_level,
        "preliminary_signal": preliminary_signal,
        "signal_confidence": signal_confidence,
        "risk_notes": risk_notes,
    }
    logger.info(
        "Risk report generated | signal=%s | confidence=%.2f | expected_return_7d=%.4f | risk_level=%s",
        preliminary_signal,
        signal_confidence,
        expected_return_7d,
        risk_level,
    )
    return report


def _manual_review(reason: str) -> Dict[str, Any]:
    logger.warning("Risk report forced to MANUAL_REVIEW | reason=%s", reason)
    return {
        "expected_return_7d": 0.0,
        "downside_risk_95": 0.0,
        "upside_potential_95": 0.0,
        "risk_reward_ratio": 0.0,
        "var_95": 0.0,
        "expected_shortfall": 0.0,
        "risk_level": "EXTREME",
        "preliminary_signal": "MANUAL_REVIEW",
        "signal_confidence": 0.0,
        "risk_notes": [reason],
    }


def _extract_validation_metrics(validation_metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not validation_metrics:
        return {}
    metrics = validation_metrics.get("metrics") if isinstance(validation_metrics, dict) else None
    if not isinstance(metrics, dict):
        return {}
    return {
        "mape": float(metrics.get("mape", metrics.get("MAPE", 0.0)) or 0.0),
        "rmse": float(metrics.get("rmse", metrics.get("RMSE", 0.0)) or 0.0),
        "directional_accuracy": float(metrics.get("directional_accuracy", 0.0) or 0.0),
        "interval_95_coverage": float(metrics.get("interval_95_coverage", metrics.get("interval_coverage", 0.0)) or 0.0),
    }


def _risk_level(
    var_95: float,
    expected_shortfall: float,
    regime_report: Optional[Dict[str, Any]],
    drift_report: Optional[Dict[str, Any]],
    risk_notes: List[str],
) -> str:
    if drift_report and drift_report.get("severity") == "HIGH":
        risk_notes.append("High drift severity detected.")
        return "EXTREME"
    if regime_report and regime_report.get("volatility_regime") == "EXTREME_VOLATILITY":
        risk_notes.append("Extreme volatility regime detected.")
        return "EXTREME"
    if var_95 >= 0.12 or expected_shortfall >= 0.15:
        risk_notes.append("Tail loss estimate is elevated.")
        return "HIGH"
    if var_95 >= 0.07:
        risk_notes.append("Tail loss estimate is moderate.")
        return "MEDIUM"
    risk_notes.append("Tail risk is within normal monitoring range.")
    return "LOW"


def _preliminary_signal(
    expected_return_7d: float,
    risk_reward_ratio: float,
    var_95: float,
    risk_level: str,
    drift_report: Optional[Dict[str, Any]],
    regime_report: Optional[Dict[str, Any]],
    validation_summary: Dict[str, float],
    risk_notes: List[str],
) -> str:
    if risk_level == "EXTREME":
        return "MANUAL_REVIEW"
    if drift_report and drift_report.get("severity") == "HIGH":
        return "MANUAL_REVIEW"
    if regime_report and regime_report.get("liquidity_regime") == "LOW_LIQUIDITY":
        risk_notes.append("Low liquidity requires manual review before acting on any signal.")
        return "MANUAL_REVIEW"
    if validation_summary.get("directional_accuracy", 1.0) < 0.48:
        risk_notes.append("Directional accuracy is weak; signal capped at WATCH/HOLD.")
        return "WATCH" if abs(expected_return_7d) > 0.03 else "HOLD"
    if expected_return_7d > 0.03 and risk_reward_ratio >= 1.25 and var_95 <= 0.10:
        return "BUY"
    if expected_return_7d < -0.04 or (expected_return_7d < -0.02 and var_95 >= 0.08):
        return "SELL"
    if abs(expected_return_7d) >= 0.025 or risk_level == "HIGH":
        return "WATCH"
    return "HOLD"


def _signal_confidence(
    preliminary_signal: str,
    validation_summary: Dict[str, float],
    risk_level: str,
    drift_report: Optional[Dict[str, Any]],
) -> float:
    confidence = 0.55
    confidence += min(max(validation_summary.get("directional_accuracy", 0.5) - 0.5, -0.2), 0.25)
    coverage = validation_summary.get("interval_95_coverage", 0.75)
    confidence += min(max(coverage - 0.75, -0.2), 0.15)
    if risk_level == "LOW":
        confidence += 0.05
    elif risk_level == "HIGH":
        confidence -= 0.10
    elif risk_level == "EXTREME":
        confidence -= 0.25
    if drift_report and drift_report.get("severity") == "MEDIUM":
        confidence -= 0.08
    if preliminary_signal in {"MANUAL_REVIEW", "WATCH"}:
        confidence = min(confidence, 0.60)
    return round(min(max(confidence, 0.0), 0.95), 2)
