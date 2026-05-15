from __future__ import annotations

from typing import Any, Dict, Optional


RISK_ORDER = {
    "LOW": 0,
    "LOW_RISK": 0,
    "MEDIUM": 1,
    "MEDIUM_RISK": 1,
    "HIGH": 2,
    "HIGH_RISK": 2,
    "EXTREME": 3,
    "EXTREME_RISK": 3,
}


def compare_model_candidates(
    champion_metrics: Dict[str, Any],
    challenger_metrics: Dict[str, Any],
    governance_config: Dict[str, Any],
) -> Dict[str, Any]:
    rules = governance_config.get("promotion_rules", governance_config)
    if not isinstance(rules, dict):
        rules = {}

    champion_mape = _as_float(champion_metrics.get("mape"))
    challenger_mape = _as_float(challenger_metrics.get("mape"))
    champion_da = _as_float(champion_metrics.get("directional_accuracy"))
    challenger_da = _as_float(challenger_metrics.get("directional_accuracy"))
    champion_cov95 = _as_float(champion_metrics.get("interval_95_coverage"))
    challenger_cov95 = _as_float(challenger_metrics.get("interval_95_coverage"))
    champion_pinball = _as_float(champion_metrics.get("pinball_loss"))
    challenger_pinball = _as_float(challenger_metrics.get("pinball_loss"))

    metric_deltas = {
        "mape_delta": _delta(challenger_mape, champion_mape),
        "rmse_delta": _delta(_as_float(challenger_metrics.get("rmse")), _as_float(champion_metrics.get("rmse"))),
        "directional_accuracy_delta": _delta(challenger_da, champion_da),
        "interval_95_coverage_delta": _delta(challenger_cov95, champion_cov95),
        "pinball_loss_delta_pct": _pct_delta(challenger_pinball, champion_pinball),
    }

    if champion_mape is None or challenger_mape is None:
        return _decision(
            "MANUAL_REVIEW",
            False,
            "Missing walk-forward MAPE for champion or challenger; promotion requires manual review.",
            metric_deltas,
            champion_metrics,
            challenger_metrics,
            {"mape_available": False},
        )

    max_mape_degradation = float(rules.get("max_mape_degradation", 0.002))
    min_mape_improvement = float(rules.get("min_mape_improvement_for_auto_promote", 0.001))
    max_da_drop = float(rules.get("max_directional_accuracy_drop", 0.03))
    max_cov_drop = float(rules.get("max_interval_95_coverage_drop", 0.05))
    max_pinball_increase_pct = float(rules.get("max_pinball_loss_increase_pct", 0.05))
    max_allowed_mape = _as_float(rules.get("max_allowed_mape_for_promotion"))
    min_da_for_promotion = _as_float(rules.get("min_directional_accuracy_for_promotion"))
    min_cov95_for_promotion = _as_float(rules.get("min_interval_95_coverage_for_promotion"))

    mape_delta = metric_deltas["mape_delta"]
    mape_improves = mape_delta is not None and mape_delta < 0
    mape_materially_improves = mape_delta is not None and mape_delta <= -min_mape_improvement
    mape_within_degradation = mape_delta is not None and mape_delta <= max_mape_degradation
    challenger_mape_within_threshold = _upper_bound_ok(challenger_mape, max_allowed_mape)
    da_ok = _drop_within_limit(champion_da, challenger_da, max_da_drop)
    challenger_da_meets_minimum = _lower_bound_ok(challenger_da, min_da_for_promotion)
    cov95_ok = _drop_within_limit(champion_cov95, challenger_cov95, max_cov_drop)
    challenger_cov95_meets_minimum = _lower_bound_ok(challenger_cov95, min_cov95_for_promotion)
    pinball_ok = _pinball_within_limit(champion_pinball, challenger_pinball, max_pinball_increase_pct)
    risk_ok = _risk_not_worse(
        champion_metrics.get("risk_level"),
        challenger_metrics.get("risk_level"),
        bool(rules.get("reject_if_risk_level_worsens", True)),
    )
    drift_ok = _drift_allowed(
        challenger_metrics.get("drift_severity"),
        bool(rules.get("reject_if_drift_severity_high", False)),
    )
    both_poor_guardrail = not (
        bool(rules.get("reject_if_both_poor", True))
        and _both_candidates_poor(
            champion_mape=champion_mape,
            challenger_mape=challenger_mape,
            max_mape=max_allowed_mape,
            champion_da=champion_da,
            challenger_da=challenger_da,
            min_da=min_da_for_promotion,
            champion_cov95=champion_cov95,
            challenger_cov95=challenger_cov95,
            min_cov95=min_cov95_for_promotion,
        )
    )

    gates = {
        "mape_improves": mape_improves,
        "mape_materially_improves": mape_materially_improves,
        "mape_within_allowed_degradation": mape_within_degradation,
        "challenger_mape_within_threshold": challenger_mape_within_threshold,
        "directional_accuracy_ok": da_ok,
        "challenger_directional_accuracy_meets_minimum": challenger_da_meets_minimum,
        "interval_95_coverage_ok": cov95_ok,
        "challenger_interval_95_coverage_meets_minimum": challenger_cov95_meets_minimum,
        "pinball_loss_ok": pinball_ok,
        "risk_level_ok": risk_ok,
        "drift_severity_ok": drift_ok,
        "both_poor_guardrail": both_poor_guardrail,
    }

    if (
        mape_within_degradation
        and challenger_mape_within_threshold
        and da_ok
        and challenger_da_meets_minimum
        and cov95_ok
        and challenger_cov95_meets_minimum
        and pinball_ok
        and risk_ok
        and drift_ok
        and both_poor_guardrail
    ):
        reason = (
            "Challenger passed multi-metric governance: MAPE is within promotion threshold and allowed degradation, "
            "directional accuracy and interval coverage meet promotion gates, pinball loss is acceptable, "
            "and risk/drift gates passed."
        )
        return _decision(
            "PROMOTE_CHALLENGER",
            True,
            reason.replace("MAPE", "walk-forward MAPE"),
            metric_deltas,
            champion_metrics,
            challenger_metrics,
            gates,
        )

    failed_gates = [name for name, passed in gates.items() if not passed]
    reason = "Challenger rejected because governance gates failed: " + ", ".join(failed_gates) + "."
    return _decision(
        "KEEP_CHAMPION",
        False,
        reason,
        metric_deltas,
        champion_metrics,
        challenger_metrics,
        gates,
    )


def _decision(
    decision: str,
    accepted: bool,
    reason: str,
    metric_deltas: Dict[str, Optional[float]],
    champion_metrics: Dict[str, Any],
    challenger_metrics: Dict[str, Any],
    gates: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "decision": decision,
        "accepted_challenger": accepted,
        "accepted": accepted,
        "reason": reason,
        "metric_deltas": metric_deltas,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "gates": gates,
    }


def _drop_within_limit(old_value: Optional[float], new_value: Optional[float], max_drop: float) -> bool:
    if old_value is None or new_value is None:
        return True
    return new_value >= old_value - max_drop


def _pinball_within_limit(old_value: Optional[float], new_value: Optional[float], max_increase_pct: float) -> bool:
    if old_value is None or new_value is None:
        return True
    if abs(old_value) < 1e-12:
        return new_value <= old_value
    increase_pct = (new_value - old_value) / abs(old_value)
    return increase_pct <= max_increase_pct


def _risk_not_worse(old_level: Any, new_level: Any, reject_if_worse: bool) -> bool:
    if not reject_if_worse:
        return True
    old_rank = RISK_ORDER.get(str(old_level or "").upper())
    new_rank = RISK_ORDER.get(str(new_level or "").upper())
    if old_rank is None or new_rank is None:
        return True
    return new_rank <= old_rank


def _drift_allowed(drift_severity: Any, reject_if_high: bool) -> bool:
    if not reject_if_high:
        return True
    return str(drift_severity or "").upper() != "HIGH"


def _upper_bound_ok(value: Optional[float], maximum: Optional[float]) -> bool:
    if value is None or maximum is None:
        return True
    return value <= maximum


def _lower_bound_ok(value: Optional[float], minimum: Optional[float]) -> bool:
    if value is None or minimum is None:
        return True
    return value >= minimum


def _both_candidates_poor(
    *,
    champion_mape: Optional[float],
    challenger_mape: Optional[float],
    max_mape: Optional[float],
    champion_da: Optional[float],
    challenger_da: Optional[float],
    min_da: Optional[float],
    champion_cov95: Optional[float],
    challenger_cov95: Optional[float],
    min_cov95: Optional[float],
) -> bool:
    return any(
        [
            max_mape is not None
            and champion_mape is not None
            and challenger_mape is not None
            and champion_mape > max_mape
            and challenger_mape > max_mape,
            min_da is not None
            and champion_da is not None
            and challenger_da is not None
            and champion_da < min_da
            and challenger_da < min_da,
            min_cov95 is not None
            and champion_cov95 is not None
            and challenger_cov95 is not None
            and champion_cov95 < min_cov95
            and challenger_cov95 < min_cov95,
        ]
    )


def _delta(new_value: Optional[float], old_value: Optional[float]) -> Optional[float]:
    if old_value is None or new_value is None:
        return None
    return new_value - old_value


def _pct_delta(new_value: Optional[float], old_value: Optional[float]) -> Optional[float]:
    if old_value is None or new_value is None or abs(old_value) < 1e-12:
        return None
    return (new_value - old_value) / abs(old_value)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

