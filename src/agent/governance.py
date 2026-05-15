from __future__ import annotations

from typing import Any, Dict


def compare_model_candidates(
    current_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether the candidate config is better than the current config."""
    current_mape = _as_float(current_metrics.get("mape"))
    candidate_mape = _as_float(candidate_metrics.get("mape"))
    deltas = {
        "mape_delta": _delta(candidate_metrics.get("mape"), current_metrics.get("mape")),
        "rmse_delta": _delta(candidate_metrics.get("rmse"), current_metrics.get("rmse")),
        "directional_accuracy_delta": _delta(candidate_metrics.get("directional_accuracy"), current_metrics.get("directional_accuracy")),
        "interval_95_coverage_delta": _delta(candidate_metrics.get("interval_95_coverage"), current_metrics.get("interval_95_coverage")),
        "pinball_loss_delta": _delta(candidate_metrics.get("pinball_loss"), current_metrics.get("pinball_loss")),
    }

    accepted = False
    reason = "Candidate was not better on primary walk-forward metrics."
    if current_mape is not None and candidate_mape is not None:
        if candidate_mape < current_mape:
            accepted = True
            reason = f"Candidate MAPE improved from {current_mape:.4f} to {candidate_mape:.4f}."
        elif abs(candidate_mape - current_mape) <= 0.0005 and _tie_breaker_ok(current_metrics, candidate_metrics):
            accepted = True
            reason = "Candidate MAPE was effectively tied and secondary metrics improved."
    else:
        reason = "MAPE unavailable for current or candidate run."

    return {
        "decision": "SAVE_CANDIDATE_CONFIG" if accepted else "KEEP_CURRENT_CONFIG",
        "accepted_candidate": accepted,
        "accepted_challenger": accepted,
        "accepted": accepted,
        "reason": reason,
        "metric_deltas": deltas,
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
    }


def _tie_breaker_ok(current_metrics: Dict[str, Any], candidate_metrics: Dict[str, Any]) -> bool:
    da_delta = _delta(candidate_metrics.get("directional_accuracy"), current_metrics.get("directional_accuracy"))
    coverage_delta = _delta(candidate_metrics.get("interval_95_coverage"), current_metrics.get("interval_95_coverage"))
    pinball_delta = _delta(candidate_metrics.get("pinball_loss"), current_metrics.get("pinball_loss"))
    return (
        (da_delta is None or da_delta >= 0)
        and (coverage_delta is None or coverage_delta >= 0)
        and (pinball_delta is None or pinball_delta <= 0)
    )


def _delta(new_value: Any, old_value: Any) -> float | None:
    new_float = _as_float(new_value)
    old_float = _as_float(old_value)
    if new_float is None or old_float is None:
        return None
    return new_float - old_float


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
