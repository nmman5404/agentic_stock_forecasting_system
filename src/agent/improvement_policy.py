from __future__ import annotations

from typing import Any, Dict

from src.agent.config_patch_validator import DEFAULT_PARAMETER_RANGES


def assess_retrain_need(diagnostics: dict, improvement_config: dict | None = None) -> dict:
    """Small deterministic gate before spending an LLM/retrain call."""
    config = improvement_config or {}
    rules = config.get("retrain", {}) or config.get("trigger_rules", {})
    walk_forward = _as_dict(diagnostics.get("walk_forward"))
    drift = _as_dict(diagnostics.get("drift"))
    risk = _as_dict(diagnostics.get("risk"))

    mape = _as_float(walk_forward.get("mape"))
    threshold = _as_float(rules.get("mape_threshold"), 0.03)
    concept_level = str(drift.get("concept_drift_level", "NONE")).upper()
    risk_level = str(risk.get("risk_level", "LOW_RISK")).upper()

    reasons: list[str] = []
    if mape is not None and mape > threshold:
        reasons.append(f"MAPE {mape:.4f} > threshold {threshold:.4f}.")
    if concept_level == "HIGH":
        reasons.append("Concept drift level is HIGH.")
    if risk_level in {"HIGH_RISK", "EXTREME_RISK"}:
        reasons.append(f"Risk level is {risk_level}.")

    return {
        "should_retrain": bool(reasons),
        "should_plan_retrain": bool(reasons),
        "reasons": reasons,
        "parameter_ranges": parameter_ranges(),
    }


def should_train_challenger(diagnostics: dict, improvement_config: dict | None = None) -> dict:
    """Compatibility wrapper for older imports."""
    result = assess_retrain_need(diagnostics, improvement_config)
    return {
        "should_train": result["should_retrain"],
        "reasons": result["reasons"],
        "parameter_ranges": result["parameter_ranges"],
    }


def parameter_ranges() -> Dict[str, list[float]]:
    return {key: [bounds[0], bounds[1]] for key, bounds in DEFAULT_PARAMETER_RANGES.items()}


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
