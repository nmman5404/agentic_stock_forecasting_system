from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.config_patch_validator import validate_config_patch
from src.agent.diagnostics import build_model_diagnostics
from src.agent.governance import compare_model_candidates
from src.agent.improvement_policy import assess_retrain_need
from src.agent.prompts import build_agent_diagnosis_prompt, build_config_patch_repair_prompt
from src.agent.state import AgentState
from src.agent.tools import tool_search_google_news
from src.modeling.predictor import generate_7_day_forecast
from src.monitoring.drift_detector import detect_drift
from src.monitoring.regime_detector import detect_regime
from src.processing.db_manager import load_from_sqlite
from src.risk.risk_engine import calculate_risk_report
from utils.logger import get_logger

logger = get_logger("AgentNodes")
AGENT_DEBUG_DIR = Path("data/agent_debug")

_LLM: Optional[ChatGoogleGenerativeAI] = None

DEFAULT_AGENT_PLAN = {
    "diagnosis": "INSUFFICIENT_EVIDENCE",
    "decision": "MANUAL_REVIEW",
    "strategy": "NO_ACTION",
    "config_patch": {},
    "reason": "Agent plan was unavailable; using conservative manual review.",
    "confidence": 0.0,
    "evidence_used": [],
}


def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def node_validate_forecast(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=validate_forecast", ticker)

    candidate = _candidate(state, "champion")
    forecast_data = _as_dict(candidate.get("forecast_data"))
    quantiles_fixed = _fix_quantile_crossing(forecast_data)
    candidate["forecast_data"] = forecast_data
    candidate["quantiles_fixed"] = quantiles_fixed
    _append_audit(
        state,
        "validate_forecast",
        "PASS",
        f"Champion forecast validated; quantile crossing fixed={quantiles_fixed}.",
    )
    logger.info("Forecast validation completed | ticker=%s | quantiles_fixed=%s", ticker, quantiles_fixed)
    return state


def node_evaluate_monitoring(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=evaluate_monitoring", ticker)

    _evaluate_candidate_monitoring(state, "champion")
    diagnostics = build_model_diagnostics(state, "champion")
    _candidate(state, "champion")["diagnostics"] = diagnostics
    _as_dict(state["workflow"])["model_health_status"] = _candidate(state, "champion").get(
        "monitoring_summary", {}
    ).get("model_health_status", "MANUAL_REVIEW")
    _append_audit(
        state,
        "evaluate_monitoring",
        "PASS",
        "Champion walk-forward, drift, regime, and risk monitoring completed.",
        {"model_health_status": state["workflow"].get("model_health_status")},
    )
    return state


def node_search_news_context(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=search_news_context", ticker)

    try:
        raw_news = tool_search_google_news(ticker, state.get("run_id"))
        state["news"] = _group_news_result(raw_news)
        status = state["news"].get("status", "UNKNOWN")
        message = f"Google News context search completed with status={status}."
    except Exception as exc:
        state["news"] = {
            "context": "NO_NEWS",
            "found": False,
            "items": [],
            "items_count": 0,
            "evidence_level": "NONE",
            "shock_type": "NO_NEWS",
            "status": "GOOGLE_NEWS_ERROR",
            "sources": [],
            "errors": [str(exc)],
            "debug_path": "",
            "keywords": [],
            "queries": [],
            "google_news_used": True,
        }
        message = f"Google News context search failed: {exc}"
        logger.warning("News context search failed | ticker=%s | error=%s", ticker, exc)

    _append_audit(state, "search_news_context", "PASS", message, {"news": state["news"]})
    return state


def node_plan_retrain(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=plan_retrain", ticker)

    improvement_config = load_yaml_config("configs/improvement_config.yaml")
    patch_policy = load_yaml_config("configs/config_patch_policy.yaml")
    diagnostics = build_model_diagnostics(state, "champion")
    retrain_policy = assess_retrain_need(diagnostics, improvement_config)

    plan = dict(DEFAULT_AGENT_PLAN)
    if not retrain_policy.get("should_plan_retrain"):
        health = state["workflow"].get("model_health_status", "OK")
        plan = {
            "diagnosis": "MODEL_OK" if health == "OK" else "RISK_OR_MONITORING_REVIEW",
            "decision": "MONITOR" if health in {"OK", "DEGRADED"} else "MANUAL_REVIEW",
            "strategy": "NO_ACTION",
            "config_patch": {},
            "reason": "Deterministic retrain policy did not require challenger planning.",
            "confidence": 0.6,
            "evidence_used": retrain_policy.get("reasons", []),
        }
    else:
        prompt = build_agent_diagnosis_prompt(diagnostics, improvement_config, patch_policy)
        try:
            response = _get_llm().invoke(
                [
                    SystemMessage(content="Return valid JSON only. Do not reveal chain-of-thought."),
                    HumanMessage(content=prompt),
                ]
            ).content
            parsed = _parse_json_response(response)
            if parsed:
                plan = _normalize_agent_plan(parsed)
            else:
                debug_path = _write_agent_debug_response(ticker, state.get("run_id"), response, "plan_retrain")
                state["audit"].setdefault("debug_paths", {})["agent_plan_raw_response"] = str(debug_path)
                logger.warning("Agent plan JSON parse failed | ticker=%s | debug_path=%s", ticker, debug_path)
        except Exception as exc:
            logger.warning("Agent retrain planning failed | ticker=%s | error=%s", ticker, exc)

    state["improvement"] = {
        "diagnostics": diagnostics,
        "retrain_policy": retrain_policy,
        "diagnosis": plan.get("diagnosis"),
        "decision": plan.get("decision"),
        "strategy": plan.get("strategy"),
        "config_patch": plan.get("config_patch", {}),
        "reason": plan.get("reason"),
        "confidence": plan.get("confidence", 0.0),
        "evidence_used": plan.get("evidence_used", []),
        "technical_retrain_required": bool(retrain_policy.get("should_plan_retrain")),
        "technical_retrain_reasons": retrain_policy.get("reasons", []),
        "technical_retrain_strategy": retrain_policy.get("recommended_strategy", "NO_ACTION"),
        "config_patch_source": "AGENT_PROPOSED" if plan.get("config_patch") else "NOT_REQUIRED",
        "repair_attempts": 0,
        "repair_history": [],
    }
    _append_audit(
        state,
        "technical_retrain_check",
        "PASS",
        "Technical retrain required=%s because %s"
        % (
            state["improvement"]["technical_retrain_required"],
            "; ".join(state["improvement"]["technical_retrain_reasons"]) or "no technical degradation trigger fired",
        ),
        {"retrain_policy": retrain_policy},
    )
    _append_audit(
        state,
        "plan_retrain",
        "PASS",
        str(plan.get("reason", "Retrain plan prepared.")),
        {"diagnosis": plan.get("diagnosis"), "decision": plan.get("decision"), "strategy": plan.get("strategy")},
    )
    return state


def node_validate_or_repair_patch(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=validate_or_repair_patch", ticker)

    improvement = state["improvement"]
    retrain_required = bool(improvement.get("technical_retrain_required"))
    decision = str(improvement.get("decision", "MONITOR")).upper()
    train_requested = decision == "TRAIN_CHALLENGER"
    if not retrain_required and not train_requested:
        improvement.update(
            {
                "validated_config_patch": {},
                "config_patch_valid": None,
                "config_patch_validation_status": "NOT_REQUIRED",
                "config_patch_warnings": [],
                "config_patch_source": "NOT_REQUIRED",
            }
        )
        _append_audit(
            state,
            "validate_config_patch",
            "NOT_REQUIRED",
            "Config patch validation skipped because no challenger training was requested.",
        )
        return state

    patch_policy = load_yaml_config("configs/config_patch_policy.yaml")
    base_config = load_yaml_config("configs/model_config.yaml")
    diagnostics = improvement.get("diagnostics") or build_model_diagnostics(state, "champion")
    patch = _as_dict(improvement.get("config_patch"))
    effective_decision = "TRAIN_CHALLENGER"

    validated_patch, warnings, is_valid = validate_config_patch(patch, patch_policy, base_config, decision=effective_decision)
    repair_history = []
    max_attempts = _max_patch_repair_attempts(patch_policy)

    attempt = 0
    while not is_valid and attempt < max_attempts:
        attempt += 1
        repaired_plan = _repair_agent_plan(
            state=state,
            previous_plan=improvement,
            validation_warnings=warnings,
            diagnostics=diagnostics,
            config_patch_policy=patch_policy,
            base_config=base_config,
            attempt=attempt,
        )
        repaired_patch = _as_dict(repaired_plan.get("config_patch"))
        validated_patch, repaired_warnings, is_valid = validate_config_patch(
            repaired_patch,
            patch_policy,
            base_config,
            decision=effective_decision,
        )
        repair_history.append(
            {
                "attempt": attempt,
                "input_patch": patch,
                "warnings": warnings,
                "repaired_patch": repaired_patch,
                "valid": is_valid,
            }
        )
        patch = repaired_patch
        warnings = repaired_warnings
        improvement.update(_normalize_agent_plan(repaired_plan))

    improvement["repair_attempts"] = attempt
    improvement["repair_history"] = repair_history

    if is_valid:
        improvement.update(
            {
                "decision": "TRAIN_CHALLENGER",
                "validated_config_patch": validated_patch,
                "config_patch_valid": True,
                "config_patch_validation_status": "VALID",
                "config_patch_warnings": warnings,
                "config_patch_source": "AGENT_PROPOSED" if attempt == 0 else "AGENT_REPAIRED",
            }
        )
        status = "PASS"
        message = "Config patch validation completed successfully."
    else:
        improvement.update(
            {
                "decision": "MANUAL_REVIEW",
                "validated_config_patch": {},
                "config_patch_valid": False,
                "config_patch_validation_status": "INVALID",
                "config_patch_warnings": warnings,
                "config_patch_source": "INVALID",
                "reason": "Technical retrain was required, but config patch validation failed.",
            }
        )
        state["governance"] = {
            "decision": "MANUAL_REVIEW",
            "accepted_challenger": False,
            "final_model": "champion",
            "reason": improvement["reason"],
        }
        status = "FAIL"
        message = improvement["reason"]

    _append_audit(
        state,
        "validate_config_patch",
        status,
        message,
        {
            "valid": is_valid,
            "warnings": warnings,
            "validated_patch": validated_patch,
            "repair_attempts": attempt,
        },
    )
    return state


def node_train_challenger(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=train_challenger", ticker)

    improvement = state["improvement"]
    patch = _as_dict(improvement.get("validated_config_patch"))
    if not improvement.get("config_patch_valid") or not patch:
        state["workflow"]["retrain_attempted"] = False
        state["governance"] = {
            "decision": "KEEP_CHAMPION",
            "accepted_challenger": False,
            "final_model": "champion",
            "reason": "No valid config patch was available for challenger training.",
        }
        _append_audit(state, "train_challenger", "SKIP", state["governance"]["reason"])
        return state

    base_config = load_yaml_config("configs/model_config.yaml")
    model_params, train_window_days = _build_challenger_model_params(base_config, patch)
    try:
        df = load_from_sqlite(f"processed_{ticker}")
        if df.empty:
            raise ValueError("Processed training data is empty.")
        if train_window_days:
            df = df.tail(train_window_days)
        forecast_data = generate_7_day_forecast(df, model_params=model_params)
        forecast_data["ticker"] = ticker
        state["challenger"] = {
            "forecast_data": forecast_data,
            "validation_metrics": forecast_data.get("validation_metrics", {}),
            "config_patch": patch,
            "train_window_days": train_window_days,
        }
        state["workflow"]["retrain_count"] = int(state["workflow"].get("retrain_count", 0)) + 1
        state["workflow"]["retrain_attempted"] = True
        _append_audit(
            state,
            "train_challenger",
            "PASS",
            "Challenger model trained successfully.",
            {"config_patch": patch, "train_window_days": train_window_days},
        )
    except Exception as exc:
        state["workflow"]["retrain_attempted"] = True
        state["audit"].setdefault("errors", []).append(str(exc))
        state["governance"] = {
            "decision": "KEEP_CHAMPION",
            "accepted_challenger": False,
            "final_model": "champion",
            "reason": f"Challenger training failed: {exc}",
        }
        _append_audit(state, "train_challenger", "FAIL", state["governance"]["reason"])
        logger.warning("Challenger training failed | ticker=%s | error=%s", ticker, exc)
    return state


def node_evaluate_challenger(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=evaluate_challenger", ticker)

    if not _candidate(state, "challenger").get("forecast_data"):
        _append_audit(state, "evaluate_challenger", "SKIP", "No challenger forecast was available.")
        return state

    forecast_data = _candidate(state, "challenger")["forecast_data"]
    _candidate(state, "challenger")["quantiles_fixed"] = _fix_quantile_crossing(forecast_data)
    _evaluate_candidate_monitoring(state, "challenger")
    _candidate(state, "challenger")["diagnostics"] = build_model_diagnostics(state, "challenger")
    _append_audit(state, "evaluate_challenger", "PASS", "Challenger monitoring completed.")
    return state


def node_compare_models(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=compare_models", ticker)

    if not _candidate(state, "challenger").get("forecast_data"):
        state["governance"] = {
            "decision": state.get("governance", {}).get("decision", "KEEP_CHAMPION"),
            "accepted_challenger": False,
            "final_model": "champion",
            "reason": state.get("governance", {}).get("reason", "No challenger candidate was available."),
        }
        _append_audit(state, "governance_review", state["governance"]["decision"], state["governance"]["reason"])
        return state

    champion_metrics = _metric_bundle(_candidate(state, "champion"))
    challenger_metrics = _metric_bundle(_candidate(state, "challenger"))
    governance_result = compare_model_candidates(
        champion_metrics,
        challenger_metrics,
        load_yaml_config("configs/governance_config.yaml"),
    )

    final_model = "challenger" if governance_result.get("accepted_challenger") else "champion"
    governance_result.update(
        {
            "final_model": final_model,
            "champion_metrics": champion_metrics,
            "challenger_metrics": challenger_metrics,
        }
    )

    if not governance_result.get("accepted_challenger") and _both_candidates_poor(champion_metrics, challenger_metrics):
        governance_result["decision"] = "MANUAL_REVIEW_AFTER_RETRAIN"
        governance_result["reason"] += " Both champion and challenger remain below reliability gates."
        final_model = "champion"
        governance_result["final_model"] = final_model

    state["governance"] = governance_result
    state["workflow"]["active_candidate"] = final_model
    _append_audit(
        state,
        "governance_review",
        governance_result.get("decision", "KEEP_CHAMPION"),
        governance_result.get("reason", "Governance review completed."),
        governance_result,
    )
    return state


def node_generate_report(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Agent node started | ticker=%s | phase=generate_report", ticker)

    final_model = _as_dict(state.get("governance")).get("final_model") or state["workflow"].get("active_candidate") or "champion"
    if final_model not in {"champion", "challenger"}:
        final_model = "champion"
    final_candidate = _candidate(state, final_model)
    recommendation = _build_recommendation(state, final_model, final_candidate)
    state["recommendation"] = recommendation
    _append_audit(
        state,
        "final_recommendation",
        recommendation.get("final_action", "MANUAL_REVIEW"),
        recommendation.get("decision_rationale", "Final recommendation generated."),
        recommendation,
    )
    state["workflow"]["current_phase"] = "completed"
    logger.info(
        "Final recommendation generated | ticker=%s | final_model=%s | action=%s | confidence=%.2f",
        ticker,
        final_model,
        recommendation.get("final_action"),
        float(recommendation.get("confidence", 0.0) or 0.0),
    )
    return state


def _evaluate_candidate_monitoring(state: AgentState, candidate_name: str) -> None:
    ticker = state.get("ticker", "UNKNOWN")
    candidate = _candidate(state, candidate_name)
    forecast_data = _as_dict(candidate.get("forecast_data"))
    validation_metrics = _as_dict(candidate.get("validation_metrics") or forecast_data.get("validation_metrics"))

    df = load_from_sqlite(f"processed_{ticker}")
    if df.empty:
        candidate["monitoring_summary"] = {
            "model_health_status": "MANUAL_REVIEW",
            "reasons": ["Processed data unavailable for monitoring."],
        }
        return

    regime_report = detect_regime(df)
    reference_df, current_df = _split_reference_current(df)
    drift_report = detect_drift(reference_df, current_df, validation_metrics)
    risk_report = calculate_risk_report(
        forecast_data,
        validation_metrics=validation_metrics,
        regime_report=regime_report,
        drift_report=drift_report,
    )
    monitoring_summary = _monitoring_summary(validation_metrics, drift_report, regime_report, risk_report)

    candidate.update(
        {
            "validation_metrics": validation_metrics,
            "regime_report": regime_report,
            "drift_report": drift_report,
            "risk_report": risk_report,
            "monitoring_summary": monitoring_summary,
        }
    )


def _monitoring_summary(
    validation_metrics: Dict[str, Any],
    drift_report: Dict[str, Any],
    regime_report: Dict[str, Any],
    risk_report: Dict[str, Any],
) -> Dict[str, Any]:
    improvement_config = load_yaml_config("configs/improvement_config.yaml")
    rules = _as_dict(improvement_config.get("trigger_rules"))
    metrics = _as_dict(validation_metrics.get("metrics"))
    mape = _safe_float(metrics.get("mape"))
    directional_accuracy = _safe_float(metrics.get("directional_accuracy"))
    interval_95_coverage = _safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage")))
    prediction_bias_pct = _safe_float(metrics.get("prediction_bias_pct"))
    mape_threshold = _safe_float(rules.get("wf_mape_threshold")) or _safe_float(rules.get("mape_threshold")) or 0.03
    min_da = _safe_float(rules.get("min_directional_accuracy")) or 0.50
    min_cov = _safe_float(rules.get("min_interval_95_coverage")) or 0.85
    max_bias = _safe_float(rules.get("max_abs_prediction_bias_pct"))

    retrain_reasons = []
    warning_reasons = []
    if mape is None:
        warning_reasons.append("Walk-forward MAPE unavailable.")
    elif mape > mape_threshold:
        retrain_reasons.append(f"Walk-forward MAPE {mape:.4f} exceeded threshold {mape_threshold:.4f}.")
    if directional_accuracy is not None and directional_accuracy < min_da:
        retrain_reasons.append(f"Directional accuracy {directional_accuracy:.4f} below minimum {min_da:.4f}.")
    if interval_95_coverage is not None and interval_95_coverage < min_cov:
        retrain_reasons.append(f"95% interval coverage {interval_95_coverage:.4f} below minimum {min_cov:.4f}.")
    if max_bias is not None and prediction_bias_pct is not None and abs(prediction_bias_pct) > max_bias:
        warning_reasons.append(f"Prediction bias pct {prediction_bias_pct:.4f} exceeded maximum {max_bias:.4f}.")
    if drift_report.get("severity") == "HIGH" and float(drift_report.get("concept_score", 0) or 0) > 0:
        retrain_reasons.append("High drift severity with concept-drift evidence.")
    if risk_report.get("risk_level") in {"EXTREME_RISK", "EXTREME"} and drift_report.get("severity") in {"MEDIUM", "HIGH"}:
        retrain_reasons.append("Extreme risk coincides with elevated drift.")

    manual_review_reasons = []
    if risk_report.get("risk_level") in {"EXTREME_RISK", "EXTREME"}:
        manual_review_reasons.append("Risk level requires manual review.")
    if drift_report.get("severity") == "HIGH" and float(drift_report.get("concept_score", 0) or 0) >= 3:
        manual_review_reasons.append("High drift severity has strong concept evidence.")

    if retrain_reasons:
        health = "RETRAIN_REQUIRED"
    elif manual_review_reasons:
        health = "MANUAL_REVIEW"
    elif warning_reasons or drift_report.get("severity") == "MEDIUM" or risk_report.get("risk_level") in {"MEDIUM_RISK", "HIGH_RISK"}:
        health = "DEGRADED"
    else:
        health = "OK"

    return {
        "model_health_status": health,
        "reasons": retrain_reasons + manual_review_reasons + warning_reasons,
        "walk_forward": {
            "mape": mape,
            "directional_accuracy": directional_accuracy,
            "interval_95_coverage": interval_95_coverage,
            "prediction_bias_pct": prediction_bias_pct,
        },
        "drift_severity": drift_report.get("severity"),
        "drift_scores": {
            "feature_score": drift_report.get("feature_score"),
            "target_score": drift_report.get("target_score"),
            "concept_score": drift_report.get("concept_score"),
            "total_score": drift_report.get("total_score"),
        },
        "regime_label": regime_report.get("final_regime_label"),
        "risk_level": risk_report.get("risk_level"),
    }


def _build_recommendation(state: AgentState, final_model: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    forecast_data = _as_dict(candidate.get("forecast_data"))
    validation_metrics = _as_dict(candidate.get("validation_metrics"))
    monitoring = _as_dict(candidate.get("monitoring_summary"))
    drift = _as_dict(candidate.get("drift_report"))
    regime = _as_dict(candidate.get("regime_report"))
    risk = _as_dict(candidate.get("risk_report"))
    news = _as_dict(state.get("news"))
    governance = _as_dict(state.get("governance"))
    improvement = _as_dict(state.get("improvement"))

    action, reasons = _final_action(
        forecast_data=forecast_data,
        validation_metrics=validation_metrics,
        monitoring=monitoring,
        drift=drift,
        risk=risk,
        news=news,
        governance=governance,
        improvement=improvement,
    )
    confidence = _recommendation_confidence(action, validation_metrics, risk, drift, news)
    evidence_used = _evidence_used(validation_metrics, drift, regime, risk, news, governance)
    summary = (
        f"Final research action is {action}. Final model={final_model}; "
        f"risk_level={risk.get('risk_level', 'UNKNOWN')}; drift_severity={drift.get('severity', 'UNKNOWN')}; "
        f"governance_decision={governance.get('decision', 'NOT_REQUIRED')}."
    )
    return {
        "final_action": action,
        "confidence": confidence,
        "assessment_summary": summary,
        "decision_rationale": " ".join(reasons),
        "interpretation": {
            "validation": _validation_summary(validation_metrics),
            "risk": risk,
            "drift": {
                "severity": drift.get("severity"),
                "concept_score": drift.get("concept_score"),
                "recommended_action": drift.get("recommended_action"),
            },
            "regime": {
                "final_regime_label": regime.get("final_regime_label"),
                "volatility_regime": regime.get("volatility_regime"),
                "trend_regime": regime.get("trend_regime"),
                "volume_regime": regime.get("volume_regime"),
            },
            "news": {
                "status": news.get("status", "NO_NEWS"),
                "evidence_level": news.get("evidence_level", "NONE"),
                "shock_type": news.get("shock_type", "NO_NEWS"),
            },
        },
        "evidence_used": evidence_used,
        "final_report": (
            f"{summary} This is research output only, intended for paper-trading review and not financial advice."
        ),
    }


def _final_action(
    *,
    forecast_data: Dict[str, Any],
    validation_metrics: Dict[str, Any],
    monitoring: Dict[str, Any],
    drift: Dict[str, Any],
    risk: Dict[str, Any],
    news: Dict[str, Any],
    governance: Dict[str, Any],
    improvement: Dict[str, Any],
) -> tuple[str, list[str]]:
    metrics = _as_dict(validation_metrics.get("metrics"))
    expected_return = _safe_float(risk.get("expected_return"), 0.0)
    risk_reward_ratio = _safe_float(risk.get("risk_reward_ratio"), 0.0)
    downside_risk = _safe_float(risk.get("downside_risk_95"), 0.0)
    risk_level = str(risk.get("risk_level", "EXTREME_RISK")).upper()
    drift_severity = str(drift.get("severity", "LOW")).upper()
    concept_score = _safe_float(drift.get("concept_score"), 0.0)
    reliability = _walk_forward_reliability(metrics)
    news_evidence = str(news.get("evidence_level", "NONE")).upper()
    reasons = []

    if not forecast_data.get("forecasts"):
        return "MANUAL_REVIEW", ["No valid forecast was available."]
    if improvement.get("config_patch_validation_status") == "INVALID":
        return "MANUAL_REVIEW", ["Technical retrain was required, but config patch validation failed."]
    if governance.get("decision") in {"MANUAL_REVIEW", "MANUAL_REVIEW_AFTER_RETRAIN"}:
        return "MANUAL_REVIEW", [str(governance.get("reason", "Governance requires manual review."))]
    if risk_level in {"EXTREME_RISK", "EXTREME"}:
        return "MANUAL_REVIEW", ["Extreme forecast risk requires manual review."]
    if drift_severity == "HIGH" and concept_score > 0:
        return "MANUAL_REVIEW", ["High drift with concept evidence requires manual review."]

    if risk_level in {"HIGH_RISK", "MEDIUM_RISK", "HIGH", "MEDIUM"}:
        reasons.append(f"Risk level {risk_level} caps the action at WATCH.")
        return "WATCH", reasons
    if drift_severity == "MEDIUM":
        return "WATCH", ["Medium drift severity caps the action at WATCH."]
    if reliability in {"WEAK", "DEGRADED"}:
        return "WATCH", [f"Walk-forward reliability is {reliability}."]
    if news_evidence in {"MEDIUM", "HIGH"}:
        return "WATCH", ["Material news evidence requires watch state before directional action."]

    if expected_return > 0.03 and risk_reward_ratio >= 1.25:
        return "BUY", ["Expected return and risk/reward are positive while risk and drift gates are acceptable."]
    if expected_return < -0.04 or downside_risk <= -0.08:
        return "SELL", ["Expected return or downside risk is materially negative."]
    return "HOLD", ["No strong directional condition passed after risk controls."]


def _recommendation_confidence(
    action: str,
    validation_metrics: Dict[str, Any],
    risk: Dict[str, Any],
    drift: Dict[str, Any],
    news: Dict[str, Any],
) -> float:
    metrics = _as_dict(validation_metrics.get("metrics"))
    confidence = 0.55
    da = _safe_float(metrics.get("directional_accuracy"), 0.5)
    coverage = _safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage")), 0.75)
    confidence += max(min(da - 0.5, 0.20), -0.20)
    confidence += max(min(coverage - 0.75, 0.15), -0.15)
    if risk.get("risk_level") == "LOW_RISK":
        confidence += 0.05
    if drift.get("severity") == "MEDIUM":
        confidence -= 0.08
    if news.get("evidence_level") in {"MEDIUM", "HIGH"}:
        confidence -= 0.05
    if action in {"WATCH", "MANUAL_REVIEW"}:
        confidence = min(confidence, 0.60)
    return round(min(max(confidence, 0.0), 0.95), 2)


def _group_news_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "context": raw.get("news_context", "NO_NEWS"),
        "found": bool(raw.get("news_found", False)),
        "items": raw.get("news_items", []),
        "items_count": int(raw.get("news_items_count", 0) or 0),
        "evidence_level": raw.get("evidence_level", "NONE"),
        "shock_type": raw.get("shock_type", "NO_NEWS"),
        "status": raw.get("news_status", "NO_NEWS"),
        "sources": raw.get("news_sources", []),
        "errors": raw.get("news_errors", []),
        "debug_path": raw.get("news_debug_path", ""),
        "keywords": raw.get("news_keywords", []),
        "queries": raw.get("google_news_queries", []),
        "google_news_used": bool(raw.get("google_news_used", True)),
        "raw_news_items_count": int(raw.get("raw_news_items_count", 0) or 0),
        "matched_news_items_count": int(raw.get("matched_news_items_count", 0) or 0),
    }


def _repair_agent_plan(
    *,
    state: AgentState,
    previous_plan: Dict[str, Any],
    validation_warnings: list[str],
    diagnostics: Dict[str, Any],
    config_patch_policy: Dict[str, Any],
    base_config: Dict[str, Any],
    attempt: int,
) -> Dict[str, Any]:
    try:
        prompt = build_config_patch_repair_prompt(
            previous_plan=previous_plan,
            validation_warnings=validation_warnings,
            diagnostics=diagnostics,
            config_patch_policy=config_patch_policy,
            base_config=base_config,
        )
        response = _get_llm().invoke(
            [
                SystemMessage(content="Return valid JSON only. Do not reveal chain-of-thought."),
                HumanMessage(content=prompt),
            ]
        ).content
        parsed = _parse_json_response(response)
        if parsed:
            return _normalize_agent_plan(parsed)
        debug_path = _write_agent_debug_response(state.get("ticker", "UNKNOWN"), state.get("run_id"), response, f"repair_patch_{attempt}")
        state["audit"].setdefault("debug_paths", {})[f"repair_patch_{attempt}_raw_response"] = str(debug_path)
    except Exception as exc:
        logger.warning("Config patch repair failed | ticker=%s | attempt=%s | error=%s", state.get("ticker"), attempt, exc)
    return dict(DEFAULT_AGENT_PLAN)


def _get_llm() -> ChatGoogleGenerativeAI:
    global _LLM
    if _LLM is None:
        load_dotenv()
        _LLM = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
    return _LLM


def _ensure_state_defaults(state: AgentState) -> None:
    state.setdefault("workflow", {})
    state.setdefault("champion", {})
    state.setdefault("challenger", {})
    state.setdefault("news", {})
    state.setdefault("improvement", {})
    state.setdefault("governance", {})
    state.setdefault("recommendation", {})
    state.setdefault("audit", {})
    state["workflow"].setdefault("current_phase", "agent_workflow")
    state["workflow"].setdefault("model_health_status", "UNKNOWN")
    state["workflow"].setdefault("retrain_count", 0)
    state["workflow"].setdefault("max_retries", _configured_max_retries())
    state["workflow"].setdefault("retrain_attempted", False)
    state["workflow"].setdefault("active_candidate", "champion")
    state["audit"].setdefault("trail", [])
    state["audit"].setdefault("debug_paths", {})
    state["audit"].setdefault("errors", [])


def _candidate(state: AgentState, candidate_name: str) -> Dict[str, Any]:
    _ensure_state_defaults(state)
    return state.setdefault(candidate_name, {})


def _fix_quantile_crossing(forecast_data: Dict[str, Any]) -> bool:
    fixed = False
    keys = ["q_0.025", "q_0.1", "q_0.5", "q_0.9", "q_0.975"]
    for step in forecast_data.get("forecasts", []) if isinstance(forecast_data, dict) else []:
        if not all(key in step for key in keys):
            continue
        values = [step[key] for key in keys]
        sorted_values = sorted(values)
        if values != sorted_values:
            fixed = True
            for key, value in zip(keys, sorted_values):
                step[key] = value
    return fixed


def _build_challenger_model_params(base_config: Dict[str, Any], patch: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[int]]:
    params = _as_dict(base_config.get("lightgbm_params")).copy()
    train_window_days = None
    for key, value in patch.items():
        if key == "train_window_days":
            train_window_days = int(value)
        else:
            params[key] = value
    return params, train_window_days


def _metric_bundle(candidate: Dict[str, Any]) -> Dict[str, Any]:
    validation = _as_dict(candidate.get("validation_metrics"))
    metrics = _as_dict(validation.get("metrics"))
    risk = _as_dict(candidate.get("risk_report"))
    drift = _as_dict(candidate.get("drift_report"))
    return {
        "mape": _safe_float(metrics.get("mape")),
        "rmse": _safe_float(metrics.get("rmse")),
        "mae": _safe_float(metrics.get("mae")),
        "directional_accuracy": _safe_float(metrics.get("directional_accuracy")),
        "interval_95_coverage": _safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage"))),
        "pinball_loss": _safe_float(metrics.get("pinball_loss")),
        "prediction_bias_pct": _safe_float(metrics.get("prediction_bias_pct")),
        "risk_level": risk.get("risk_level"),
        "drift_severity": drift.get("severity"),
    }


def _both_candidates_poor(champion: Dict[str, Any], challenger: Dict[str, Any]) -> bool:
    return (
        (_safe_float(champion.get("mape"), 0.0) > 0.03 and _safe_float(challenger.get("mape"), 0.0) > 0.03)
        or (_safe_float(champion.get("directional_accuracy"), 1.0) < 0.50 and _safe_float(challenger.get("directional_accuracy"), 1.0) < 0.50)
        or (_safe_float(champion.get("interval_95_coverage"), 1.0) < 0.80 and _safe_float(challenger.get("interval_95_coverage"), 1.0) < 0.80)
    )


def _validation_summary(validation_metrics: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _as_dict(validation_metrics.get("metrics"))
    return {
        "evaluation_method": validation_metrics.get("evaluation_method", "walk_forward"),
        "fold_count": validation_metrics.get("fold_count"),
        "mape": metrics.get("mape"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "smape": metrics.get("smape"),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "interval_80_coverage": metrics.get("interval_80_coverage"),
        "interval_95_coverage": metrics.get("interval_95_coverage", metrics.get("interval_coverage")),
        "pinball_loss": metrics.get("pinball_loss"),
        "prediction_bias_pct": metrics.get("prediction_bias_pct"),
        "quantile_crossing_rate": metrics.get("quantile_crossing_rate"),
    }


def _walk_forward_reliability(metrics: Dict[str, Any]) -> str:
    mape = _safe_float(metrics.get("mape"))
    da = _safe_float(metrics.get("directional_accuracy"))
    cov = _safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage")))
    if mape is not None and mape > 0.05:
        return "DEGRADED"
    if da is not None and da < 0.48:
        return "DEGRADED"
    if cov is not None and cov < 0.70:
        return "DEGRADED"
    if (mape is not None and mape > 0.03) or (da is not None and da < 0.55) or (cov is not None and cov < 0.80):
        return "WEAK"
    return "RELIABLE"


def _evidence_used(
    validation_metrics: Dict[str, Any],
    drift: Dict[str, Any],
    regime: Dict[str, Any],
    risk: Dict[str, Any],
    news: Dict[str, Any],
    governance: Dict[str, Any],
) -> list[str]:
    evidence = [
        "walk_forward_validation",
        "forecast_risk_metrics",
        "drift_evidence_scores",
        "regime_components",
    ]
    if news.get("google_news_used"):
        evidence.append("google_news_rss")
    if governance:
        evidence.append("governance_comparison")
    return evidence


def _normalize_agent_plan(parsed: Dict[str, Any]) -> Dict[str, Any]:
    plan = dict(DEFAULT_AGENT_PLAN)
    if not isinstance(parsed, dict):
        return plan
    plan.update(parsed)
    decision = str(plan.get("decision", "MONITOR")).upper()
    if decision in {"KEEP_MODEL", "NO_ACTION"}:
        decision = "MONITOR"
    if decision in {"RETRAIN_RECENT_WINDOW", "TRAIN_CHALLENGER"}:
        decision = "TRAIN_CHALLENGER"
    if decision in {"MANUAL_REVIEW_ONLY", "MANUAL_REVIEW"}:
        decision = "MANUAL_REVIEW"
    if decision not in {"MONITOR", "TRAIN_CHALLENGER", "MANUAL_REVIEW"}:
        decision = "MANUAL_REVIEW"
    plan["decision"] = decision
    plan["strategy"] = str(plan.get("strategy", "NO_ACTION")).upper()
    if plan["strategy"] == "MANUAL_REVIEW_ONLY":
        plan["strategy"] = "NO_ACTION"
    plan["config_patch"] = _as_dict(plan.get("config_patch"))
    plan["confidence"] = max(0.0, min(float(_safe_float(plan.get("confidence"), 0.0)), 1.0))
    evidence = plan.get("evidence_used", [])
    plan["evidence_used"] = evidence if isinstance(evidence, list) else [str(evidence)]
    plan["reason"] = str(plan.get("reason", "No reason supplied."))
    plan["diagnosis"] = str(plan.get("diagnosis", "INSUFFICIENT_EVIDENCE"))
    return plan


def _parse_json_response(content: Any) -> Dict[str, Any]:
    content = _coerce_response_text(content).strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    json_object = _extract_first_json_object(content)
    if not json_object:
        return {}
    try:
        return json.loads(json_object)
    except json.JSONDecodeError:
        return {}


def _coerce_response_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                return _coerce_response_text(parsed)
            except (ValueError, SyntaxError):
                return content
        return content
    return str(content or "")


def _extract_first_json_object(content: str) -> str:
    start = content.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return ""


def _write_agent_debug_response(ticker: str, run_id: str | None, response: Any, phase: str) -> Path:
    AGENT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d")
    debug_id = run_id or datetime.now().strftime("%H%M%S")
    path = AGENT_DEBUG_DIR / f"{ticker}_{phase}_{run_date}_{debug_id}.txt"
    path.write_text(str(response), encoding="utf-8")
    return path


def _append_audit(
    state: AgentState,
    phase: str,
    status: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure_state_defaults(state)
    state["audit"].setdefault("trail", []).append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "phase": phase,
            "node_execution_status": status,
            "message": message,
            "details": details or {},
        }
    )


def _split_reference_current(df: pd.DataFrame, current_window: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= current_window * 2:
        split_idx = max(1, int(len(df) * 0.7))
        return df.iloc[:split_idx], df.iloc[split_idx:]
    return df.iloc[:-current_window], df.iloc[-current_window:]


def _configured_max_retries() -> int:
    agent_config = load_yaml_config("configs/agent_config.yaml")
    try:
        return int(_as_dict(agent_config.get("thresholds")).get("max_retries", 1))
    except (TypeError, ValueError):
        return 1


def _max_patch_repair_attempts(policy: Dict[str, Any]) -> int:
    rules = _as_dict(policy.get("policy"))
    try:
        return int(rules.get("max_patch_repair_attempts", 1))
    except (TypeError, ValueError):
        return 1


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
