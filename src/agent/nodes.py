from __future__ import annotations

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
from src.monitoring.risk_engine import calculate_risk_report
from src.processing.db_manager import load_from_sqlite
from utils.logger import get_logger

logger = get_logger("AgentNodes")
AGENT_DEBUG_DIR = Path("data/agent_debug")
MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
AGENT_CONFIG_PATH = Path("configs/agent_config.yaml")

_LLM: Optional[ChatGoogleGenerativeAI] = None


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def node_validate_forecast(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    forecast = _candidate(state, "champion").get("forecast_data", {})
    fixed = _fix_quantile_crossing(forecast)
    _candidate(state, "champion")["forecast_data"] = forecast
    _candidate(state, "champion")["quantiles_fixed"] = fixed
    _append_audit(state, "validate_forecast", "PASS", f"Forecast validated; quantile crossing fixed={fixed}.")
    return state


def node_evaluate_monitoring(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    _evaluate_run(state, "champion")
    diagnostics = build_model_diagnostics(state, "champion")
    _candidate(state, "champion")["diagnostics"] = diagnostics

    health, reasons = _model_health_status(diagnostics, load_yaml_config(AGENT_CONFIG_PATH))
    state["workflow"]["model_health_status"] = health
    _candidate(state, "champion")["monitoring_summary"] = {
        "model_health_status": health,
        "reasons": reasons,
        "drift_label": diagnostics["drift"].get("final_drift_label"),
        "regime_label": diagnostics["regime"].get("final_regime_label"),
        "risk_level": diagnostics["risk"].get("risk_level"),
    }
    _append_audit(state, "evaluate_monitoring", "PASS", f"Monitoring completed; health={health}.", {"reasons": reasons})
    return state


def node_search_news_context(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    ticker = state.get("ticker", "UNKNOWN")
    try:
        state["news"] = _compact_news(tool_search_google_news(ticker, state.get("run_id")))
        status = state["news"].get("status", "NO_NEWS")
    except Exception as exc:
        state["news"] = {"status": "NEWS_ERROR", "found": False, "context": "NO_NEWS", "errors": [str(exc)]}
        status = "NEWS_ERROR"
        logger.warning("News search failed | ticker=%s | error=%s", ticker, exc)
    _append_audit(state, "search_news_context", "PASS", f"News context status={status}.")
    return state


def node_plan_retrain(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    diagnostics = build_model_diagnostics(state, "champion")
    improvement_config = load_yaml_config(AGENT_CONFIG_PATH)
    patch_policy = load_yaml_config("configs/config_patch_policy.yaml")
    retrain_gate = assess_retrain_need(diagnostics, improvement_config)

    improvement = {
        "should_retrain": retrain_gate["should_retrain"],
        "technical_retrain_required": retrain_gate["should_retrain"],
        "technical_retrain_reasons": retrain_gate["reasons"],
        "diagnostics": diagnostics,
        "config_patch": {},
        "reason": "Model is healthy enough; no candidate retrain requested.",
        "decision": "MONITOR",
    }

    if retrain_gate["should_retrain"]:
        prompt = build_agent_diagnosis_prompt(diagnostics, improvement_config, patch_policy)
        parsed = _ask_gemini_for_json(prompt, "plan_retrain", state)
        patch = {key: parsed.get(key) for key in ("learning_rate", "max_depth", "num_leaves", "min_child_samples") if key in parsed}
        improvement.update(
            {
                "decision": "TRAIN_CANDIDATE" if patch else "MANUAL_REVIEW",
                "config_patch": patch,
                "reason": str(parsed.get("reason", "Gemini did not provide a usable reason.")),
            }
        )

    state["improvement"] = improvement
    _append_audit(
        state,
        "plan_retrain",
        "PASS",
        improvement["reason"],
        {"should_retrain": improvement["should_retrain"], "config_patch": improvement["config_patch"]},
    )
    return state


def node_validate_or_repair_patch(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    improvement = state["improvement"]
    patch_policy = load_yaml_config("configs/config_patch_policy.yaml")
    base_config = load_yaml_config(MODEL_CONFIG_PATH)
    patch = improvement.get("config_patch", {})

    if not improvement.get("should_retrain"):
        improvement.update(
            {
                "config_patch_valid": None,
                "validated_config_patch": {},
                "config_patch_warnings": [],
                "config_patch_validation_status": "NOT_REQUIRED",
            }
        )
        _append_audit(state, "validate_config_patch", "NOT_REQUIRED", "No candidate retrain requested.")
        return state

    valid_patch, warnings, is_valid = validate_config_patch(patch, patch_policy, base_config)
    if not is_valid:
        repair_prompt = build_config_patch_repair_prompt(
            previous_plan={**patch, "reason": improvement.get("reason", "")},
            validation_warnings=warnings,
            diagnostics=improvement.get("diagnostics", {}),
            config_patch_policy=patch_policy,
            base_config=base_config,
        )
        repaired = _ask_gemini_for_json(repair_prompt, "repair_config", state)
        repaired_patch = {
            key: repaired.get(key)
            for key in ("learning_rate", "max_depth", "num_leaves", "min_child_samples")
            if key in repaired
        }
        valid_patch, warnings, is_valid = validate_config_patch(repaired_patch, patch_policy, base_config)
        if repaired_patch:
            improvement["config_patch"] = repaired_patch
            improvement["reason"] = str(repaired.get("reason", improvement.get("reason", "")))

    improvement.update(
        {
            "config_patch_valid": is_valid,
            "validated_config_patch": valid_patch if is_valid else {},
            "config_patch_warnings": warnings,
            "config_patch_validation_status": "VALID" if is_valid else "INVALID",
        }
    )
    _append_audit(
        state,
        "validate_config_patch",
        "PASS" if is_valid else "FAIL",
        "Config proposal valid." if is_valid else "Config proposal invalid.",
        {"warnings": warnings, "validated_patch": valid_patch},
    )
    return state


def node_train_candidate(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    improvement = state["improvement"]
    patch = improvement.get("validated_config_patch", {})
    if not improvement.get("config_patch_valid") or not patch:
        state["workflow"]["retrain_attempted"] = False
        _append_audit(state, "train_candidate", "SKIP", "No valid candidate config was available.")
        return state

    base_config = load_yaml_config(MODEL_CONFIG_PATH)
    model_params = {**_as_dict(base_config.get("lightgbm_params")), **patch}
    df = load_from_sqlite(f"processed_{state.get('ticker', 'UNKNOWN')}")
    if df.empty:
        raise ValueError("Processed training data is empty.")

    forecast = generate_7_day_forecast(df, model_params=model_params)
    forecast["ticker"] = state.get("ticker")
    state["challenger"] = {
        "forecast_data": forecast,
        "validation_metrics": forecast.get("validation_metrics", {}),
        "config_patch": patch,
        "model_params": model_params,
    }
    state["workflow"]["retrain_attempted"] = True
    state["workflow"]["retrain_count"] = int(state["workflow"].get("retrain_count", 0) or 0) + 1
    _append_audit(state, "train_candidate", "PASS", "Candidate model trained.", {"config_patch": patch})
    return state


def node_evaluate_candidate(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    if not _candidate(state, "challenger").get("forecast_data"):
        _append_audit(state, "evaluate_candidate", "SKIP", "No candidate forecast available.")
        return state
    forecast = _candidate(state, "challenger")["forecast_data"]
    _candidate(state, "challenger")["quantiles_fixed"] = _fix_quantile_crossing(forecast)
    _evaluate_run(state, "challenger")
    _candidate(state, "challenger")["diagnostics"] = build_model_diagnostics(state, "challenger")
    _append_audit(state, "evaluate_candidate", "PASS", "Candidate monitoring completed.")
    return state


def node_compare_metrics(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    if not _candidate(state, "challenger").get("forecast_data"):
        state["governance"] = {
            "decision": "KEEP_CURRENT_CONFIG",
            "accepted_candidate": False,
            "accepted_challenger": False,
            "final_model": "champion",
            "reason": "No candidate run was available.",
        }
        _append_audit(state, "compare_metrics", "KEEP_CURRENT_CONFIG", state["governance"]["reason"])
        return state

    current_metrics = _metric_bundle(_candidate(state, "champion"))
    candidate_metrics = _metric_bundle(_candidate(state, "challenger"))
    result = compare_model_candidates(current_metrics, candidate_metrics)
    result["final_model"] = "challenger" if result["accepted_candidate"] else "champion"
    result["accepted_challenger"] = result["accepted_candidate"]
    state["governance"] = result
    state["workflow"]["active_candidate"] = result["final_model"]
    _append_audit(state, "compare_metrics", result["decision"], result["reason"], result)
    return state


def node_save_or_reject_config(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    governance = state.get("governance", {})
    if not governance.get("accepted_candidate"):
        governance["config_saved"] = False
        _append_audit(state, "save_or_reject_config", "KEEP_CURRENT_CONFIG", governance.get("reason", "Candidate rejected."))
        return state

    config = load_yaml_config(MODEL_CONFIG_PATH)
    config["lightgbm_params"] = _candidate(state, "challenger").get("model_params", config.get("lightgbm_params", {}))
    with MODEL_CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    governance["config_saved"] = True
    _append_audit(state, "save_or_reject_config", "SAVE_CANDIDATE_CONFIG", "Candidate config saved to model_config.yaml.")
    return state


def node_generate_report(state: AgentState) -> AgentState:
    _ensure_state_defaults(state)
    final_model = _as_dict(state.get("governance")).get("final_model") or state["workflow"].get("active_candidate") or "champion"
    if final_model not in {"champion", "challenger"}:
        final_model = "champion"
    state["recommendation"] = _build_recommendation(state, final_model)
    state["workflow"]["current_phase"] = "completed"
    _append_audit(
        state,
        "final_recommendation",
        state["recommendation"].get("final_action", "MANUAL_REVIEW"),
        state["recommendation"].get("decision_rationale", ""),
    )
    return state


node_train_challenger = node_train_candidate
node_evaluate_challenger = node_evaluate_candidate
node_compare_models = node_compare_metrics


def _evaluate_run(state: AgentState, candidate_name: str) -> None:
    ticker = state.get("ticker", "UNKNOWN")
    candidate = _candidate(state, candidate_name)
    forecast = _as_dict(candidate.get("forecast_data"))
    validation = _as_dict(candidate.get("validation_metrics") or forecast.get("validation_metrics"))
    df = load_from_sqlite(f"processed_{ticker}")
    if df.empty:
        raise ValueError(f"Processed data unavailable for monitoring: processed_{ticker}.")

    regime = detect_regime(df)
    reference_df, current_df = _split_reference_current(df)
    drift = detect_drift(reference_df, current_df, validation)
    risk = calculate_risk_report(forecast, validation_metrics=validation, regime_report=regime, drift_report=drift)

    candidate.update({"validation_metrics": validation, "regime_report": regime, "drift_report": drift, "risk_report": risk})


def _model_health_status(diagnostics: dict, improvement_config: dict) -> tuple[str, list[str]]:
    rules = _as_dict(improvement_config.get("retrain")) or _as_dict(improvement_config.get("trigger_rules"))
    wf = _as_dict(diagnostics.get("walk_forward"))
    drift = _as_dict(diagnostics.get("drift"))
    risk = _as_dict(diagnostics.get("risk"))
    threshold = _safe_float(rules.get("mape_threshold"), 0.03)
    mape = _safe_float(wf.get("mape"))
    reasons: list[str] = []

    if mape is None:
        reasons.append("MAPE unavailable.")
    elif mape > threshold:
        reasons.append(f"MAPE {mape:.4f} > threshold {threshold:.4f}.")
    if str(drift.get("concept_drift_level", "NONE")).upper() == "HIGH":
        reasons.append("Concept drift level is HIGH.")
    if str(risk.get("risk_level", "LOW_RISK")).upper() in {"HIGH_RISK", "EXTREME_RISK"}:
        reasons.append(f"Risk level is {risk.get('risk_level')}.")

    return ("NEEDS_REVIEW", reasons) if reasons else ("OK", ["Model passed simple health checks."])


def _build_recommendation(state: AgentState, final_model: str) -> Dict[str, Any]:
    candidate = _candidate(state, final_model)
    risk = _as_dict(candidate.get("risk_report"))
    drift = _as_dict(candidate.get("drift_report"))
    news = _as_dict(state.get("news"))
    governance = _as_dict(state.get("governance"))
    improvement = _as_dict(state.get("improvement"))
    action, reasons = _final_action(candidate, risk, drift, news, governance, improvement)
    confidence = 0.55 if action in {"HOLD", "BUY", "SELL"} else 0.45
    summary = (
        f"Final research action is {action}. Final run={final_model}; "
        f"risk={risk.get('risk_level', 'UNKNOWN')}; drift={drift.get('final_drift_label', 'UNKNOWN')}; "
        f"config_decision={governance.get('decision', 'NOT_REQUIRED')}."
    )
    return {
        "final_action": action,
        "confidence": confidence,
        "assessment_summary": summary,
        "decision_rationale": " ".join(reasons),
        "interpretation": {
            "risk": risk,
            "drift": drift,
            "regime": candidate.get("regime_report", {}),
            "news": news,
            "config_decision": governance,
        },
        "evidence_used": ["walk_forward_validation", "monitoring_labels", "forecast_risk", "news_context", "metric_comparison"],
        "final_report": f"{summary} This is research output only, intended for paper-trading review and not financial advice.",
    }


def _final_action(
    candidate: Dict[str, Any],
    risk: Dict[str, Any],
    drift: Dict[str, Any],
    news: Dict[str, Any],
    governance: Dict[str, Any],
    improvement: Dict[str, Any],
) -> tuple[str, list[str]]:
    forecast = _as_dict(candidate.get("forecast_data"))
    if not forecast.get("forecasts"):
        return "MANUAL_REVIEW", ["No valid forecast was available."]
    if improvement.get("should_retrain") and improvement.get("config_patch_valid") is False:
        return "MANUAL_REVIEW", ["Retrain was needed but Gemini config was invalid."]

    risk_level = str(risk.get("risk_level", "UNKNOWN")).upper()
    drift_level = _overall_drift_level(drift)
    news_level = str(news.get("evidence_level", "NONE")).upper()
    if risk_level == "EXTREME_RISK":
        return "MANUAL_REVIEW", ["Forecast risk is EXTREME_RISK."]
    if risk_level == "HIGH_RISK" or drift_level == "HIGH" or news_level == "HIGH":
        return "WATCH", ["High risk, drift, or news context requires watch state."]

    expected_return = _safe_float(risk.get("expected_return"), 0.0) or 0.0
    risk_reward = _safe_float(risk.get("risk_reward_ratio"), 0.0) or 0.0
    downside = _safe_float(risk.get("downside_risk_95"), 0.0) or 0.0
    if expected_return > 0.03 and risk_reward >= 1.25:
        return "BUY", ["Expected return and risk/reward are positive."]
    if expected_return < -0.04 or downside <= -0.08:
        return "SELL", ["Expected return or downside risk is materially negative."]
    return "HOLD", ["No strong directional condition passed."]


def _compact_news(raw: Dict[str, Any]) -> Dict[str, Any]:
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
        "queries": raw.get("google_news_queries", []),
        "google_news_used": bool(raw.get("google_news_used", True)),
    }


def _ask_gemini_for_json(prompt: str, phase: str, state: AgentState) -> Dict[str, Any]:
    try:
        response = _get_llm().invoke(
            [
                SystemMessage(content="Return valid JSON only. Do not reveal chain-of-thought."),
                HumanMessage(content=prompt),
            ]
        ).content
        parsed = _parse_json(response)
        if parsed:
            return parsed
        _write_agent_debug_response(state.get("ticker", "UNKNOWN"), state.get("run_id"), response, phase)
    except Exception as exc:
        logger.warning("Gemini JSON request failed | phase=%s | error=%s", phase, exc)
    return {}


def _get_llm() -> ChatGoogleGenerativeAI:
    global _LLM
    if _LLM is None:
        load_dotenv()
        _LLM = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
    return _LLM


def _fix_quantile_crossing(forecast: Dict[str, Any]) -> bool:
    fixed = False
    keys = ["q_0.025", "q_0.1", "q_0.5", "q_0.9", "q_0.975"]
    for step in forecast.get("forecasts", []) if isinstance(forecast, dict) else []:
        if not all(key in step for key in keys):
            continue
        values = [step[key] for key in keys]
        sorted_values = sorted(values)
        if values != sorted_values:
            fixed = True
            for key, value in zip(keys, sorted_values):
                step[key] = value
    return fixed


def _metric_bundle(candidate: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _as_dict(_as_dict(candidate.get("validation_metrics")).get("metrics"))
    risk = _as_dict(candidate.get("risk_report"))
    drift = _as_dict(candidate.get("drift_report"))
    return {
        "mape": _safe_float(metrics.get("mape")),
        "rmse": _safe_float(metrics.get("rmse")),
        "mae": _safe_float(metrics.get("mae")),
        "directional_accuracy": _safe_float(metrics.get("directional_accuracy")),
        "interval_95_coverage": _safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage"))),
        "pinball_loss": _safe_float(metrics.get("pinball_loss")),
        "risk_level": risk.get("risk_level"),
        "drift_level": _overall_drift_level(drift),
        "drift_label": drift.get("final_drift_label"),
    }


def _overall_drift_level(drift: Dict[str, Any]) -> str:
    levels = [
        str(drift.get("feature_drift_level", "NONE")).upper(),
        str(drift.get("target_drift_level", "NONE")).upper(),
        str(drift.get("concept_drift_level", "NONE")).upper(),
    ]
    order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    return max(levels, key=lambda level: order.get(level, 0))


def _parse_json(content: Any) -> Dict[str, Any]:
    text = _response_text(content).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _response_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content or "")


def _write_agent_debug_response(ticker: str, run_id: str | None, response: Any, phase: str) -> None:
    AGENT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d")
    debug_id = run_id or datetime.now().strftime("%H%M%S")
    (AGENT_DEBUG_DIR / f"{ticker}_{phase}_{run_date}_{debug_id}.txt").write_text(str(response), encoding="utf-8")


def _split_reference_current(df: pd.DataFrame, current_window: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= current_window * 2:
        split_idx = max(1, int(len(df) * 0.7))
        return df.iloc[:split_idx], df.iloc[split_idx:]
    return df.iloc[:-current_window], df.iloc[-current_window:]


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
    state["workflow"].setdefault("retrain_attempted", False)
    state["workflow"].setdefault("active_candidate", "champion")
    state["audit"].setdefault("trail", [])
    state["audit"].setdefault("errors", [])


def _candidate(state: AgentState, candidate_name: str) -> Dict[str, Any]:
    _ensure_state_defaults(state)
    return state.setdefault(candidate_name, {})


def _append_audit(
    state: AgentState,
    phase: str,
    status: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure_state_defaults(state)
    state["audit"]["trail"].append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "phase": phase,
            "node_execution_status": status,
            "message": message,
            "details": details or {},
        }
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
