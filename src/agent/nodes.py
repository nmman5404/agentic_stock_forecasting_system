from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.state import AgentState
from src.agent.tools import tool_adjust_model_hyperparams, tool_search_vietstock_news
from src.modeling.predictor import generate_7_day_forecast
from src.processing.db_manager import load_from_sqlite
from src.risk.risk_engine import calculate_risk_report
from utils.logger import get_logger

logger = get_logger("AgentNodes")

_LLM: Optional[ChatGoogleGenerativeAI] = None

ALLOWED_SHOCK_TYPES = {
    "NO_NEWS",
    "TREND_SHIFT",
    "EVENT_DRIVEN",
    "BLACK_SWAN",
    "DATA_ISSUE",
    "MODEL_DEGRADATION",
}


def load_config() -> Dict[str, Any]:
    with Path("configs/agent_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_llm() -> ChatGoogleGenerativeAI:
    global _LLM
    if _LLM is None:
        load_dotenv()
        _LLM = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
    return _LLM


def node_validate(state: AgentState) -> AgentState:
    ticker = state.get("ticker", "UNKNOWN")
    _ensure_state_defaults(state)
    logger.info("Agent node started | ticker=%s | phase=validate_quantiles", ticker)

    forecast_data = state["forecast_data"]
    quantiles_fixed = False
    for step in forecast_data.get("forecasts", []):
        keys = ["q_0.025", "q_0.1", "q_0.5", "q_0.9", "q_0.975"]
        values = [step[key] for key in keys if key in step]
        if len(values) != len(keys):
            continue
        sorted_values = sorted(values)
        if values != sorted_values:
            quantiles_fixed = True
            for key, value in zip(keys, sorted_values):
                step[key] = value

    state["quantiles_fixed"] = quantiles_fixed
    _append_audit(
        state,
        "validate_quantiles",
        "PASS",
        f"Quantile crossing fixed={quantiles_fixed}.",
    )
    logger.info(
        "Agent node completed | ticker=%s | phase=validate_quantiles | quantiles_fixed=%s",
        ticker,
        quantiles_fixed,
    )
    return state


def node_evaluate(state: AgentState) -> AgentState:
    ticker = state.get("ticker", "UNKNOWN")
    _ensure_state_defaults(state)
    logger.info("Agent node started | ticker=%s | phase=evaluate_model", ticker)

    metrics = state["forecast_data"].get("metrics", {})
    validation_metrics = state["forecast_data"].get("validation_metrics", {})
    state["validation_metrics"] = validation_metrics
    threshold = load_config()["thresholds"]["max_mape"]
    mape = float(metrics.get("MAPE", metrics.get("mape", 0.0)) or 0.0)
    validation_summary = _validation_summary(validation_metrics)

    if mape > threshold:
        state["evaluation_status"] = "ABNORMAL"
        state["evaluation_reason"] = f"Holdout MAPE {mape:.4f} exceeds threshold {threshold:.4f}."
        log_level = logger.warning
    else:
        state["evaluation_status"] = "PASS"
        state["evaluation_reason"] = f"Holdout MAPE {mape:.4f} is within threshold {threshold:.4f}."
        log_level = logger.info

    _append_audit(
        state,
        "evaluate_model",
        state["evaluation_status"],
        state["evaluation_reason"],
        {
            "holdout_mape": mape,
            "walk_forward_directional_accuracy": validation_summary.get("directional_accuracy"),
            "walk_forward_interval_95_coverage": validation_summary.get("interval_95_coverage"),
        },
    )
    log_level(
        "Model evaluation completed | ticker=%s | status=%s | holdout_mape=%.4f | threshold=%.4f | wf_da=%s | wf_cov95=%s",
        ticker,
        state["evaluation_status"],
        mape,
        threshold,
        _fmt(validation_summary.get("directional_accuracy")),
        _fmt(validation_summary.get("interval_95_coverage")),
    )
    return state


def node_contextualize(state: AgentState) -> AgentState:
    ticker = state.get("ticker", "UNKNOWN")
    _ensure_state_defaults(state)
    logger.info("Agent node started | ticker=%s | phase=contextualize_news", ticker)

    news_payload = tool_search_vietstock_news(ticker)
    state["news_context"] = news_payload.get("news_context", "NO_NEWS")
    state["news_found"] = bool(news_payload.get("news_found", False))
    state["news_items_count"] = int(news_payload.get("news_items_count", 0))
    state["news_items"] = news_payload.get("news_items", [])
    state["evidence_level"] = news_payload.get("evidence_level", "NONE")
    state["news_evidence"] = {
        "news_found": state["news_found"],
        "news_items_count": state["news_items_count"],
        "evidence_level": state["evidence_level"],
        "rss_errors": news_payload.get("rss_errors", []),
    }

    if not state["news_found"]:
        assessment = {
            "assessment": "No material company-specific or macro news was detected.",
            "interpretation": (
                "Forecast degradation is likely model-related or caused by short-term market noise. "
                "There is insufficient evidence to attribute it to a specific external event."
            ),
            "shock_type": "NO_NEWS",
            "evidence_level": "NONE",
        }
    else:
        assessment = _invoke_context_committee(state)

    shock_type = assessment.get("shock_type", "NO_NEWS")
    if shock_type not in ALLOWED_SHOCK_TYPES:
        shock_type = "EVENT_DRIVEN" if state["news_found"] else "NO_NEWS"

    state["shock_type"] = shock_type
    state["evidence_level"] = assessment.get("evidence_level", state["evidence_level"])
    state["news_analysis"] = json.dumps(assessment, ensure_ascii=False)
    state["agent_assessment_summary"] = _format_assessment_text(assessment)

    _append_audit(
        state,
        "contextualize_news",
        shock_type,
        assessment.get("interpretation", ""),
        state["news_evidence"],
    )
    logger.info(
        "Agent assessment summary | ticker=%s | shock_type=%s | evidence_level=%s | news_items=%s",
        ticker,
        state["shock_type"],
        state["evidence_level"],
        state["news_items_count"],
    )
    return state


def node_improve(state: AgentState) -> AgentState:
    ticker = state.get("ticker", "UNKNOWN")
    _ensure_state_defaults(state)
    logger.info("Agent node started | ticker=%s | phase=model_governance", ticker)

    retry_count = int(state.get("retry_count", 0))
    max_retries = load_config()["thresholds"]["max_retries"]
    if retry_count >= max_retries:
        state["action_taken"] = "MAX_RETRIES_REACHED"
        state["governance_decision"] = {
            "decision": "KEEP_CHAMPION",
            "accepted": False,
            "reason": "Maximum retrain attempts reached.",
            "retry_count": retry_count,
        }
        _append_audit(state, "model_governance", "KEEP_CHAMPION", "Maximum retrain attempts reached.")
        return state

    champion_forecast = state["forecast_data"]
    config_path = Path("configs/model_config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        backup_config = yaml.safe_load(f)

    shock_type = state.get("shock_type", "NO_NEWS")
    if shock_type in {"TREND_SHIFT", "EVENT_DRIVEN"}:
        action = tool_adjust_model_hyperparams("ADAPT_SHOCK")
    elif retry_count == 0:
        action = tool_adjust_model_hyperparams("FIX_OVERFITTING")
    else:
        action = tool_adjust_model_hyperparams("FIX_UNDERFITTING")

    df = load_from_sqlite(f"processed_{ticker}")
    challenger_forecast = generate_7_day_forecast(df)
    decision = _compare_champion_challenger(champion_forecast, challenger_forecast)
    decision["config_action"] = action
    decision["retry_count"] = retry_count + 1

    if decision["accepted"]:
        state["forecast_data"] = challenger_forecast
        state["validation_metrics"] = challenger_forecast.get("validation_metrics", {})
        state["risk_report"] = calculate_risk_report(
            challenger_forecast,
            state.get("validation_metrics"),
            state.get("regime_report"),
            state.get("drift_report"),
        )
        state["signal_confidence"] = state["risk_report"].get("signal_confidence", 0.0)
        state["action_taken"] = (
            f"{action} | ACCEPT_CHALLENGER | reason={decision['reason']}"
        )
        logger.info(
            "Governance decision | ticker=%s | decision=ACCEPT_CHALLENGER | reason=%s",
            ticker,
            decision["reason"],
        )
    else:
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(backup_config, f, sort_keys=False)
        state["action_taken"] = f"{action} | KEEP_CHAMPION | reason={decision['reason']}"
        logger.warning(
            "Governance decision | ticker=%s | decision=KEEP_CHAMPION | reason=%s",
            ticker,
            decision["reason"],
        )

    state["governance_decision"] = decision
    state["retry_count"] = retry_count + 1
    _append_audit(state, "model_governance", decision["decision"], decision["reason"], decision)
    return state


def node_recommend(state: AgentState) -> AgentState:
    ticker = state.get("ticker", "UNKNOWN")
    _ensure_state_defaults(state)
    logger.info("Agent node started | ticker=%s | phase=research_signal", ticker)

    if not state.get("risk_report"):
        state["risk_report"] = calculate_risk_report(
            state["forecast_data"],
            state.get("validation_metrics") or state["forecast_data"].get("validation_metrics"),
            state.get("regime_report"),
            state.get("drift_report"),
        )

    risk_report = state["risk_report"]
    state["trading_signal"] = risk_report.get("preliminary_signal", "HOLD")
    state["signal_confidence"] = float(risk_report.get("signal_confidence", 0.0) or 0.0)

    committee_assessment = _invoke_recommendation_committee(state)
    state["committee_assessment"] = committee_assessment
    state["assessment_summary"] = committee_assessment.get(
        "assessment_summary",
        committee_assessment.get("assessment", "Committee assessment unavailable."),
    )
    state["interpretation"] = committee_assessment.get("interpretation", {})
    state["decision_rationale"] = committee_assessment.get(
        "decision_rationale",
        committee_assessment.get("final_signal_reason", "Signal follows risk engine and governance checks."),
    )
    state["final_recommendation"] = committee_assessment.get(
        "final_recommendation",
        "Research signal only. No broker execution.",
    )
    state["evidence_used"] = committee_assessment.get("evidence_used", [])
    state["agent_assessment_summary"] = state["assessment_summary"]

    _append_audit(
        state,
        "research_signal",
        state["trading_signal"],
        f"Final research signal={state['trading_signal']}; confidence={state['signal_confidence']:.2f}.",
        {"risk_level": risk_report.get("risk_level"), "risk_notes": risk_report.get("risk_notes", [])},
    )
    logger.warning(
        "Final research signal | ticker=%s | signal=%s | confidence=%.2f | risk_level=%s",
        ticker,
        state["trading_signal"],
        state["signal_confidence"],
        risk_report.get("risk_level", "N/A"),
    )
    return state


def _ensure_state_defaults(state: AgentState) -> None:
    state.setdefault("retry_count", 0)
    state.setdefault("audit_trail", [])
    state.setdefault("news_context", "NO_NEWS")
    state.setdefault("news_found", False)
    state.setdefault("news_items_count", 0)
    state.setdefault("evidence_level", "NONE")
    state.setdefault("shock_type", "NO_NEWS")
    state.setdefault("governance_decision", {})
    if "validation_metrics" not in state and "forecast_data" in state:
        state["validation_metrics"] = state["forecast_data"].get("validation_metrics", {})


def _append_audit(
    state: AgentState,
    phase: str,
    status: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    state.setdefault("audit_trail", [])
    state["audit_trail"].append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "phase": phase,
            "status": status,
            "message": message,
            "details": details or {},
        }
    )


def _invoke_context_committee(state: AgentState) -> Dict[str, Any]:
    prompt = f"""
You are a Quantitative Risk Committee Agent and Market Context Analyst.

Use only the evidence below. Do not invent news, causes, funds, trading activity, or operational incidents.
Do not reveal chain-of-thought. Provide a concise JSON object only.

Ticker: {state.get("ticker")}
Evaluation status: {state.get("evaluation_status")}
Evaluation reason: {state.get("evaluation_reason")}
News evidence level: {state.get("evidence_level")}
News items:
{state.get("news_context", "NO_NEWS")}

Allowed shock_type values:
NO_NEWS, TREND_SHIFT, EVENT_DRIVEN, BLACK_SWAN, DATA_ISSUE, MODEL_DEGRADATION

Return JSON with keys:
assessment, interpretation, shock_type, evidence_level.
If evidence is weak, say "insufficient evidence" in interpretation.
"""
    try:
        response = _get_llm().invoke([HumanMessage(content=prompt)]).content
        parsed = _parse_json_response(response)
        if parsed:
            return parsed
    except Exception as exc:
        logger.warning("Context committee LLM call failed | error=%s", exc)

    return {
        "assessment": "Relevant news was detected, but the automated assessment could not be completed.",
        "interpretation": "insufficient evidence to attribute forecast degradation to a specific external event.",
        "shock_type": "EVENT_DRIVEN",
        "evidence_level": state.get("evidence_level", "LOW"),
    }


def _invoke_recommendation_committee(state: AgentState) -> Dict[str, Any]:
    evidence = _recommendation_evidence_bundle(state)
    prompt = f"""
You are a Quantitative Risk Committee Agent and Model Governance Reviewer.

Use only the provided state. Do not invent news, hidden market activity, fund activity, or causes without evidence.
Do not reveal chain-of-thought. This is research/paper trading only, not order execution.
Return concise JSON only. Missing values must be written as "not available" or omitted safely.

Required writing style:
- Explain what each metric means, whether it is supportive or risky, and how it affects the signal.
- Use phrases such as "based on available metrics", "the data suggests", and
  "this should be treated as a risk-control signal, not a directional trading conviction" when appropriate.
- If model_status is PASS but the risk engine signal is MANUAL_REVIEW, explicitly explain the conflict.
- If news_found is false, write "No relevant news evidence was found." and do not speculate about events.
- Do not instruct the user to buy, sell, hold, maintain, or reduce exposure. Describe the research signal and controls only.

Ticker: {state.get("ticker")}
Evidence bundle:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Return JSON with keys:
assessment_summary, interpretation, decision_rationale, final_recommendation, evidence_used.

The interpretation value must be an object with exactly these keys:
forecast_performance, validation_reliability, risk_profile, market_regime,
drift_condition, news_context_evidence, governance_retrain_status.

The interpretation must address these evidence groups when available:
1. Forecast performance: holdout MAE, holdout RMSE, holdout MAPE, model status, threshold.
2. Walk-forward validation: MAPE, directional accuracy, interval_95_coverage, fold count.
3. Forecast distribution: expected_return_7d, downside_risk_95, upside_potential_95,
   forecast interval width, median forecast.
4. Risk engine: risk_level, VaR 95, Expected Shortfall, risk_reward_ratio,
   preliminary/final signal, signal_confidence.
5. Market regime: volatility_regime, trend_regime, liquidity_regime, regime_confidence.
6. Drift detection: severity, feature/target/concept drift flags, drifted_features.
7. News/context: news_found, evidence_level, shock_type, news summary.
8. Governance/retrain: retrain_attempts, governance_decision, action_taken,
   champion/challenger comparison, rollback if present.

Keep the answer professional, evidence-based, and not financial advice.
"""
    try:
        response = _get_llm().invoke(
            [
                SystemMessage(content="You produce evidence-based quant risk summaries in valid JSON."),
                HumanMessage(content=prompt),
            ]
        ).content
        parsed = _parse_json_response(response)
        if parsed:
            return _normalize_committee_assessment(parsed, state, evidence)
    except Exception as exc:
        logger.warning("Recommendation committee LLM call failed | error=%s", exc)

    return _fallback_committee_assessment(state, evidence)


def _recommendation_evidence_bundle(state: AgentState) -> Dict[str, Any]:
    forecast_data = state.get("forecast_data", {})
    if not isinstance(forecast_data, dict):
        forecast_data = {}
    holdout = forecast_data.get("metrics", {}) if isinstance(forecast_data, dict) else {}
    validation_report = state.get("validation_metrics") or forecast_data.get("validation_metrics", {})
    validation = _validation_summary(validation_report)
    risk = state.get("risk_report", {})
    regime = state.get("regime_report", {})
    drift = state.get("drift_report", {})
    governance = state.get("governance_decision", {})
    threshold = _configured_mape_threshold()
    distribution = _forecast_distribution_summary(forecast_data, risk)
    news = {
        "news_found": state.get("news_found", False),
        "news_items_count": state.get("news_items_count", 0),
        "evidence_level": state.get("evidence_level", "NONE"),
        "shock_type": state.get("shock_type", "NO_NEWS"),
        "summary": state.get("news_context", "NO_NEWS"),
        "news_evidence": state.get("news_evidence", {}),
    }

    evidence_used = []
    if holdout:
        evidence_used.append("holdout_metrics")
    if validation:
        evidence_used.append("walk_forward_validation")
    if distribution:
        evidence_used.append("forecast_distribution")
    if risk:
        evidence_used.append("risk_report")
    if regime:
        evidence_used.append("regime_report")
    if drift:
        evidence_used.append("drift_report")
    evidence_used.append("news_context")
    evidence_used.append("governance_status")

    return {
        "model": {
            "status": state.get("evaluation_status", "not available"),
            "reason": state.get("evaluation_reason", "not available"),
            "threshold": threshold,
            "holdout_mae": holdout.get("MAE", holdout.get("mae", "not available")),
            "holdout_rmse": holdout.get("RMSE", holdout.get("rmse", "not available")),
            "holdout_mape": holdout.get("MAPE", holdout.get("mape", "not available")),
        },
        "walk_forward_validation": validation,
        "forecast_distribution": distribution,
        "risk_engine": {
            "risk_level": risk.get("risk_level", "not available"),
            "var_95": risk.get("var_95", "not available"),
            "expected_shortfall": risk.get("expected_shortfall", "not available"),
            "risk_reward_ratio": risk.get("risk_reward_ratio", "not available"),
            "expected_return_7d": risk.get("expected_return_7d", "not available"),
            "downside_risk_95": risk.get("downside_risk_95", "not available"),
            "upside_potential_95": risk.get("upside_potential_95", "not available"),
            "preliminary_signal": risk.get("preliminary_signal", "not available"),
            "final_signal": state.get("trading_signal", risk.get("preliminary_signal", "not available")),
            "signal_confidence": state.get("signal_confidence", risk.get("signal_confidence", "not available")),
            "risk_notes": risk.get("risk_notes", []),
        },
        "market_regime": {
            "volatility_regime": regime.get("volatility_regime", "not available"),
            "trend_regime": regime.get("trend_regime", "not available"),
            "liquidity_regime": regime.get("liquidity_regime", "not available"),
            "regime_confidence": regime.get("regime_confidence", "not available"),
            "regime_notes": regime.get("regime_notes", []),
        },
        "drift_detection": {
            "severity": drift.get("severity", "not available"),
            "feature_drift_detected": drift.get("feature_drift_detected", "not available"),
            "target_drift_detected": drift.get("target_drift_detected", "not available"),
            "concept_drift_detected": drift.get("concept_drift_detected", "not available"),
            "drifted_features": drift.get("drifted_features", []),
            "recommended_action": drift.get("recommended_action", "not available"),
            "drift_notes": drift.get("drift_notes", []),
        },
        "news_context": news,
        "governance_retrain": {
            "retrain_attempts": state.get("retry_count", 0),
            "governance_decision": governance,
            "action_taken": state.get("action_taken", "not available"),
            "champion_metrics": governance.get("champion_metrics", {}),
            "challenger_metrics": governance.get("challenger_metrics", {}),
            "rollback": governance.get("rollback", "not available"),
        },
        "evidence_used": evidence_used,
    }


def _configured_mape_threshold() -> Any:
    try:
        return load_config().get("thresholds", {}).get("max_mape", "not available")
    except Exception:
        return "not available"


def _forecast_distribution_summary(forecast_data: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    forecasts = forecast_data.get("forecasts", []) if isinstance(forecast_data, dict) else []
    if not forecasts:
        return {}
    final_step = forecasts[-1]
    current_price = _as_float(forecast_data.get("current_price"))
    lower_95 = _as_float(final_step.get("q_0.025"))
    median = _as_float(final_step.get("q_0.5"))
    upper_95 = _as_float(final_step.get("q_0.975"))
    interval_width = None
    interval_width_pct = None
    if lower_95 is not None and upper_95 is not None:
        interval_width = upper_95 - lower_95
        if current_price and current_price > 0:
            interval_width_pct = interval_width / current_price
    return {
        "horizon_step": final_step.get("step", "not available"),
        "current_price": current_price if current_price is not None else "not available",
        "median_forecast": median if median is not None else "not available",
        "lower_95_forecast": lower_95 if lower_95 is not None else "not available",
        "upper_95_forecast": upper_95 if upper_95 is not None else "not available",
        "forecast_interval_width": interval_width if interval_width is not None else "not available",
        "forecast_interval_width_pct": interval_width_pct if interval_width_pct is not None else "not available",
        "expected_return_7d": risk.get("expected_return_7d", "not available"),
        "downside_risk_95": risk.get("downside_risk_95", "not available"),
        "upside_potential_95": risk.get("upside_potential_95", "not available"),
    }


def _normalize_committee_assessment(
    parsed: Dict[str, Any],
    state: AgentState,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    fallback = _fallback_committee_assessment(state, evidence)
    interpretation = parsed.get("interpretation", fallback["interpretation"])
    if not isinstance(interpretation, dict):
        interpretation = {"forecast_performance": str(interpretation)}
    for key, value in fallback["interpretation"].items():
        interpretation.setdefault(key, value)
    interpretation["risk_profile"] = fallback["interpretation"]["risk_profile"]
    if not state.get("news_found", False):
        interpretation["news_context_evidence"] = fallback["interpretation"]["news_context_evidence"]

    evidence_used = parsed.get("evidence_used", evidence.get("evidence_used", []))
    if not isinstance(evidence_used, list):
        evidence_used = evidence.get("evidence_used", [])

    return {
        "assessment_summary": parsed.get(
            "assessment_summary",
            parsed.get("assessment", fallback["assessment_summary"]),
        ),
        "interpretation": interpretation,
        "decision_rationale": parsed.get(
            "decision_rationale",
            parsed.get("final_signal_reason", fallback["decision_rationale"]),
        ),
        "final_recommendation": _safe_final_recommendation(
            parsed.get("final_recommendation"),
            fallback["final_recommendation"],
        ),
        "evidence_used": evidence_used,
    }


def _fallback_committee_assessment(state: AgentState, evidence: Dict[str, Any]) -> Dict[str, Any]:
    model = evidence.get("model", {})
    validation = evidence.get("walk_forward_validation", {})
    distribution = evidence.get("forecast_distribution", {})
    risk = evidence.get("risk_engine", {})
    regime = evidence.get("market_regime", {})
    drift = evidence.get("drift_detection", {})
    news = evidence.get("news_context", {})
    governance = evidence.get("governance_retrain", {})
    final_signal = risk.get("final_signal", state.get("trading_signal", "HOLD"))
    model_status = model.get("status", "not available")
    risk_level = risk.get("risk_level", "not available")
    drift_severity = drift.get("severity", "not available")

    interpretation = {
        "forecast_performance": (
            "Based on available metrics, model status is "
            f"{model_status}; holdout MAPE is {_fmt_pct(model.get('holdout_mape'))} versus the configured "
            f"threshold of {_fmt_pct(model.get('threshold'))}. Holdout MAE is {_fmt_num(model.get('holdout_mae'))} "
            f"and RMSE is {_fmt_num(model.get('holdout_rmse'))}, which describe average forecast error and "
            "larger-error sensitivity respectively."
        ),
        "validation_reliability": (
            f"Walk-forward MAPE is {_fmt_pct(validation.get('mape'))} across "
            f"{validation.get('fold_count', 'not available')} folds. Directional accuracy is "
            f"{_fmt_pct(validation.get('directional_accuracy'))}, so the data suggests direction calls are only "
            f"modestly informative. The 95% interval coverage is {_fmt_pct(validation.get('interval_95_coverage'))}; "
            "coverage below 95% reduces confidence in the uncertainty estimate."
        ),
        "risk_profile": (
            f"The 7-day median forecast is {_fmt_num(distribution.get('median_forecast'))} with a 95% interval "
            f"width of {_fmt_num(distribution.get('forecast_interval_width'))} "
            f"({_fmt_pct(distribution.get('forecast_interval_width_pct'))} of current price). Expected return is "
            f"{_fmt_pct(risk.get('expected_return_7d'))}, downside risk is {_fmt_pct(risk.get('downside_risk_95'))}, "
            f"upside potential is {_fmt_pct(risk.get('upside_potential_95'))}, VaR 95 is "
            f"{_fmt_pct(risk.get('var_95'))}, Expected Shortfall is {_fmt_pct(risk.get('expected_shortfall'))}, "
            f"and risk/reward is {_fmt_num(risk.get('risk_reward_ratio'))}. The risk engine reports "
            f"{risk_level} risk with signal confidence {_fmt_num(risk.get('signal_confidence'))}."
        ),
        "market_regime": (
            f"Current regime is volatility={regime.get('volatility_regime', 'not available')}, "
            f"trend={regime.get('trend_regime', 'not available')}, liquidity={regime.get('liquidity_regime', 'not available')}, "
            f"with confidence {_fmt_num(regime.get('regime_confidence'))}. This context affects how aggressively "
            "the forecast should be trusted."
        ),
        "drift_condition": (
            f"Drift severity is {drift_severity}; feature drift={drift.get('feature_drift_detected', 'not available')}, "
            f"target drift={drift.get('target_drift_detected', 'not available')}, and concept drift="
            f"{drift.get('concept_drift_detected', 'not available')}. Drifted feature count is "
            f"{len(drift.get('drifted_features', []) or [])}, which can reduce confidence even when holdout metrics pass."
        ),
        "news_context_evidence": _news_interpretation(news),
        "governance_retrain_status": (
            f"Retrain attempts={governance.get('retrain_attempts', 0)}; governance decision="
            f"{governance.get('governance_decision') or 'not available'}; action taken="
            f"{governance.get('action_taken', 'not available')}. Champion/challenger comparison is "
            f"{'available' if governance.get('champion_metrics') or governance.get('challenger_metrics') else 'not available'}."
        ),
    }

    conflict = ""
    if model_status == "PASS" and final_signal == "MANUAL_REVIEW":
        conflict = (
            " Although the model passed the MAPE threshold, the risk engine raised MANUAL_REVIEW because "
            f"risk_level={risk_level} and drift severity={drift_severity}; therefore the decision is conservative."
        )

    return {
        "assessment_summary": (
            f"Based on available metrics, the model status is {model_status} and the final research signal is "
            f"{final_signal} with risk level {risk_level}."
        ),
        "interpretation": interpretation,
        "decision_rationale": (
            f"The final signal remains controlled by the risk engine, not by the language model.{conflict} "
            f"The data suggests the signal should be read together with validation reliability, regime, drift, and news evidence."
        ),
        "final_recommendation": (
            f"Treat {final_signal} as a research and risk-control output only. This should be treated as a "
            "risk-control signal, not a directional trading conviction. This is research output only and not financial advice."
        ),
        "evidence_used": evidence.get("evidence_used", []),
    }


def _news_interpretation(news: Dict[str, Any]) -> str:
    if not news.get("news_found", False):
        return (
            "No relevant news evidence was found. shock_type=NO_NEWS and evidence_level=NONE; "
            "there is insufficient evidence to attribute this to a specific news event."
        )
    return (
        f"News evidence was found with evidence_level={news.get('evidence_level', 'not available')} "
        f"and shock_type={news.get('shock_type', 'not available')}. The assessment should rely only on the "
        "provided news summary and avoid unsupported causal claims."
    )


def _safe_final_recommendation(value: Any, fallback: str) -> str:
    text = str(value).strip() if value else fallback
    disclaimer = "This is research output only and not financial advice."
    if "not financial advice" not in text.lower():
        text = f"{text} {disclaimer}"
    return text


def _compare_champion_challenger(
    champion_forecast: Dict[str, Any],
    challenger_forecast: Dict[str, Any],
) -> Dict[str, Any]:
    champion = _metric_bundle(champion_forecast)
    challenger = _metric_bundle(challenger_forecast)

    old_mape = champion.get("holdout_mape") or 1.0
    new_mape = challenger.get("holdout_mape") or 1.0
    old_da = champion.get("directional_accuracy")
    new_da = challenger.get("directional_accuracy")
    old_cov = champion.get("interval_95_coverage")
    new_cov = challenger.get("interval_95_coverage")
    old_rmse = champion.get("rmse")
    new_rmse = challenger.get("rmse")

    gates = {
        "mape_improved": new_mape < old_mape,
        "directional_accuracy_ok": _not_degraded(old_da, new_da, max_degradation=0.03),
        "interval_95_coverage_ok": _not_degraded(old_cov, new_cov, max_degradation=0.08),
        "rmse_ok": True if old_rmse is None or new_rmse is None else new_rmse <= old_rmse * 1.10,
    }
    accepted = all(gates.values())
    reason = (
        "MAPE improved and validation gates passed."
        if accepted
        else "Challenger rejected because one or more governance gates failed."
    )
    return {
        "decision": "ACCEPT_CHALLENGER" if accepted else "KEEP_CHAMPION",
        "accepted": accepted,
        "reason": reason,
        "champion_metrics": champion,
        "challenger_metrics": challenger,
        "gates": gates,
    }


def _metric_bundle(forecast: Dict[str, Any]) -> Dict[str, Optional[float]]:
    holdout = forecast.get("metrics", {})
    validation = _validation_summary(forecast.get("validation_metrics", {}))
    return {
        "holdout_mape": _as_float(holdout.get("MAPE", holdout.get("mape"))),
        "holdout_mae": _as_float(holdout.get("MAE", holdout.get("mae"))),
        "rmse": _as_float(validation.get("rmse") or holdout.get("RMSE") or holdout.get("rmse")),
        "directional_accuracy": _as_float(validation.get("directional_accuracy")),
        "interval_95_coverage": _as_float(validation.get("interval_95_coverage")),
    }


def _validation_summary(validation_metrics: Dict[str, Any]) -> Dict[str, Any]:
    metrics = validation_metrics.get("metrics") if isinstance(validation_metrics, dict) else {}
    if not isinstance(metrics, dict):
        return {}
    return {
        "status": validation_metrics.get("status"),
        "fold_count": validation_metrics.get("fold_count"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mape": metrics.get("mape"),
        "smape": metrics.get("smape"),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "interval_95_coverage": metrics.get("interval_95_coverage", metrics.get("interval_coverage")),
        "pinball_loss": metrics.get("pinball_loss"),
        "prediction_bias": metrics.get("prediction_bias"),
    }


def _not_degraded(old_value: Optional[float], new_value: Optional[float], max_degradation: float) -> bool:
    if old_value is None or new_value is None:
        return True
    return new_value >= old_value - max_degradation


def _parse_json_response(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _format_assessment_text(assessment: Dict[str, Any]) -> str:
    return (
        f"Assessment:\n{assessment.get('assessment', 'N/A')}\n\n"
        f"Interpretation:\n{assessment.get('interpretation', 'N/A')}\n\n"
        f"Shock classification:\n{assessment.get('shock_type', 'NO_NEWS')}"
    )


def _format_final_recommendation(state: AgentState, committee: Dict[str, Any]) -> str:
    return (
        f"Assessment:\n{committee.get('assessment', 'N/A')}\n\n"
        f"Interpretation:\n{committee.get('interpretation', 'N/A')}\n\n"
        f"Limitations:\n{committee.get('limitations', 'Research signal only. No broker execution.')}\n\n"
        f"Final research signal:\n{state.get('trading_signal', 'HOLD')}\n"
        f"Confidence: {state.get('signal_confidence', 0.0):.2f}\n"
        f"Reason: {committee.get('final_signal_reason', 'Signal follows risk engine and governance checks.')}"
    )


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _as_float(value)
    return "N/A" if number is None else f"{number:.4f}"


def _fmt_num(value: Any) -> str:
    number = _as_float(value)
    return "not available" if number is None else f"{number:,.4f}"


def _fmt_pct(value: Any) -> str:
    number = _as_float(value)
    return "not available" if number is None else f"{number * 100:.2f}%"
