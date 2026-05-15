from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go

from src.agent.state import AgentState
from utils.logger import get_logger

logger = get_logger("ReportGenerator")


def ensure_directories() -> Dict[str, Path]:
    base = Path("reports")
    folders = {
        "json": base / "json",
        "markdown": base / "markdown",
        "html": base / "html",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def generate_reports(state: AgentState) -> Dict[str, str]:
    folders = ensure_directories()
    ticker = state["ticker"]
    today_str = datetime.now().strftime("%Y-%m-%d")

    executive_json_path = folders["json"] / f"{ticker}_executive_report_{today_str}.json"
    technical_json_path = folders["json"] / f"{ticker}_technical_pipeline_report_{today_str}.json"
    md_path = folders["markdown"] / f"{ticker}_report_{today_str}.md"
    html_path = folders["html"] / f"{ticker}_report_{today_str}.html"

    _write_json(executive_json_path, _build_executive_json_payload(state))
    _write_json(technical_json_path, _build_technical_json_payload(state))
    md_path.write_text(_build_markdown_report(state, today_str), encoding="utf-8")
    html_path.write_text(_build_html_report(state, today_str), encoding="utf-8")

    logger.info("Reports saved | ticker=%s | date=%s", ticker, today_str)
    return {
        "executive_json": str(executive_json_path),
        "technical_pipeline_json": str(technical_json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }


def _build_executive_json_payload(state: AgentState) -> Dict[str, Any]:
    final_model = _final_model_name(state)
    candidate = _candidate(state, final_model)
    forecast = _as_dict(candidate.get("forecast_data"))
    validation = _as_dict(candidate.get("validation_metrics"))
    recommendation = _as_dict(state.get("recommendation"))
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    return {
        "metadata": {
            "ticker": state.get("ticker", "UNKNOWN"),
            "run_id": state.get("run_id", "N/A"),
            "as_of_date": forecast.get("as_of_date", "N/A"),
            "generated_at": generated_at,
            "report_type": "executive",
            "final_model": final_model,
        },
        "ticker": state.get("ticker", "UNKNOWN"),
        "run_id": state.get("run_id", "N/A"),
        "as_of_date": forecast.get("as_of_date", "N/A"),
        "forecast_values": _forecast_values(forecast, state),
        "basic_model_metrics": _basic_model_metrics(validation),
        "walk_forward_summary": _walk_forward_summary(validation),
        "risk_summary": _risk_summary(_as_dict(candidate.get("risk_report"))),
        "regime_summary": _regime_summary(_as_dict(candidate.get("regime_report"))),
        "drift_summary": _drift_summary(_as_dict(candidate.get("drift_report"))),
        "news_summary": _news_summary(_as_dict(state.get("news"))),
        "improvement": _improvement_summary(state),
        "final_research_signal": {
            "signal": recommendation.get("final_action", "MANUAL_REVIEW"),
            "confidence": recommendation.get("confidence", 0.0),
        },
        "committee_assessment_summary": {
            "assessment_summary": recommendation.get("assessment_summary", "N/A"),
            "decision_rationale": recommendation.get("decision_rationale", "N/A"),
            "final_recommendation": recommendation.get("final_report", "N/A"),
        },
    }


def _build_technical_json_payload(state: AgentState) -> Dict[str, Any]:
    final_model = _final_model_name(state)
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return {
        "metadata": {
            "ticker": state.get("ticker", "UNKNOWN"),
            "run_id": state.get("run_id", "N/A"),
            "generated_at": generated_at,
            "report_type": "technical_pipeline",
            "final_model": final_model,
        },
        "workflow": state.get("workflow", {}),
        "champion": _candidate_payload(_as_dict(state.get("champion"))),
        "challenger": _candidate_payload(_as_dict(state.get("challenger"))),
        "news": state.get("news", {}),
        "improvement": state.get("improvement", {}),
        "governance": state.get("governance", {}),
        "recommendation": state.get("recommendation", {}),
        "config_patch_validation_result": _config_patch_validation_result(state),
        "audit_trail": _as_dict(state.get("audit")).get("trail", []),
        "errors_warnings": {
            "errors": _as_dict(state.get("audit")).get("errors", []),
            "news_errors": _as_dict(state.get("news")).get("errors", []),
            "config_patch_warnings": _as_dict(state.get("improvement")).get("config_patch_warnings", []),
        },
    }


def _build_markdown_report(state: AgentState, today_str: str) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    final_model = _final_model_name(state)
    candidate = _candidate(state, final_model)
    validation = _validation_metrics(_as_dict(candidate.get("validation_metrics")))
    regime = _as_dict(candidate.get("regime_report"))
    drift = _as_dict(candidate.get("drift_report"))
    risk = _as_dict(candidate.get("risk_report"))
    news = _as_dict(state.get("news"))
    recommendation = _as_dict(state.get("recommendation"))
    diagnostics = _as_dict(candidate.get("diagnostics"))
    statuses = _as_dict(diagnostics.get("statuses"))

    return f"""# Quant Research Report: {ticker} ({today_str})

## Executive Summary
- Final model: **{final_model}**
- Final research action: **{recommendation.get('final_action', 'MANUAL_REVIEW')}**
- Confidence: **{_number(recommendation.get('confidence'))}**
- Accuracy check: **{statuses.get('accuracy_check_status', 'UNKNOWN')}**
- Walk-forward reliability: **{statuses.get('walk_forward_reliability_status', 'UNKNOWN')}**
- Risk clearance: **{statuses.get('risk_clearance_status', 'WATCH')}**
- Overall trust status: **{statuses.get('overall_trust_status', 'REQUIRES_MANUAL_REVIEW')}**
- Risk level: **{risk.get('risk_level', 'N/A')}**
- Drift severity: **{drift.get('severity', 'N/A')}**
- News status: **{news.get('status', 'NO_NEWS')}**

## Forecast
- Current price: {_number(_as_dict(candidate.get('forecast_data')).get('current_price'))}
- Expected return: {_pct(risk.get('expected_return'))}
- Downside risk 95%: {_pct(risk.get('downside_risk_95'))}
- Upside potential 95%: {_pct(risk.get('upside_potential_95'))}
- Risk/reward ratio: {_number(risk.get('risk_reward_ratio'))}

## Walk-forward Validation
- Evaluation method: **walk_forward**
- Fold count: **{_as_dict(candidate.get('validation_metrics')).get('fold_count', 'N/A')}**
- MAE: {_number(validation.get('mae'))}
- RMSE: {_number(validation.get('rmse'))}
- MAPE: {_pct(validation.get('mape'))}
- SMAPE: {_pct(validation.get('smape'))}
- Directional accuracy: {_pct(validation.get('directional_accuracy'))}
- 95% interval coverage: {_pct(validation.get('interval_95_coverage'))}
- Pinball loss: {_number(validation.get('pinball_loss'))}

## Monitoring
- Regime label: **{regime.get('final_regime_label', 'N/A')}**
- Volatility regime: **{regime.get('volatility_regime', 'N/A')}**
- Trend regime: **{regime.get('trend_regime', 'N/A')}**
- Volume regime: **{regime.get('volume_regime', 'N/A')}**
- Drift feature/target/concept scores: **{drift.get('feature_score', 0)} / {drift.get('target_score', 0)} / {drift.get('concept_score', 0)}**
- Drift recommended action: **{drift.get('recommended_action', 'N/A')}**

## Agent Improvement Plan
{_improvement_markdown(state)}

## News Context
- Google News used: **{news.get('google_news_used', False)}**
- Evidence level: **{news.get('evidence_level', 'NONE')}**
- Shock type: **{news.get('shock_type', 'NO_NEWS')}**
- Raw/matched items: **{news.get('raw_news_items_count', 0)} / {news.get('matched_news_items_count', 0)}**
- Debug path: `{news.get('debug_path', 'N/A')}`

```text
{news.get('context', 'NO_NEWS')}
```

## Governance
- Decision: **{_as_dict(state.get('governance')).get('decision', 'NOT_REQUIRED')}**
- Final model: **{_as_dict(state.get('governance')).get('final_model', final_model)}**
- Accepted challenger: **{_as_dict(state.get('governance')).get('accepted_challenger', False)}**
- Reason: {_as_dict(state.get('governance')).get('reason', 'N/A')}

## Final Recommendation
{recommendation.get('final_report', 'Research output only; not financial advice.')}

## Audit Trail
{_audit_markdown(_as_dict(state.get('audit')).get('trail', []))}
"""


def _build_html_report(state: AgentState, today_str: str) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    final_model = _final_model_name(state)
    candidate = _candidate(state, final_model)
    forecasts = _as_dict(candidate.get("forecast_data")).get("forecasts", [])
    risk = _as_dict(candidate.get("risk_report"))
    drift = _as_dict(candidate.get("drift_report"))
    regime = _as_dict(candidate.get("regime_report"))
    news = _as_dict(state.get("news"))
    recommendation = _as_dict(state.get("recommendation"))
    action = recommendation.get("final_action", "MANUAL_REVIEW")
    color, bg = _signal_colors(action)
    chart_div = _forecast_chart(ticker, forecasts)

    audit_items = "".join(
        f"<li><b>{html.escape(item.get('phase', 'N/A'))}</b>: {html.escape(item.get('node_execution_status', 'N/A'))} - {html.escape(item.get('message', ''))}</li>"
        for item in _as_dict(state.get("audit")).get("trail", [])
    )

    return f"""
    <html>
        <head>
            <title>Quant Research Report {html.escape(ticker)}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, sans-serif; padding: 20px; background-color: #f8f9fa; color: #1f2933; }}
                .container {{ background-color: white; padding: 24px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); }}
                .signal {{ text-align: center; margin: 20px 0; }}
                .signal span {{ font-size: 22px; font-weight: 700; color: {color}; background-color: {bg}; padding: 12px 34px; border-radius: 24px; border: 1px solid {color}; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
                .panel {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; background: #ffffff; }}
                .panel h3 {{ margin-top: 0; font-size: 15px; color: #334e68; }}
                .section {{ background-color: #eef5ff; border-left: 4px solid #2f80ed; padding: 14px; margin: 18px 0; }}
                .audit {{ background-color: #1f2933; color: #edf2f7; padding: 14px; border-radius: 6px; margin-top: 18px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quant Research Report: {html.escape(ticker)}</h1>
                <p><b>Report date:</b> {html.escape(today_str)} | <b>Final model:</b> {html.escape(final_model)}</p>
                <div class="signal"><span>Final research action: {html.escape(action)}</span></div>
                <div class="grid">
                    <div class="panel"><h3>Risk</h3><p>Level: <b>{risk.get('risk_level', 'N/A')}</b><br>Expected return: {_pct(risk.get('expected_return'))}</p></div>
                    <div class="panel"><h3>Regime</h3><p>{regime.get('final_regime_label', 'N/A')}</p></div>
                    <div class="panel"><h3>Drift</h3><p>Severity: <b>{drift.get('severity', 'N/A')}</b><br>Total score: {drift.get('total_score', 0)}</p></div>
                    <div class="panel"><h3>News</h3><p>Status: <b>{news.get('status', 'NO_NEWS')}</b><br>Raw/matched: {news.get('raw_news_items_count', 0)} / {news.get('matched_news_items_count', 0)}</p></div>
                </div>
                <div class="section"><h3>Assessment</h3><p>{html.escape(str(recommendation.get('assessment_summary', 'N/A')))}</p><p>{html.escape(str(recommendation.get('decision_rationale', 'N/A')))}</p></div>
                <div class="section"><h3>Agent Improvement Plan</h3>{_improvement_html(state)}</div>
                {chart_div}
                <div class="audit">
                    <details>
                        <summary>Audit trail</summary>
                        <ul>{audit_items}</ul>
                    </details>
                </div>
            </div>
        </body>
    </html>
    """


def _forecast_values(forecast: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    workflow = _as_dict(state.get("workflow"))
    return {
        "current_price": forecast.get("current_price"),
        "as_of_date": forecast.get("as_of_date"),
        "price_unit_detected": workflow.get("price_unit_detected", "unknown"),
        "price_scale_note": workflow.get("price_scale_note", "N/A"),
        "horizon_days": len(forecast.get("forecasts", [])),
        "forecasts": forecast.get("forecasts", []),
    }


def _walk_forward_summary(validation: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _validation_metrics(validation)
    return {
        "evaluation_method": validation.get("evaluation_method", "walk_forward"),
        "status": validation.get("status"),
        "fold_count": validation.get("fold_count"),
        "sample_count": validation.get("sample_count"),
        "feature_count": validation.get("feature_count"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mape": metrics.get("mape"),
        "smape": metrics.get("smape"),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "interval_80_coverage": metrics.get("interval_80_coverage"),
        "interval_95_coverage": metrics.get("interval_95_coverage", metrics.get("interval_coverage")),
        "pinball_loss": metrics.get("pinball_loss"),
        "prediction_bias": metrics.get("prediction_bias"),
        "prediction_bias_pct": metrics.get("prediction_bias_pct"),
        "quantile_crossing_rate": metrics.get("quantile_crossing_rate"),
    }


def _basic_model_metrics(validation: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _validation_metrics(validation)
    return {
        "evaluation_method": validation.get("evaluation_method", "walk_forward"),
        "mape": metrics.get("mape"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "interval_95_coverage": metrics.get("interval_95_coverage", metrics.get("interval_coverage")),
    }


def _risk_summary(risk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "expected_return": risk.get("expected_return"),
        "expected_return_7d": risk.get("expected_return_7d"),
        "downside_risk_95": risk.get("downside_risk_95"),
        "upside_potential_95": risk.get("upside_potential_95"),
        "risk_reward_ratio": risk.get("risk_reward_ratio"),
        "var_95": risk.get("var_95"),
        "expected_shortfall": risk.get("expected_shortfall"),
        "risk_level": risk.get("risk_level"),
        "risk_notes": risk.get("risk_notes", []),
    }


def _regime_summary(regime: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "volatility_regime": regime.get("volatility_regime"),
        "trend_regime": regime.get("trend_regime"),
        "volume_regime": regime.get("volume_regime"),
        "final_regime_label": regime.get("final_regime_label"),
        "regime_confidence": regime.get("regime_confidence"),
        "warnings": regime.get("warnings", []),
        "regime_notes": regime.get("regime_notes", []),
    }


def _drift_summary(drift: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feature_drift_detected": drift.get("feature_drift_detected"),
        "target_drift_detected": drift.get("target_drift_detected"),
        "concept_drift_detected": drift.get("concept_drift_detected"),
        "feature_score": drift.get("feature_score"),
        "target_score": drift.get("target_score"),
        "concept_score": drift.get("concept_score"),
        "total_score": drift.get("total_score"),
        "severity": drift.get("severity"),
        "recommended_action": drift.get("recommended_action"),
        "top_drifted_features": [
            item.get("feature")
            for item in drift.get("drifted_features", [])[:5]
            if isinstance(item, dict) and item.get("feature")
        ],
        "drift_notes": drift.get("drift_notes", []),
    }


def _news_summary(news: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": news.get("status", "NO_NEWS"),
        "found": news.get("found", False),
        "items_count": news.get("items_count", 0),
        "raw_news_items_count": news.get("raw_news_items_count", 0),
        "matched_news_items_count": news.get("matched_news_items_count", 0),
        "evidence_level": news.get("evidence_level", "NONE"),
        "shock_type": news.get("shock_type", "NO_NEWS"),
        "debug_path": news.get("debug_path"),
        "queries": news.get("queries", []),
        "google_news_used": news.get("google_news_used", False),
    }


def _improvement_summary(state: AgentState) -> Dict[str, Any]:
    improvement = _as_dict(state.get("improvement"))
    governance = _as_dict(state.get("governance"))
    return {
        "agent_decision": improvement.get("decision"),
        "technical_retrain_required": improvement.get("technical_retrain_required", False),
        "technical_retrain_reasons": improvement.get("technical_retrain_reasons", []),
        "technical_retrain_strategy": improvement.get("technical_retrain_strategy"),
        "retrain_attempted": _as_dict(state.get("workflow")).get("retrain_attempted", False),
        "config_patch_source": improvement.get("config_patch_source"),
        "config_patch_validation_status": improvement.get("config_patch_validation_status"),
        "governance_decision": governance.get("decision"),
        "final_model": governance.get("final_model", _final_model_name(state)),
    }


def _config_patch_validation_result(state: AgentState) -> Dict[str, Any]:
    improvement = _as_dict(state.get("improvement"))
    return {
        "config_patch_valid": improvement.get("config_patch_valid"),
        "config_patch_validation_status": improvement.get("config_patch_validation_status", "NOT_REQUIRED"),
        "config_patch_source": improvement.get("config_patch_source", "NOT_REQUIRED"),
        "config_patch_warnings": improvement.get("config_patch_warnings", []),
        "validated_config_patch": improvement.get("validated_config_patch", {}),
        "repair_attempts": improvement.get("repair_attempts", 0),
        "repair_history": improvement.get("repair_history", []),
    }


def _improvement_markdown(state: AgentState) -> str:
    improvement = _as_dict(state.get("improvement"))
    governance = _as_dict(state.get("governance"))
    workflow = _as_dict(state.get("workflow"))
    return "\n".join(
        [
            f"- Agent diagnosis: **{improvement.get('diagnosis', 'N/A')}**",
            f"- Agent decision: **{improvement.get('decision', 'N/A')}**",
            f"- Technical retrain required: **{improvement.get('technical_retrain_required', False)}**",
            f"- Technical retrain reasons: {_join_notes(improvement.get('technical_retrain_reasons', []))}",
            f"- Technical retrain strategy: **{improvement.get('technical_retrain_strategy', 'NO_ACTION')}**",
            f"- Config patch source: **{improvement.get('config_patch_source', 'NOT_REQUIRED')}**",
            f"- Config patch validation: **{improvement.get('config_patch_validation_status', 'NOT_REQUIRED')}**",
            f"- Retrain attempted: **{workflow.get('retrain_attempted', False)}**",
            f"- Governance decision: **{governance.get('decision', 'NOT_REQUIRED')}**",
            f"- Final model: **{governance.get('final_model', _final_model_name(state))}**",
            f"- Reason: {governance.get('reason', improvement.get('reason', 'N/A'))}",
        ]
    )


def _improvement_html(state: AgentState) -> str:
    lines = _improvement_markdown(state).splitlines()
    return "<ul>" + "".join(f"<li>{html.escape(line.lstrip('- '))}</li>" for line in lines) + "</ul>"


def _forecast_chart(ticker: str, forecasts: list) -> str:
    if not forecasts:
        return "<p>No forecast data available.</p>"
    steps = [f"T+{item.get('step')}" for item in forecasts]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps + steps[::-1],
            y=[item.get("q_0.975") for item in forecasts] + [item.get("q_0.025") for item in forecasts][::-1],
            fill="toself",
            fillcolor="rgba(47, 128, 237, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps + steps[::-1],
            y=[item.get("q_0.9") for item in forecasts] + [item.get("q_0.1") for item in forecasts][::-1],
            fill="toself",
            fillcolor="rgba(47, 128, 237, 0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% interval",
        )
    )
    fig.add_trace(go.Scatter(x=steps, y=[item.get("q_0.5") for item in forecasts], mode="lines+markers", name="Median forecast"))
    fig.update_layout(title=f"7-day quantile forecast for {ticker}", xaxis_title="Horizon", yaxis_title="Price", template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _final_model_name(state: AgentState) -> str:
    governance = _as_dict(state.get("governance"))
    workflow = _as_dict(state.get("workflow"))
    final_model = governance.get("final_model") or workflow.get("active_candidate") or "champion"
    return final_model if final_model in {"champion", "challenger"} else "champion"


def _candidate(state: AgentState, name: str) -> Dict[str, Any]:
    return _as_dict(state.get(name))


def _candidate_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(candidate)
    forecast = dict(_as_dict(payload.get("forecast_data")))
    forecast.pop("validation_metrics", None)
    forecast.pop("metrics", None)
    payload["forecast_data"] = forecast
    return payload


def _validation_metrics(validation_report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = validation_report.get("metrics") if isinstance(validation_report, dict) else {}
    return metrics if isinstance(metrics, dict) else {}


def _signal_colors(signal: str) -> tuple[str, str]:
    if signal == "BUY":
        return "#1f8a4c", "#e6f4ea"
    if signal == "SELL":
        return "#c0392b", "#fdecea"
    if signal == "WATCH":
        return "#b7791f", "#fff7e6"
    if signal == "MANUAL_REVIEW":
        return "#6b46c1", "#f3e8ff"
    return "#5f6b7a", "#eef2f6"


def _audit_markdown(audit_trail: list) -> str:
    if not audit_trail:
        return "- No audit entries available."
    return "\n".join(
        f"- `{item.get('timestamp', 'N/A')}` | {item.get('phase', 'N/A')} | "
        f"**{item.get('node_execution_status', 'N/A')}** | {item.get('message', '')}"
        for item in audit_trail
    )


def _join_notes(notes: list) -> str:
    return " ".join(str(note) for note in notes) if notes else "N/A"


def _number(value: Any) -> str:
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4, default=_json_default)
    logger.info("Report saved | format=json | path=%s", path)


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
