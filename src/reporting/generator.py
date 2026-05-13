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
    technical_json_path = folders["json"] / f"{ticker}_technical_report_{today_str}.json"
    json_path = folders["json"] / f"{ticker}_report_{today_str}.json"
    md_path = folders["markdown"] / f"{ticker}_report_{today_str}.md"
    html_path = folders["html"] / f"{ticker}_report_{today_str}.html"

    executive_payload = _build_executive_json_payload(state)
    technical_payload = _build_technical_json_payload(state)
    with executive_json_path.open("w", encoding="utf-8") as f:
        json.dump(executive_payload, f, ensure_ascii=False, indent=4)
    with technical_json_path.open("w", encoding="utf-8") as f:
        json.dump(technical_payload, f, ensure_ascii=False, indent=4)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(executive_payload, f, ensure_ascii=False, indent=4)
    logger.info("Report saved | format=json_executive | path=%s", executive_json_path)
    logger.info("Report saved | format=json_technical | path=%s", technical_json_path)
    logger.info("Report saved | format=json_alias | path=%s", json_path)

    md_path.write_text(_build_markdown_report(state, today_str), encoding="utf-8")
    logger.info("Report saved | format=markdown | path=%s", md_path)

    html_path.write_text(_build_html_report(state, today_str), encoding="utf-8")
    logger.info("Report saved | format=html | path=%s", html_path)

    return {
        "json": str(json_path),
        "json_executive": str(executive_json_path),
        "json_technical": str(technical_json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }


def _build_technical_json_payload(state: AgentState) -> Dict[str, Any]:
    payload = dict(state)
    payload["report_metadata"] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "report_type": "technical",
    }
    payload["news_evidence"] = payload.get(
        "news_evidence",
        {
            "news_found": payload.get("news_found", False),
            "news_items_count": payload.get("news_items_count", 0),
            "evidence_level": payload.get("evidence_level", "NONE"),
        },
    )
    payload.setdefault("agent_assessment_summary", payload.get("final_recommendation", "N/A"))
    payload.setdefault("signal_confidence", payload.get("risk_report", {}).get("signal_confidence", 0.0))
    payload.setdefault(
        "committee_assessment",
        {
            "assessment_summary": payload.get("assessment_summary", payload.get("agent_assessment_summary", "N/A")),
            "interpretation": payload.get("interpretation", {}),
            "decision_rationale": payload.get("decision_rationale", "N/A"),
            "final_recommendation": payload.get("final_recommendation", "N/A"),
            "evidence_used": payload.get("evidence_used", []),
        },
    )
    committee = payload["committee_assessment"]
    payload.setdefault("assessment_summary", committee.get("assessment_summary", "N/A"))
    payload.setdefault("interpretation", committee.get("interpretation", {}))
    payload.setdefault("decision_rationale", committee.get("decision_rationale", "N/A"))
    payload.setdefault("final_recommendation", committee.get("final_recommendation", "N/A"))
    payload.setdefault("evidence_used", committee.get("evidence_used", []))
    if "forecast_data" in payload and isinstance(payload["forecast_data"], dict):
        payload.pop("validation_metrics", None)
    if "committee_assessment" in payload:
        payload.pop("agent_assessment_summary", None)
        payload.pop("assessment_summary", None)
        payload.pop("interpretation", None)
        payload.pop("decision_rationale", None)
        payload.pop("final_recommendation", None)
        payload.pop("evidence_used", None)
    return payload


def _build_executive_json_payload(state: AgentState) -> Dict[str, Any]:
    forecast_data = state.get("forecast_data", {})
    holdout = forecast_data.get("metrics", {}) if isinstance(forecast_data, dict) else {}
    validation_report = (
        forecast_data.get("validation_metrics", state.get("validation_metrics", {}))
        if isinstance(forecast_data, dict)
        else state.get("validation_metrics", {})
    )
    validation = _validation_metrics(validation_report)
    regime = state.get("regime_report", {})
    drift = state.get("drift_report", {})
    risk = state.get("risk_report", {})
    committee = _committee_data(state)
    forecasts = forecast_data.get("forecasts", []) if isinstance(forecast_data, dict) else []
    drifted_features = drift.get("drifted_features", []) if isinstance(drift, dict) else []
    top_drifted_features = [
        item.get("feature")
        for item in drifted_features[:5]
        if isinstance(item, dict) and item.get("feature")
    ]
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    return {
        "metadata": {
            "ticker": state.get("ticker", "UNKNOWN"),
            "run_id": state.get("run_id", "N/A"),
            "as_of_date": forecast_data.get("as_of_date", "N/A") if isinstance(forecast_data, dict) else "N/A",
            "generated_at": generated_at,
            "report_type": "executive",
        },
        "summary": {
            "model_status": state.get("evaluation_status", "N/A"),
            "final_signal": state.get("trading_signal", risk.get("preliminary_signal", "HOLD")),
            "signal_confidence": state.get("signal_confidence", risk.get("signal_confidence", 0.0)),
            "risk_level": risk.get("risk_level", "N/A"),
            "drift_severity": drift.get("severity", "N/A"),
            "news_evidence": state.get("evidence_level", "NONE"),
        },
        "forecast": {
            "current_price": forecast_data.get("current_price") if isinstance(forecast_data, dict) else None,
            "horizon_days": len(forecasts),
            "forecasts": forecasts,
        },
        "model_metrics": {
            "holdout": {
                "MAE": holdout.get("MAE", holdout.get("mae")),
                "RMSE": holdout.get("RMSE", holdout.get("rmse")),
                "MAPE": holdout.get("MAPE", holdout.get("mape")),
            },
            "walk_forward": {
                "MAE": validation.get("mae"),
                "RMSE": validation.get("rmse"),
                "MAPE": validation.get("mape"),
                "SMAPE": validation.get("smape"),
                "directional_accuracy": validation.get("directional_accuracy"),
                "interval_80_coverage": validation.get("interval_80_coverage"),
                "interval_95_coverage": validation.get("interval_95_coverage"),
                "pinball_loss": validation.get("pinball_loss"),
                "prediction_bias": validation.get("prediction_bias"),
            },
        },
        "monitoring": {
            "regime": {
                "volatility_regime": regime.get("volatility_regime", "N/A"),
                "trend_regime": regime.get("trend_regime", "N/A"),
                "liquidity_regime": regime.get("liquidity_regime", "N/A"),
                "regime_confidence": regime.get("regime_confidence"),
            },
            "drift": {
                "feature_drift_detected": drift.get("feature_drift_detected"),
                "target_drift_detected": drift.get("target_drift_detected"),
                "concept_drift_detected": drift.get("concept_drift_detected"),
                "severity": drift.get("severity", "N/A"),
                "recommended_action": drift.get("recommended_action", "N/A"),
                "top_drifted_features": top_drifted_features,
            },
        },
        "risk": {
            "expected_return_7d": risk.get("expected_return_7d"),
            "downside_risk_95": risk.get("downside_risk_95"),
            "upside_potential_95": risk.get("upside_potential_95"),
            "risk_reward_ratio": risk.get("risk_reward_ratio"),
            "var_95": risk.get("var_95"),
            "expected_shortfall": risk.get("expected_shortfall"),
            "risk_level": risk.get("risk_level", "N/A"),
            "signal_confidence": risk.get("signal_confidence", state.get("signal_confidence", 0.0)),
        },
        "agent": {
            "assessment_summary": committee.get("assessment_summary", "N/A"),
            "decision_rationale": committee.get("decision_rationale", "N/A"),
            "final_recommendation": committee.get("final_recommendation", "N/A"),
            "evidence_used": committee.get("evidence_used", []),
        },
        "audit_summary": {
            "retrain_attempts": state.get("retry_count", 0),
            "quantiles_fixed": state.get("quantiles_fixed", False),
            "audit_event_count": len(state.get("audit_trail", [])),
        },
    }


def _build_markdown_report(state: AgentState, today_str: str) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    forecast_data = state.get("forecast_data", {})
    holdout = forecast_data.get("metrics", {})
    validation = _validation_metrics(forecast_data.get("validation_metrics", state.get("validation_metrics", {})))
    regime = state.get("regime_report", {})
    drift = state.get("drift_report", {})
    risk = state.get("risk_report", {})
    governance = state.get("governance_decision", {})
    news_evidence = state.get("news_evidence", {})

    return f"""# Quant Risk Committee Report: {ticker} ({today_str})

## 1. Executive Summary
- Model status: **{state.get('evaluation_status', 'N/A')}**
- Final research signal: **{state.get('trading_signal', risk.get('preliminary_signal', 'HOLD'))}**
- Signal confidence: **{_pct_or_number(state.get('signal_confidence', risk.get('signal_confidence', 0.0)))}**
- Risk level: **{risk.get('risk_level', 'N/A')}**
- News evidence: **{news_evidence.get('evidence_level', state.get('evidence_level', 'NONE'))}**

## 2. Forecast Performance
- Holdout MAE: {_number(holdout.get('MAE'))}
- Holdout RMSE: {_number(holdout.get('RMSE'))}
- Holdout MAPE: {_pct(holdout.get('MAPE'))}
- Evaluation reason: {state.get('evaluation_reason', 'N/A')}

## 3. Walk-forward Validation
- MAE: {_number(validation.get('mae'))}
- RMSE: {_number(validation.get('rmse'))}
- MAPE: {_pct(validation.get('mape'))}
- SMAPE: {_pct(validation.get('smape'))}
- Directional accuracy: {_pct(validation.get('directional_accuracy'))}
- 95% interval coverage: {_pct(validation.get('interval_95_coverage'))}
- Pinball loss: {_number(validation.get('pinball_loss'))}
- Prediction bias: {_number(validation.get('prediction_bias'))}

## 4. Market Regime
- Volatility regime: **{regime.get('volatility_regime', 'N/A')}**
- Trend regime: **{regime.get('trend_regime', 'N/A')}**
- Liquidity regime: **{regime.get('liquidity_regime', 'N/A')}**
- Confidence: {_pct_or_number(regime.get('regime_confidence'))}
- Notes: {_join_notes(regime.get('regime_notes', []))}

## 5. Drift Detection
- Feature drift detected: **{drift.get('feature_drift_detected', False)}**
- Target drift detected: **{drift.get('target_drift_detected', False)}**
- Concept drift detected: **{drift.get('concept_drift_detected', False)}**
- Severity: **{drift.get('severity', 'N/A')}**
- Recommended action: **{drift.get('recommended_action', 'N/A')}**
- Notes: {_join_notes(drift.get('drift_notes', []))}

## 6. Risk Assessment
- Expected return 7d: {_pct(risk.get('expected_return_7d'))}
- Downside risk 95%: {_pct(risk.get('downside_risk_95'))}
- Upside potential 95%: {_pct(risk.get('upside_potential_95'))}
- Risk/reward ratio: {_number(risk.get('risk_reward_ratio'))}
- VaR 95%: {_pct(risk.get('var_95'))}
- Expected shortfall: {_pct(risk.get('expected_shortfall'))}
- Preliminary signal: **{risk.get('preliminary_signal', 'N/A')}**
- Risk notes: {_join_notes(risk.get('risk_notes', []))}

## 7. News & Event Context
- News found: **{state.get('news_found', False)}**
- News items count: **{state.get('news_items_count', 0)}**
- Evidence level: **{state.get('evidence_level', 'NONE')}**
- Shock classification: **{state.get('shock_type', 'NO_NEWS')}**

```text
{state.get('news_context', 'NO_NEWS')}
```

## 8. Model Governance Decision
- Decision: **{governance.get('decision', 'N/A')}**
- Accepted challenger: **{governance.get('accepted', False)}**
- Reason: {governance.get('reason', state.get('action_taken', 'N/A'))}
- Action taken: {state.get('action_taken', 'N/A')}

## 9. Committee Assessment
{_committee_markdown(state)}

## 10. Audit Trail
{_audit_markdown(state.get('audit_trail', []))}
"""


def _build_html_report(state: AgentState, today_str: str) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    forecasts = state.get("forecast_data", {}).get("forecasts", [])
    chart_div = _forecast_chart(ticker, forecasts)
    risk = state.get("risk_report", {})
    signal = state.get("trading_signal", risk.get("preliminary_signal", "HOLD"))
    color, bg = _signal_colors(signal)
    committee_html = _committee_html(state)
    audit_items = "".join(
        f"<li><b>{html.escape(item.get('phase', 'N/A'))}</b>: {html.escape(item.get('status', 'N/A'))} - {html.escape(item.get('message', ''))}</li>"
        for item in state.get("audit_trail", [])
    )

    return f"""
    <html>
        <head>
            <title>Quant Risk Report {html.escape(ticker)}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, sans-serif; padding: 20px; background-color: #f8f9fa; color: #1f2933; }}
                .container {{ background-color: white; padding: 24px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); }}
                .signal {{ text-align: center; margin: 20px 0; }}
                .signal span {{ font-size: 22px; font-weight: 700; color: {color}; background-color: {bg}; padding: 12px 34px; border-radius: 24px; border: 1px solid {color}; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
                .panel {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; background: #ffffff; }}
                .panel h3 {{ margin-top: 0; font-size: 15px; color: #334e68; }}
                .committee {{ background-color: #eef5ff; border-left: 4px solid #2f80ed; padding: 14px; margin: 18px 0; }}
                .audit {{ background-color: #1f2933; color: #edf2f7; padding: 14px; border-radius: 6px; margin-top: 18px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quant Risk Committee Report: {html.escape(ticker)}</h1>
                <p><b>Report date:</b> {html.escape(today_str)}</p>
                <div class="signal"><span>Final research signal: {html.escape(signal)}</span></div>
                <div class="grid">
                    <div class="panel"><h3>Risk</h3><p>Level: <b>{risk.get('risk_level', 'N/A')}</b><br>Confidence: {_pct_or_number(state.get('signal_confidence', risk.get('signal_confidence')))}</p></div>
                    <div class="panel"><h3>Regime</h3><p>{state.get('regime_report', {}).get('volatility_regime', 'N/A')} | {state.get('regime_report', {}).get('trend_regime', 'N/A')}</p></div>
                    <div class="panel"><h3>Drift</h3><p>Severity: <b>{state.get('drift_report', {}).get('severity', 'N/A')}</b></p></div>
                    <div class="panel"><h3>News</h3><p>Evidence: <b>{state.get('evidence_level', 'NONE')}</b><br>Shock: {state.get('shock_type', 'NO_NEWS')}</p></div>
                </div>
                <div class="committee"><h3>Committee Assessment</h3>{committee_html}</div>
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


def _committee_data(state: AgentState) -> Dict[str, Any]:
    committee = state.get("committee_assessment", {})
    if not isinstance(committee, dict):
        committee = {}
    return {
        "assessment_summary": committee.get(
            "assessment_summary",
            state.get("assessment_summary", state.get("agent_assessment_summary", "N/A")),
        ),
        "interpretation": committee.get("interpretation", state.get("interpretation", {})),
        "decision_rationale": committee.get(
            "decision_rationale",
            state.get("decision_rationale", "N/A"),
        ),
        "final_recommendation": committee.get(
            "final_recommendation",
            state.get("final_recommendation", "N/A"),
        ),
        "evidence_used": committee.get("evidence_used", state.get("evidence_used", [])),
    }


def _committee_markdown(state: AgentState) -> str:
    committee = _committee_data(state)
    interpretation = committee.get("interpretation", {})
    if isinstance(interpretation, dict):
        interpretation_md = "\n".join(
            f"- {label}: {interpretation.get(key, 'not available')}"
            for key, label in [
                ("forecast_performance", "Forecast performance"),
                ("validation_reliability", "Validation reliability"),
                ("risk_profile", "Risk profile"),
                ("market_regime", "Market regime"),
                ("drift_condition", "Drift condition"),
                ("news_context_evidence", "News/context evidence"),
                ("governance_retrain_status", "Governance/retrain status"),
            ]
        )
    else:
        interpretation_md = f"- {interpretation or 'N/A'}"

    evidence = committee.get("evidence_used", [])
    evidence_text = ", ".join(str(item) for item in evidence) if evidence else "N/A"
    return (
        f"### Assessment\n{committee.get('assessment_summary', 'N/A')}\n\n"
        f"### Interpretation\n{interpretation_md}\n\n"
        f"### Decision Rationale\n{committee.get('decision_rationale', 'N/A')}\n\n"
        f"### Final Recommendation\n{committee.get('final_recommendation', 'N/A')}\n\n"
        f"### Evidence Used\n{evidence_text}"
    )


def _committee_html(state: AgentState) -> str:
    committee = _committee_data(state)
    interpretation = committee.get("interpretation", {})
    if isinstance(interpretation, dict):
        items = "".join(
            f"<li><b>{html.escape(label)}:</b> {html.escape(str(interpretation.get(key, 'not available')))}</li>"
            for key, label in [
                ("forecast_performance", "Forecast performance"),
                ("validation_reliability", "Validation reliability"),
                ("risk_profile", "Risk profile"),
                ("market_regime", "Market regime"),
                ("drift_condition", "Drift condition"),
                ("news_context_evidence", "News/context evidence"),
                ("governance_retrain_status", "Governance/retrain status"),
            ]
        )
    else:
        items = f"<li>{html.escape(str(interpretation or 'N/A'))}</li>"
    evidence = committee.get("evidence_used", [])
    evidence_text = ", ".join(str(item) for item in evidence) if evidence else "N/A"
    return (
        f"<h4>Assessment</h4><p>{html.escape(str(committee.get('assessment_summary', 'N/A')))}</p>"
        f"<h4>Interpretation</h4><ul>{items}</ul>"
        f"<h4>Decision Rationale</h4><p>{html.escape(str(committee.get('decision_rationale', 'N/A')))}</p>"
        f"<h4>Final Recommendation</h4><p>{html.escape(str(committee.get('final_recommendation', 'N/A')))}</p>"
        f"<h4>Evidence Used</h4><p>{html.escape(evidence_text)}</p>"
    )


def _forecast_chart(ticker: str, forecasts: list) -> str:
    if not forecasts:
        return "<p>No forecast data available.</p>"
    steps = [f"T+{f['step']}" for f in forecasts]
    q50 = [f["q_0.5"] for f in forecasts]
    q10 = [f["q_0.1"] for f in forecasts]
    q90 = [f["q_0.9"] for f in forecasts]
    q025 = [f["q_0.025"] for f in forecasts]
    q975 = [f["q_0.975"] for f in forecasts]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps + steps[::-1],
            y=q975 + q025[::-1],
            fill="toself",
            fillcolor="rgba(47, 128, 237, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps + steps[::-1],
            y=q90 + q10[::-1],
            fill="toself",
            fillcolor="rgba(47, 128, 237, 0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% interval",
        )
    )
    fig.add_trace(go.Scatter(x=steps, y=q50, mode="lines+markers", name="Median forecast"))
    fig.update_layout(
        title=f"7-day quantile forecast for {ticker}",
        xaxis_title="Horizon",
        yaxis_title="Price",
        template="plotly_white",
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


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
        f"**{item.get('status', 'N/A')}** | {item.get('message', '')}"
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


def _pct_or_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.2f}"
