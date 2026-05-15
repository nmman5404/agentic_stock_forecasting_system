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
    ticker = state.get("ticker", "UNKNOWN")
    today_str = datetime.now().strftime("%Y-%m-%d")

    executive_json_path = folders["json"] / f"{ticker}_executive_report_{today_str}.json"
    technical_json_path = folders["json"] / f"{ticker}_technical_pipeline_report_{today_str}.json"
    md_path = folders["markdown"] / f"{ticker}_report_{today_str}.md"
    html_path = folders["html"] / f"{ticker}_report_{today_str}.html"

    _write_json(executive_json_path, _build_json_payload(state, "executive"))
    _write_json(technical_json_path, _build_json_payload(state, "technical"))
    md_path.write_text(_build_markdown_report(state, today_str), encoding="utf-8")
    html_path.write_text(_build_html_report(state, today_str), encoding="utf-8")

    return {
        "executive_json": str(executive_json_path),
        "technical_pipeline_json": str(technical_json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }

def _build_json_payload(state: AgentState, report_type: str) -> Dict[str, Any]:
    return {
        "metadata": {
            "ticker": state.get("ticker", "UNKNOWN"),
            "report_type": report_type,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "retrain_count": state.get("retry_count", 0),
        },
        "forecast_data": state.get("forecast_data", {}),
        "validation_metrics": state.get("validation_metrics", {}),
        "monitoring": state.get("monitoring", {}),
        "current_config": state.get("current_config", {}),
        "rejected_configs": state.get("rejected_configs", []),
        "evaluation": state.get("evaluation", {}),
        "news_context": state.get("news_context", ""),
        "final_report": state.get("final_report", {}),
    }

def _safe_pct(val: Any) -> str:
    try: return f"{float(val)*100:.2f}%"
    except: return "N/A"

def _safe_num(val: Any) -> str:
    try: return f"{float(val):.4f}"
    except: return "N/A"

def _build_markdown_report(state: AgentState, today_str: str) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    final = state.get("final_report", {})
    eval_dict = state.get("evaluation", {})
    mon = state.get("monitoring", {})
    risk = mon.get("risk", {})
    regime = mon.get("regime", {})
    drift = mon.get("drift", {})
    val = state.get("validation_metrics", {}).get("metrics", {})
    fc = state.get("forecast_data", {})
    
    rejected = state.get("rejected_configs", [])
    rejected_md = "\n".join([f"  - Thử nghiệm bị loại: {r.get('reason_it_failed', '')}" for r in rejected])
    if not rejected_md:
        rejected_md = "  - Không có thử nghiệm nào bị loại."

    return f"""# Quant Research Report: {ticker} ({today_str})

## 1. Executive Summary
- **Final Action:** {final.get('action', 'N/A')}
- **Model Status:** {eval_dict.get('status', 'N/A')} (Retries: {state.get('retry_count', 0)})
- **Summary:** {final.get('summary', 'N/A')}

### Luận điểm đầu tư (Reasoning):
{final.get('reasoning', 'N/A')}

## 2. Forecast & Risk
- Current Price: {_safe_num(fc.get('current_price'))}
- Expected Return: {_safe_pct(risk.get('expected_return'))}
- Risk Level: **{risk.get('risk_level', 'N/A')}**
- Value at Risk (95%): {_safe_pct(risk.get('var_95'))}
- Downside 95%: {_safe_pct(risk.get('downside_risk_95'))}

## 3. Monitoring Context
- **Regime:** {regime.get('final_regime_label', 'N/A')}
- **Drift:** {drift.get('final_drift_label', 'N/A')}

## 4. Model Validation (Walk-forward)
- MAPE: {_safe_pct(val.get('mape'))}
- Directional Accuracy: {_safe_pct(val.get('directional_accuracy'))}
- Interval Coverage 80%: **{_safe_pct(val.get('interval_80_coverage'))}**
- Interval Coverage 95%: **{_safe_pct(val.get('interval_95_coverage'))}**

## 5. Agent Workflow Logs
- **LLM Evaluation Reason:** {eval_dict.get('reasoning', 'N/A')}
- **News Context Found:** {"Yes" if state.get('news_context') else "No"}
- **Retries Attempted:** {state.get('retry_count', 0)}
- **Reflection & Self-Correction:**
{rejected_md}
"""

def _build_html_report(state: AgentState, today_str: str) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    final = state.get("final_report", {})
    action = final.get("action", "MANUAL_REVIEW")
    val = state.get("validation_metrics", {}).get("metrics", {})
    eval_dict = state.get("evaluation", {})
    
    bg_color = "#f3f4f6"
    text_color = "#1f2937"
    if action == "BUY": bg_color, text_color = "#def7ec", "#03543f"
    elif action == "SELL": bg_color, text_color = "#fde8e8", "#9b1c1c"
    elif action == "WATCH": bg_color, text_color = "#fef3c7", "#92400e"

    # Chart - Đã bổ sung 80% interval
    forecasts = state.get("forecast_data", {}).get("forecasts", [])
    chart_div = "<p>No forecast data available.</p>"
    if forecasts:
        steps = [f"T+{item.get('step')}" for item in forecasts]
        fig = go.Figure()
        
        # 95% interval (Outer, nhạt)
        fig.add_trace(go.Scatter(x=steps + steps[::-1],
            y=[item.get("q_0.975") for item in forecasts] + [item.get("q_0.025") for item in forecasts][::-1],
            fill="toself", fillcolor="rgba(47, 128, 237, 0.15)", line=dict(color="rgba(255,255,255,0)"), name="95% Interval"))
        
        # 80% interval (Inner, đậm hơn một chút)
        fig.add_trace(go.Scatter(x=steps + steps[::-1],
            y=[item.get("q_0.9") for item in forecasts] + [item.get("q_0.1") for item in forecasts][::-1],
            fill="toself", fillcolor="rgba(47, 128, 237, 0.3)", line=dict(color="rgba(255,255,255,0)"), name="80% Interval"))
            
        # Median Forecast
        fig.add_trace(go.Scatter(x=steps, y=[item.get("q_0.5") for item in forecasts], mode="lines+markers", line=dict(color="#1e40af", width=2), name="Median Forecast"))
        
        fig.update_layout(title=f"7-Day Quantile Forecast for {ticker}", template="plotly_white")
        chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Workflow Logs HTML
    rejected = state.get("rejected_configs", [])
    rejected_html = "".join([f"<li><i>Từ chối:</i> {html.escape(r.get('reason_it_failed', ''))}</li>" for r in rejected])
    if not rejected_html:
        rejected_html = "<li>Không có thử nghiệm nào bị loại.</li>"

    return f"""
    <html>
        <head><style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; padding: 20px; line-height: 1.6; background-color: #f8f9fa; color: #374151; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }}
            .signal {{ padding: 15px; border-radius: 8px; font-size: 22px; font-weight: bold; background: {bg_color}; color: {text_color}; text-align: center; margin-bottom: 20px; }}
            .section {{ margin-top: 20px; padding: 20px; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px; }}
            .metric-box {{ background: white; padding: 15px; border: 1px solid #e5e7eb; border-radius: 6px; text-align: center; }}
            .metric-box h4 {{ margin: 0 0 5px 0; font-size: 14px; color: #6b7280; }}
            .metric-box span {{ font-size: 18px; font-weight: bold; color: #111827; }}
            ul {{ margin-top: 5px; }}
        </style></head>
        <body>
            <div class="container">
                <h1>Quant Research Report: {ticker} ({today_str})</h1>
                <div class="signal">Final Action: {action}</div>
                
                <div class="section">
                    <h3>Assessment Summary</h3>
                    <p>{html.escape(final.get('summary', ''))}</p>
                    <p><b>Reasoning:</b> {html.escape(final.get('reasoning', ''))}</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-box"><h4>MAPE</h4><span>{_safe_pct(val.get('mape'))}</span></div>
                    <div class="metric-box"><h4>Coverage 80%</h4><span>{_safe_pct(val.get('interval_80_coverage'))}</span></div>
                    <div class="metric-box"><h4>Coverage 95%</h4><span>{_safe_pct(val.get('interval_95_coverage'))}</span></div>
                </div>

                <div class="section" style="background: white; padding:0; border:none; margin-top: 30px;">
                    {chart_div}
                </div>
                
                <div class="section" style="background: #1f2937; color: #f3f4f6;">
                    <h3 style="color: white; margin-top:0;">Agent Workflow Logs (System Observability)</h3>
                    <ul>
                        <li><b>LLM Evaluation Reason:</b> {html.escape(eval_dict.get('reasoning', 'N/A'))}</li>
                        <li><b>News Context Found:</b> {"Yes" if state.get('news_context') else "No"}</li>
                        <li><b>Retries Attempted:</b> {state.get('retry_count', 0)}</li>
                    </ul>
                    <b style="margin-left: 20px;">Reflection & Self-Correction (Auto-tuning):</b>
                    <ul style="color: #9ca3af;">
                        {rejected_html}
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4, default=str)
    logger.info("Report saved | format=json | path=%s", path)