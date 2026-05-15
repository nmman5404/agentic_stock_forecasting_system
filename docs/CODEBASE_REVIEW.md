# Codebase Review

## Current Shape

The codebase is organized around a champion model with an optional challenger model:

- `src/orchestration/daily_pipeline.py` runs ingestion, feature processing, champion forecast generation, LangGraph workflow, and reports.
- `src/modeling/` owns LightGBM Quantile forecasting and walk-forward validation.
- `src/monitoring/` owns drift and regime reports.
- `src/risk/risk_engine.py` owns risk measurement only.
- `src/agent/` owns grouped `AgentState`, LangGraph nodes, retrain policy, prompts, config patch validation, and governance.
- `src/reporting/generator.py` writes executive JSON, technical JSON, Markdown, and HTML reports.

## AgentState Groups

Top-level state is grouped:

```python
{
    "ticker": str,
    "run_id": str,
    "workflow": {...},
    "champion": {...},
    "challenger": {...},
    "news": {...},
    "improvement": {...},
    "governance": {...},
    "recommendation": {...},
    "audit": {...},
}
```

`champion` and `challenger` hold forecast data, validation metrics, drift report, regime report, risk report, diagnostics, and monitoring summary.

## Graph

```text
validate forecast
  -> evaluate champion monitoring
  -> if OK: generate report
  -> otherwise search Google News RSS
  -> plan retrain
  -> validate or repair config patch
  -> if valid train challenger
  -> evaluate challenger
  -> compare models
  -> generate report
```

## Decision Boundaries

- Drift detector reports evidence and severity.
- Regime detector reports market state.
- Risk engine reports risk metrics and risk level.
- Improvement policy opens retrain planning when technical degradation is present.
- Agent proposes a config patch in strict JSON.
- Config patch validator is the safety boundary.
- Governance decides whether the challenger can replace the champion.
- Recommendation node decides the final research action.

## Reports

The report layer writes:

- compact executive JSON
- detailed technical JSON
- Markdown report
- HTML report

Technical JSON is the debug artifact. Executive JSON is the reviewer-facing summary.
