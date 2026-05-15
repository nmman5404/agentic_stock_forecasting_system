# Agentic Stock Forecasting System

Research-only forecasting pipeline for Vingroup-related Vietnamese equities. The system ingests OHLCV data, persists raw and processed tables in SQLite, trains LightGBM Quantile forecasts, validates with walk-forward evaluation, monitors drift/regime/risk, optionally uses Google News RSS for context, and generates JSON, Markdown, and HTML reports.

This project is for assessment and research demonstration. It does not connect to a broker and is not financial advice.

## Workflow

```text
Market data ingestion
  -> SQLite raw + processed tables
  -> LightGBM Quantile 7-day forecast
  -> Forecast quantile validation
  -> Walk-forward evaluation
  -> Drift detection
  -> Regime detection
  -> Pure risk measurement
  -> Model health decision
  -> Optional Google News RSS context
  -> Agent retrain plan
  -> Config patch validation / repair
  -> Challenger training
  -> Champion vs challenger governance
  -> Final research recommendation
  -> Reports
```

## Architecture Boundaries

- `src/modeling/`: LightGBM Quantile training and walk-forward validation.
- `src/monitoring/drift_detector.py`: drift evidence metrics, scoring, severity, monitoring action.
- `src/monitoring/regime_detector.py`: independent volatility, trend, and volume regime components.
- `src/risk/risk_engine.py`: forecast risk metrics only.
- `src/agent/improvement_policy.py`: deterministic retrain gate and strategy hint.
- `src/agent/prompts.py`: strict JSON retrain-plan and patch-repair prompts.
- `src/agent/config_patch_validator.py`: safety boundary for LLM-generated config patches.
- `src/agent/governance.py`: champion/challenger comparison using walk-forward metrics.
- `src/agent/nodes.py`: LangGraph nodes for champion evaluation, optional challenger flow, and final recommendation.
- `src/reporting/generator.py`: executive JSON, technical JSON, Markdown, and HTML reports.

## LangGraph Flow

```text
START
  -> node_validate_forecast
  -> node_evaluate_monitoring
      -> OK: node_generate_report
      -> DEGRADED / RETRAIN_REQUIRED / MANUAL_REVIEW:
           node_search_news_context
           -> node_plan_retrain
           -> node_validate_or_repair_patch
                -> invalid/not required: node_generate_report
                -> valid: node_train_challenger
                         -> node_evaluate_challenger
                         -> node_compare_models
                         -> node_generate_report
  -> END
```

## Evaluation

Walk-forward validation is the source of truth. The validation output keeps:

```python
validation_metrics = {
    "evaluation_method": "walk_forward",
    "metrics": {
        "mape": ...,
        "rmse": ...,
        "mae": ...,
        "smape": ...,
        "directional_accuracy": ...,
        "interval_80_coverage": ...,
        "interval_95_coverage": ...,
        "pinball_loss": ...,
        "prediction_bias": ...,
        "prediction_bias_pct": ...,
        "quantile_crossing_rate": ...,
    },
    "fold_count": ...,
    "folds": [...],
    "notes": [...]
}
```

## Risk and Recommendation

`RiskEngine` measures:

- expected return
- downside/upside quantile risk
- VaR 95
- expected shortfall
- risk/reward ratio
- risk level: `LOW_RISK`, `MEDIUM_RISK`, `HIGH_RISK`, `EXTREME_RISK`

The final research action is decided later in `node_generate_report`:

```text
BUY / SELL / HOLD / WATCH / MANUAL_REVIEW
```

## News Source

The active news source is Google News RSS only. News is evidence context, not a trading or retraining authority.

## Outputs

Each run writes four report outputs:

- `reports/json/{ticker}_executive_report_{date}.json`
- `reports/json/{ticker}_technical_pipeline_report_{date}.json`
- `reports/markdown/{ticker}_report_{date}.md`
- `reports/html/{ticker}_report_{date}.html`

The executive JSON is compact. The technical JSON keeps full workflow state, candidate details, retrain planning, governance, recommendation, audit trail, and warnings.

## Run

```bash
python main.py
```

Direct smoke from Python:

```bash
python -c "from src.orchestration.daily_pipeline import run_daily_pipeline; run_daily_pipeline('VHM')"
```

## Test

```bash
python -m compileall main.py src scripts
python -m unittest discover -s tests
```
