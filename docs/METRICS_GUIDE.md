# Metrics Guide

## Walk-forward Validation

Walk-forward validation is the model evaluation source of truth. Important fields:

- `mape`: average absolute percentage error.
- `rmse`: root mean squared price error.
- `mae`: mean absolute price error.
- `smape`: symmetric percentage error.
- `directional_accuracy`: agreement between predicted and realized direction.
- `interval_80_coverage`: realized prices inside the 80% prediction interval.
- `interval_95_coverage`: realized prices inside the 95% prediction interval.
- `pinball_loss`: quantile loss across forecast quantiles.
- `prediction_bias_pct`: average signed percentage error.
- `quantile_crossing_rate`: fraction of rows whose raw quantiles crossed before sorting.

## Drift Metrics

Drift detector produces evidence levels:

```python
EVIDENCE_SCORE = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}
```

It reports:

- feature drift score
- target drift score
- concept drift score
- total score
- severity: `LOW`, `MEDIUM`, `HIGH`
- recommended action: `MONITOR`, `REVALIDATE`, `MANUAL_REVIEW`

## Regime Metrics

Regime detector reports three independent components:

- volatility regime: `LOW_VOLATILITY`, `NORMAL_VOLATILITY`, `HIGH_VOLATILITY`, `EXTREME_VOLATILITY`
- trend regime: `UPTREND`, `DOWNTREND`, `SIDEWAYS`, `MIXED_TREND`
- volume regime: `VOLUME_SPIKE`, `LOW_VOLUME`, `NORMAL_VOLUME`

Final label:

```python
final_regime_label = f"{volatility_regime}__{trend_regime}__{volume_regime}"
```

## Risk Metrics

Risk engine reports:

- `expected_return`
- `downside_risk_95`
- `upside_potential_95`
- `risk_reward_ratio`
- `var_95`
- `expected_shortfall`
- `risk_level`

Risk levels:

```text
LOW_RISK / MEDIUM_RISK / HIGH_RISK / EXTREME_RISK
```

## Governance Metrics

Champion and challenger are compared using walk-forward metric bundles:

- MAPE
- RMSE
- MAE
- directional accuracy
- interval 95 coverage
- pinball loss
- prediction bias percent
- risk level
- drift severity
