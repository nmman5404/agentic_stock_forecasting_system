# Quant Risk Committee Report: VIC (2026-05-12)

## 1. Executive Summary
- Model status: **ABNORMAL**
- Final research signal: **MANUAL_REVIEW**
- Signal confidence: **0.44**
- Risk level: **EXTREME**
- News evidence: **NONE**

## 2. Forecast Performance
- Holdout MAE: 5,097.4997
- Holdout RMSE: 6,545.8669
- Holdout MAPE: 2.81%
- Evaluation reason: Holdout MAPE 0.0281 exceeds threshold 0.0100.

## 3. Walk-forward Validation
- MAE: 4,025.9632
- RMSE: 5,384.1549
- MAPE: 2.85%
- SMAPE: 2.85%
- Directional accuracy: 54.72%
- 95% interval coverage: 84.28%
- Pinball loss: 956.9847
- Prediction bias: -581.6494

## 4. Market Regime
- Volatility regime: **NORMAL_VOLATILITY**
- Trend regime: **UPTREND**
- Liquidity regime: **NORMAL_LIQUIDITY**
- Confidence: 0.63
- Notes: 20-day volatility percentile=0.68. MA7/MA21 gap=0.1083; 20-day return=0.4879; 5-day return=0.0114. Volume z-score=-0.16; recent volume ratio=0.92.

## 5. Drift Detection
- Feature drift detected: **True**
- Target drift detected: **False**
- Concept drift detected: **False**
- Severity: **HIGH**
- Recommended action: **MANUAL_REVIEW**
- Notes: Target return mean_shift_z=0.18; volatility_ratio=1.41. Validation snapshot: MAPE=0.0285; directional_accuracy=0.5472; interval_95_coverage=0.8428. Feature drift detected in 14 monitored features.

## 6. Risk Assessment
- Expected return 7d: -0.67%
- Downside risk 95%: -17.72%
- Upside potential 95%: 23.25%
- Risk/reward ratio: 1.3120
- VaR 95%: 17.72%
- Expected shortfall: 20.38%
- Preliminary signal: **MANUAL_REVIEW**
- Risk notes: High drift severity detected.

## 7. News & Event Context
- News found: **False**
- News items count: **0**
- Evidence level: **NONE**
- Shock classification: **NO_NEWS**

```text
NO_NEWS
```

## 8. Model Governance Decision
- Decision: **KEEP_CHAMPION**
- Accepted challenger: **False**
- Reason: Challenger rejected because one or more governance gates failed.
- Action taken: Adjusted config for underfitting risk: max_depth=3, learning_rate=0.040, n_estimators=70 | KEEP_CHAMPION | reason=Challenger rejected because one or more governance gates failed.

## 9. Final Research Signal
```text
Assessment:
Model status is ABNORMAL with HIGH severity feature drift detected across 14 features, including price-based indicators (close, MA7, MA14, ATR14) showing significant PSI values (>7.0).

Interpretation:
The model is experiencing significant input distribution shifts, leading to an 'EXTREME' risk level. While the current champion model maintains a 0.84 interval coverage, the directional accuracy is low (0.547) and the model is currently biased toward under-prediction (-581.65).

Limitations:
High feature drift suggests the current model parameters are no longer representative of the underlying data generating process. Governance gates for challenger replacement failed, leaving the system in a state of high uncertainty.

Final research signal:
MANUAL_REVIEW
Confidence: 0.44
Reason: Manual review required due to high drift severity, extreme risk metrics, and failure of the challenger model to outperform the champion under current market conditions.
```

## 10. Audit Trail
- `2026-05-12T16:53:29Z` | pipeline_initialization | **PASS** | Initial forecast, monitoring, and risk reports prepared.
- `2026-05-12T16:53:29Z` | validate_quantiles | **PASS** | Quantile crossing fixed=False.
- `2026-05-12T16:53:29Z` | evaluate_model | **ABNORMAL** | Holdout MAPE 0.0281 exceeds threshold 0.0100.
- `2026-05-12T16:53:29Z` | contextualize_news | **NO_NEWS** | Forecast degradation is likely model-related or caused by short-term market noise. There is insufficient evidence to attribute it to a specific external event.
- `2026-05-12T16:53:29Z` | model_governance | **KEEP_CHAMPION** | Challenger rejected because one or more governance gates failed.
- `2026-05-12T16:53:29Z` | validate_quantiles | **PASS** | Quantile crossing fixed=False.
- `2026-05-12T16:53:29Z` | evaluate_model | **ABNORMAL** | Holdout MAPE 0.0281 exceeds threshold 0.0100.
- `2026-05-12T16:53:30Z` | model_governance | **KEEP_CHAMPION** | Challenger rejected because one or more governance gates failed.
- `2026-05-12T16:53:35Z` | research_signal | **MANUAL_REVIEW** | Final research signal=MANUAL_REVIEW; confidence=0.44.
