# Quant Risk Committee Report: VIC (2026-05-13)

## 1. Executive Summary
- Model status: **PASS**
- Final research signal: **MANUAL_REVIEW**
- Signal confidence: **0.44**
- Risk level: **EXTREME**
- News evidence: **NONE**

## 2. Forecast Performance
- Holdout MAE: 5,097.4997
- Holdout RMSE: 6,545.8669
- Holdout MAPE: 2.81%
- Evaluation reason: Holdout MAPE 0.0281 is within threshold 0.0300.

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
- Decision: **N/A**
- Accepted challenger: **False**
- Reason: N/A
- Action taken: N/A

## 9. Committee Assessment
### Assessment
The model for VIC currently maintains a PASS status on performance metrics; however, the risk engine has issued a MANUAL_REVIEW signal due to significant feature drift. This should be treated as a risk-control signal, not a directional trading conviction.

### Interpretation
- Forecast performance: The model remains within performance thresholds with a holdout MAPE of 0.0281, which is below the 0.0300 limit, indicating acceptable predictive accuracy on historical holdout data.
- Validation reliability: Walk-forward validation across 8 folds shows a MAPE of 0.0285 and a directional accuracy of 54.72%. The 95% interval coverage of 84.28% suggests the model's uncertainty estimates may be slightly optimistic.
- Risk profile: The 7-day median forecast is 220,520.7336 with a 95% interval width of 90,956.9874 (40.97% of current price). Expected return is -0.67%, downside risk is -17.72%, upside potential is 23.25%, VaR 95 is 17.72%, Expected Shortfall is 20.38%, and risk/reward is 1.3120. The risk engine reports EXTREME risk with signal confidence 0.4400.
- Market regime: The market is characterized by an uptrend and normal volatility/liquidity. The regime confidence is 63%, supported by a 20-day return of 48.79% and a moderate MA7/MA21 gap.
- Drift condition: High drift severity is detected. 14 features, including 'close', 'ma_7', 'ma_14', and 'atr_14', show significant PSI values and mean shifts, indicating that the current market data distribution has diverged from the training distribution.
- News/context evidence: No relevant news evidence was found. shock_type=NO_NEWS and evidence_level=NONE; there is insufficient evidence to attribute this to a specific news event.
- Governance/retrain status: No retrain attempts have been initiated; the model status remains PASS, but the drift-induced MANUAL_REVIEW signal necessitates human oversight.

### Decision Rationale
The conflict between the PASS model status and the MANUAL_REVIEW risk signal arises because the model's internal performance metrics (MAPE) remain stable, while the input feature space (drift detection) has undergone a structural shift. The high PSI values for price-based features suggest that the model's current inputs are statistically distinct from those used during training, rendering the model's output unreliable despite its historical accuracy.

### Final Recommendation
MANUAL_REVIEW: The model should not be utilized for automated decision-making until the high feature drift is investigated and the model is either recalibrated or validated against the new data regime. This is research output only and not financial advice.

### Evidence Used
holdout_metrics, walk_forward_validation, forecast_distribution, risk_report, regime_report, drift_report, news_context, governance_status

## 10. Audit Trail
- `2026-05-13T06:24:21Z` | pipeline_initialization | **PASS** | Initial forecast, monitoring, and risk reports prepared.
- `2026-05-13T06:24:21Z` | validate_quantiles | **PASS** | Quantile crossing fixed=False.
- `2026-05-13T06:24:21Z` | evaluate_model | **PASS** | Holdout MAPE 0.0281 is within threshold 0.0300.
- `2026-05-13T06:25:24Z` | research_signal | **MANUAL_REVIEW** | Final research signal=MANUAL_REVIEW; confidence=0.44.
