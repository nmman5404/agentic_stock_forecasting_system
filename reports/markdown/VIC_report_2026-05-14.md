# Quant Research Report: VIC (2026-05-14)

## Executive Summary
- Final model: **champion**
- Final research action: **MANUAL_REVIEW**
- Confidence: **0.5600**
- Accuracy check: **ACCURACY_OK**
- Walk-forward reliability: **DEGRADED**
- Risk clearance: **MANUAL_REVIEW_REQUIRED**
- Overall trust status: **REQUIRES_MANUAL_REVIEW**
- Risk level: **EXTREME_RISK**
- Drift severity: **HIGH**
- News status: **NEWS_FOUND_GOOGLE_NEWS**

## Forecast
- Current price: 229.8000
- Expected return: -0.57%
- Downside risk 95%: -17.53%
- Upside potential 95%: 25.04%
- Risk/reward ratio: 1.4285

## Walk-forward Validation
- Evaluation method: **walk_forward**
- Fold count: **8**
- MAE: 4.2536
- RMSE: 5.5747
- MAPE: 2.90%
- SMAPE: 2.90%
- Directional accuracy: 49.66%
- 95% interval coverage: 80.95%
- Pinball loss: 1.0003

## Monitoring
- Regime label: **NORMAL_VOLATILITY__UPTREND__NORMAL_VOLUME**
- Volatility regime: **NORMAL_VOLATILITY**
- Trend regime: **UPTREND**
- Volume regime: **NORMAL_VOLUME**
- Drift feature/target/concept scores: **82 / 1 / 2**
- Drift recommended action: **MANUAL_REVIEW**

## Agent Improvement Plan
- Agent diagnosis: **FEATURE_DRIFT_AND_UNCERTAINTY_UNDER_COVERAGE**
- Agent decision: **TRAIN_CHALLENGER**
- Technical retrain required: **True**
- Technical retrain reasons: Directional accuracy 0.4966 below minimum 0.5000. 95% interval coverage 0.8095 below minimum 0.8500. High drift severity with concept-drift evidence. Risk level is EXTREME_RISK with drift severity HIGH.
- Technical retrain strategy: **RETRAIN_RECENT_WINDOW**
- Config patch source: **AGENT_REPAIRED**
- Config patch validation: **VALID**
- Retrain attempted: **True**
- Governance decision: **KEEP_CHAMPION**
- Final model: **champion**
- Reason: Challenger rejected because governance gates failed: mape_materially_improves, challenger_interval_95_coverage_meets_minimum, both_poor_guardrail.

## News Context
- Google News used: **True**
- Evidence level: **HIGH**
- Shock type: **EVENT_DRIVEN**
- Raw/matched items: **39 / 10**
- Debug path: `data\news_debug\VIC_news_debug_2026-05-14_ad591d37.json`

```text
[Thu, 14 May 2026 01:11:44 GMT] Vingroup (VIC) lập thêm công ty mới trong lĩnh vực robot phẫu thuật - Nhịp sống nhà đất
Summary: Vingroup (VIC) lập thêm công ty mới trong lĩnh vực robot phẫu thuật  Nhịp sống nhà đất
Source: Google News RSS
Matched keywords: VIC, Vingroup, VinGroup

[Thu, 14 May 2026 02:43:12 GMT] Vingroup lập công ty nghiên cứu robot phẫu thuật - Mekong ASEAN
Summary: Vingroup lập công ty nghiên cứu robot phẫu thuật  Mekong ASEAN
Source: Google News RSS
Matched keywords: Vingroup, VinGroup

[Thu, 14 May 2026 01:03:50 GMT] Vingroup (VIC) lập thêm công ty mới - 24HMoney
Summary: Vingroup (VIC) lập thêm công ty mới  24HMoney
Source: Google News RSS
Matched keywords: VIC, Vingroup, VinGroup

[Thu, 14 May 2026 06:32:00 GMT] Vingroup thanh toán hơn 2.060 tỷ đồng gốc, lãi trái phiếu - Chứng khoán DNSE
Summary: Vingroup thanh toán hơn 2.060 tỷ đồng gốc, lãi trái phiếu  Chứng khoán DNSE
Source: Google News RSS
Matched keywords: Vingroup, VinGroup

[Wed, 13 May 2026 09:35:00 GMT] Chứng khoán phiên 13/5: Cổ phiếu nhóm Vingroup biến động mạnh, kéo giảm VN-Index - baodautu
Summary: Chứng khoán phiên 13/5: Cổ phiếu nhóm Vingroup biến động mạnh, kéo giảm VN-Index  baodautu
Source: Google News RSS
Matched keywords: Vingroup, VinGroup
```

## Governance
- Decision: **KEEP_CHAMPION**
- Final model: **champion**
- Accepted challenger: **False**
- Reason: Challenger rejected because governance gates failed: mape_materially_improves, challenger_interval_95_coverage_meets_minimum, both_poor_guardrail.

## Final Recommendation
Final research action is MANUAL_REVIEW. Final model=champion; risk_level=EXTREME_RISK; drift_severity=HIGH; governance_decision=KEEP_CHAMPION. This is research output only, intended for paper-trading review and not financial advice.

## Audit Trail
- `2026-05-14T09:00:44Z` | pipeline_initialization | **PASS** | Initial champion forecast prepared.
- `2026-05-14T09:00:44Z` | validate_forecast | **PASS** | Champion forecast validated; quantile crossing fixed=False.
- `2026-05-14T09:00:44Z` | evaluate_monitoring | **PASS** | Champion walk-forward, drift, regime, and risk monitoring completed.
- `2026-05-14T09:00:48Z` | search_news_context | **PASS** | Google News context search completed with status=NEWS_FOUND_GOOGLE_NEWS.
- `2026-05-14T09:00:55Z` | technical_retrain_check | **PASS** | Technical retrain required=True because Directional accuracy 0.4966 below minimum 0.5000.; 95% interval coverage 0.8095 below minimum 0.8500.; High drift severity with concept-drift evidence.; Risk level is EXTREME_RISK with drift severity HIGH.
- `2026-05-14T09:00:55Z` | plan_retrain | **PASS** | The model exhibits high feature drift (score 85) and significant news-driven volatility regarding Vingroup (VIC). While accuracy metrics are currently within bounds, the combination of high drift, under-coverage of uncertainty intervals, and recent corporate news events necessitates a manual review before authorizing a retrain to ensure the model adapts to the new structural environment.
- `2026-05-14T09:00:58Z` | validate_config_patch | **PASS** | Config patch validation completed successfully.
- `2026-05-14T09:01:02Z` | train_challenger | **PASS** | Challenger model trained successfully.
- `2026-05-14T09:01:02Z` | evaluate_challenger | **PASS** | Challenger monitoring completed.
- `2026-05-14T09:01:02Z` | governance_review | **KEEP_CHAMPION** | Challenger rejected because governance gates failed: mape_materially_improves, challenger_interval_95_coverage_meets_minimum, both_poor_guardrail.
- `2026-05-14T09:01:02Z` | final_recommendation | **MANUAL_REVIEW** | Extreme forecast risk requires manual review.
