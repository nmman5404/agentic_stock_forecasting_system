# Quant Research Report: VHM (2026-05-14)

## Executive Summary
- Final model: **champion**
- Final research action: **MANUAL_REVIEW**
- Confidence: **0.4800**
- Accuracy check: **ACCURACY_DEGRADED**
- Walk-forward reliability: **DEGRADED**
- Risk clearance: **MANUAL_REVIEW_REQUIRED**
- Overall trust status: **REQUIRES_MANUAL_REVIEW**
- Risk level: **EXTREME_RISK**
- Drift severity: **HIGH**
- News status: **NEWS_FOUND_GOOGLE_NEWS**

## Forecast
- Current price: 157.0000
- Expected return: -0.21%
- Downside risk 95%: -18.29%
- Upside potential 95%: 21.37%
- Risk/reward ratio: 1.1682

## Walk-forward Validation
- Evaluation method: **walk_forward**
- Fold count: **8**
- MAE: 3.5477
- RMSE: 4.6105
- MAPE: 3.07%
- SMAPE: 3.07%
- Directional accuracy: 42.86%
- 95% interval coverage: 80.27%
- Pinball loss: 0.8442

## Monitoring
- Regime label: **EXTREME_VOLATILITY__UPTREND__NORMAL_VOLUME**
- Volatility regime: **EXTREME_VOLATILITY**
- Trend regime: **UPTREND**
- Volume regime: **NORMAL_VOLUME**
- Drift feature/target/concept scores: **71 / 1 / 4**
- Drift recommended action: **MANUAL_REVIEW**

## Agent Improvement Plan
- Agent diagnosis: **FEATURE_DRIFT, CONCEPT_DRIFT, REGIME_SHIFT, NEWS_EVENT_DRIVEN, INTERVAL_UNDERCOVERAGE**
- Agent decision: **TRAIN_CHALLENGER**
- Technical retrain required: **True**
- Technical retrain reasons: Walk-forward MAPE 0.0307 exceeded threshold 0.0300. Directional accuracy 0.4286 below minimum 0.5000. 95% interval coverage 0.8027 below minimum 0.8500. High drift severity with concept-drift evidence. Risk level is EXTREME_RISK with drift severity HIGH.
- Technical retrain strategy: **RETRAIN_RECENT_WINDOW**
- Config patch source: **AGENT_PROPOSED**
- Config patch validation: **VALID**
- Retrain attempted: **True**
- Governance decision: **MANUAL_REVIEW_AFTER_RETRAIN**
- Final model: **champion**
- Reason: Challenger rejected because governance gates failed: mape_improves, mape_materially_improves, mape_within_allowed_degradation, challenger_mape_within_threshold, pinball_loss_ok, both_poor_guardrail. Both champion and challenger remain below reliability gates.

## News Context
- Google News used: **True**
- Evidence level: **HIGH**
- Shock type: **EVENT_DRIVEN**
- Raw/matched items: **7 / 7**
- Debug path: `data\news_debug\VHM_news_debug_2026-05-14_5056c923.json`

```text
[Wed, 13 May 2026 07:49:42 GMT] Vinhomes (VHM) tung liên tiếp 2 lô trái phiếu 3.000 tỷ: Cỗ máy huy động vốn hoạt động hết công suất, tham vọng lớn đang dần lộ diện - Nhịp sống nhà đất
Summary: Vinhomes (VHM) tung liên tiếp 2 lô trái phiếu 3.000 tỷ: Cỗ máy huy động vốn hoạt động hết công suất, tham vọng lớn đang dần lộ diện  Nhịp sống nhà đất
Source: Google News RSS
Matched keywords: VHM, Vinhomes

[Sun, 10 May 2026 03:41:00 GMT] Vinhomes (VHM) phát hành 5000 tỷ đồng trái phiếu, lãi quý I tăng gấp 10 lần - Báo Pháp Luật Việt Nam
Summary: Vinhomes (VHM) phát hành 5000 tỷ đồng trái phiếu, lãi quý I tăng gấp 10 lần  Báo Pháp Luật Việt Nam
Source: Google News RSS
Matched keywords: VHM, Vinhomes

[Wed, 13 May 2026 05:35:00 GMT] Vinhomes phát hành 2 lô trái phiếu, huy động 3.000 tỷ đồng - Nhịp sống kinh doanh
Summary: Vinhomes phát hành 2 lô trái phiếu, huy động 3.000 tỷ đồng  Nhịp sống kinh doanh
Source: Google News RSS
Matched keywords: Vinhomes

[Fri, 08 May 2026 13:20:00 GMT] Vinhomes (VHM) chuẩn bị khởi công siêu dự án 23.600 tỷ đồng trong vài ngày tới - 24HMoney
Summary: Vinhomes (VHM) chuẩn bị khởi công siêu dự án 23.600 tỷ đồng trong vài ngày tới  24HMoney
Source: Google News RSS
Matched keywords: VHM, Vinhomes

[Sat, 09 May 2026 00:44:00 GMT] Vinhomes dự kiến chào bán 5.000 tỷ đồng trái phiếu riêng lẻ - Chứng khoán DNSE
Summary: Vinhomes dự kiến chào bán 5.000 tỷ đồng trái phiếu riêng lẻ  Chứng khoán DNSE
Source: Google News RSS
Matched keywords: Vinhomes
```

## Governance
- Decision: **MANUAL_REVIEW_AFTER_RETRAIN**
- Final model: **champion**
- Accepted challenger: **False**
- Reason: Challenger rejected because governance gates failed: mape_improves, mape_materially_improves, mape_within_allowed_degradation, challenger_mape_within_threshold, pinball_loss_ok, both_poor_guardrail. Both champion and challenger remain below reliability gates.

## Final Recommendation
Final research action is MANUAL_REVIEW. Final model=champion; risk_level=EXTREME_RISK; drift_severity=HIGH; governance_decision=MANUAL_REVIEW_AFTER_RETRAIN. This is research output only, intended for paper-trading review and not financial advice.

## Audit Trail
- `2026-05-14T08:58:40Z` | pipeline_initialization | **PASS** | Initial champion forecast prepared.
- `2026-05-14T08:58:40Z` | validate_forecast | **PASS** | Champion forecast validated; quantile crossing fixed=False.
- `2026-05-14T08:58:40Z` | evaluate_monitoring | **PASS** | Champion walk-forward, drift, regime, and risk monitoring completed.
- `2026-05-14T08:58:42Z` | search_news_context | **PASS** | Google News context search completed with status=NEWS_FOUND_GOOGLE_NEWS.
- `2026-05-14T08:58:58Z` | technical_retrain_check | **PASS** | Technical retrain required=True because Walk-forward MAPE 0.0307 exceeded threshold 0.0300.; Directional accuracy 0.4286 below minimum 0.5000.; 95% interval coverage 0.8027 below minimum 0.8500.; High drift severity with concept-drift evidence.; Risk level is EXTREME_RISK with drift severity HIGH.
- `2026-05-14T08:58:58Z` | plan_retrain | **PASS** | The model is experiencing high feature drift (score 71) and concept drift, compounded by an extreme volatility regime and significant news-driven events (Vinhomes bond issuance). Current performance shows degraded accuracy and interval under-coverage. A retrain using a shorter, more recent window (252 days) with increased regularization is required to adapt to the new market regime.
- `2026-05-14T08:58:58Z` | validate_config_patch | **PASS** | Config patch validation completed successfully.
- `2026-05-14T08:58:58Z` | train_challenger | **PASS** | Challenger model trained successfully.
- `2026-05-14T08:58:58Z` | evaluate_challenger | **PASS** | Challenger monitoring completed.
- `2026-05-14T08:58:58Z` | governance_review | **MANUAL_REVIEW_AFTER_RETRAIN** | Challenger rejected because governance gates failed: mape_improves, mape_materially_improves, mape_within_allowed_degradation, challenger_mape_within_threshold, pinball_loss_ok, both_poor_guardrail. Both champion and challenger remain below reliability gates.
- `2026-05-14T08:58:58Z` | final_recommendation | **MANUAL_REVIEW** | Challenger rejected because governance gates failed: mape_improves, mape_materially_improves, mape_within_allowed_degradation, challenger_mape_within_threshold, pinball_loss_ok, both_poor_guardrail. Both champion and challenger remain below reliability gates.
