# Quant Research Report: VIC (2026-05-15)

## Executive Summary
- Final run: **champion**
- Final research action: **WATCH**
- Confidence: **0.4500**
- Model health: **NEEDS_REVIEW**
- Risk level: **HIGH_RISK**
- Drift label: **FEATURE_HIGH__TARGET_LOW__CONCEPT_MEDIUM**
- News status: **NEWS_FOUND_GOOGLE_RSS**

## Forecast
- Current price: 228.0000
- Expected return: 0.96%
- Downside risk 95%: -17.57%
- Upside potential 95%: 24.14%
- Risk/reward ratio: 1.3743

## Walk-forward Validation
- Evaluation method: **walk_forward**
- Fold count: **8**
- MAE: 4.2565
- RMSE: 5.6032
- MAPE: 2.89%
- SMAPE: 2.89%
- Directional accuracy: 53.42%
- 95% interval coverage: 83.56%
- Pinball loss: 1.0085

## Monitoring
- Regime label: **NORMAL_VOLATILITY__UPTREND__NORMAL_VOLUME**
- Volatility regime: **NORMAL_VOLATILITY**
- Trend regime: **UPTREND**
- Volume regime: **NORMAL_VOLUME**
- Drift levels feature/target/concept: **HIGH / LOW / MEDIUM**
- Final drift label: **FEATURE_HIGH__TARGET_LOW__CONCEPT_MEDIUM**

## Agent Improvement Plan
- Gemini decision: **TRAIN_CANDIDATE**
- Technical retrain required: **True**
- Technical retrain reasons: Risk level is HIGH_RISK.
- Proposed config: `{'learning_rate': 0.01, 'max_depth': 5, 'num_leaves': 32, 'min_child_samples': 50}`
- Config patch validation: **VALID**
- Retrain attempted: **True**
- Config decision: **KEEP_CURRENT_CONFIG**
- Candidate config accepted: **False**
- Config saved: **False**
- Final run: **champion**
- Reason: Candidate was not better on primary walk-forward metrics.

## News Context
- Evidence level: **HIGH**
- Shock type: **EVENT_DRIVEN**

```text
[Thu, 14 May 2026 13:24:00 GMT] Vingroup (VIC) tất toán hàng ngàn tỷ đồng trái phiếu, lấn sân mảng thiết bị y tế công nghệ cao - Báo Pháp Luật Việt Nam
Summary: Vingroup (VIC) tất toán hàng ngàn tỷ đồng trái phiếu, lấn sân mảng thiết bị y tế công nghệ cao  Báo Pháp Luật Việt Nam
Publisher: Báo Pháp Luật Việt Nam
Matched keywords: VIC, Vingroup

[Thu, 14 May 2026 11:17:06 GMT] Vingroup (VIC) tất toán 2 lô trái phiếu 4.000 tỷ đồng - Nhịp sống nhà đất
Summary: Vingroup (VIC) tất toán 2 lô trái phiếu 4.000 tỷ đồng  Nhịp sống nhà đất
Publisher: Nhịp sống nhà đất
Matched keywords: VIC, Vingroup

[Thu, 14 May 2026 15:10:00 GMT] Cổ phiếu họ Vingroup kéo VN-Index lập đỉnh kỷ lục - cafeland.vn
Summary: Cổ phiếu họ Vingroup kéo VN-Index lập đỉnh kỷ lục  cafeland.vn
Publisher: cafeland.vn
Matched keywords: Vingroup

[Thu, 14 May 2026 10:25:14 GMT] Chứng khoán tăng 27 điểm, lực đẩy nhờ nhóm cổ phiếu Vingroup - Báo Dân trí
Summary: Chứng khoán tăng 27 điểm, lực đẩy nhờ nhóm cổ phiếu Vingroup  Báo Dân trí
Publisher: Báo Dân trí
Matched keywords: Vingroup

[Thu, 14 May 2026 01:03:50 GMT] Vingroup (VIC) lập thêm công ty mới - 24HMoney
Summary: Vingroup (VIC) lập thêm công ty mới  24HMoney
Publisher: 24HMoney
Matched keywords: VIC, Vingroup
```

## Config Decision
- Decision: **KEEP_CURRENT_CONFIG**
- Final run: **champion**
- Candidate config accepted: **False**
- Config saved: **False**
- Reason: Candidate was not better on primary walk-forward metrics.

## Final Recommendation
Final research action is WATCH. Final run=champion; risk=HIGH_RISK; drift=FEATURE_HIGH__TARGET_LOW__CONCEPT_MEDIUM; config_decision=KEEP_CURRENT_CONFIG. This is research output only, intended for paper-trading review and not financial advice.

## Audit Trail
- `2026-05-15T07:58:56Z` | pipeline_initialization | **PASS** | Initial champion forecast prepared.
- `2026-05-15T07:58:56Z` | validate_forecast | **PASS** | Forecast validated; quantile crossing fixed=False.
- `2026-05-15T07:58:56Z` | evaluate_monitoring | **PASS** | Monitoring completed; health=NEEDS_REVIEW.
- `2026-05-15T07:58:59Z` | search_news_context | **PASS** | News context status=NEWS_FOUND_GOOGLE_RSS.
- `2026-05-15T07:59:01Z` | plan_retrain | **PASS** | High feature drift and event-driven news suggest a need for higher regularization to prevent overfitting to recent noise; lower learning rate and increased min_child_samples improve generalization under volatile conditions.
- `2026-05-15T07:59:01Z` | validate_config_patch | **PASS** | Config proposal valid.
- `2026-05-15T07:59:02Z` | train_candidate | **PASS** | Candidate model trained.
- `2026-05-15T07:59:02Z` | evaluate_candidate | **PASS** | Candidate monitoring completed.
- `2026-05-15T07:59:02Z` | compare_metrics | **KEEP_CURRENT_CONFIG** | Candidate was not better on primary walk-forward metrics.
- `2026-05-15T07:59:02Z` | save_or_reject_config | **KEEP_CURRENT_CONFIG** | Candidate was not better on primary walk-forward metrics.
- `2026-05-15T07:59:02Z` | final_recommendation | **WATCH** | High risk, drift, or news context requires watch state.
