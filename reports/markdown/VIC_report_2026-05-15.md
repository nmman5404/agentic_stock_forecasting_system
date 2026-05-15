# Quant Research Report: VIC (2026-05-15)

## Executive Summary
- Final model: **champion**
- Final research action: **MANUAL_REVIEW**
- Confidence: **0.6000**
- Accuracy check: **ACCURACY_OK**
- Walk-forward reliability: **WEAK**
- Risk clearance: **MANUAL_REVIEW_REQUIRED**
- Overall trust status: **REQUIRES_MANUAL_REVIEW**
- Risk level: **EXTREME_RISK**
- Drift severity: **HIGH**
- News status: **NEWS_FOUND_GOOGLE_NEWS**

## Forecast
- Current price: 227.0000
- Expected return: 0.96%
- Downside risk 95%: -17.57%
- Upside potential 95%: 24.14%
- Risk/reward ratio: 1.3743

## Walk-forward Validation
- Evaluation method: **walk_forward**
- Fold count: **8**
- MAE: 4.2633
- RMSE: 5.6069
- MAPE: 2.89%
- SMAPE: 2.89%
- Directional accuracy: 53.42%
- 95% interval coverage: 83.56%
- Pinball loss: 1.0092

## Monitoring
- Regime label: **NORMAL_VOLATILITY__UPTREND__NORMAL_VOLUME**
- Volatility regime: **NORMAL_VOLATILITY**
- Trend regime: **UPTREND**
- Volume regime: **NORMAL_VOLUME**
- Drift feature/target/concept scores: **82 / 1 / 2**
- Drift recommended action: **MANUAL_REVIEW**

## Agent Improvement Plan
- Agent diagnosis: **FEATURE_DRIFT, NEWS_EVENT_DRIVEN, INTERVAL_UNDERCOVERAGE**
- Agent decision: **TRAIN_CHALLENGER**
- Technical retrain required: **True**
- Technical retrain reasons: 95% interval coverage 0.8356 below minimum 0.8500. High drift severity with concept-drift evidence. Risk level is EXTREME_RISK with drift severity HIGH.
- Technical retrain strategy: **RETRAIN_RECENT_WINDOW**
- Config patch source: **AGENT_PROPOSED**
- Config patch validation: **VALID**
- Retrain attempted: **True**
- Governance decision: **KEEP_CHAMPION**
- Final model: **champion**
- Reason: Challenger rejected because governance gates failed: directional_accuracy_ok, pinball_loss_ok.

## News Context
- Google News used: **True**
- Evidence level: **HIGH**
- Shock type: **EVENT_DRIVEN**
- Raw/matched items: **58 / 10**
- Debug path: `data\news_debug\VIC_news_debug_2026-05-15_3ef9fa87.json`

```text
[Thu, 14 May 2026 13:24:00 GMT] Vingroup (VIC) tất toán hàng ngàn tỷ đồng trái phiếu, lấn sân mảng thiết bị y tế công nghệ cao - Báo Pháp Luật Việt Nam
Summary: Vingroup (VIC) tất toán hàng ngàn tỷ đồng trái phiếu, lấn sân mảng thiết bị y tế công nghệ cao  Báo Pháp Luật Việt Nam
Source: Google News RSS
Matched keywords: VIC, Vingroup, VinGroup

[Thu, 14 May 2026 11:17:06 GMT] Vingroup (VIC) tất toán 2 lô trái phiếu 4.000 tỷ đồng - Nhịp sống nhà đất
Summary: Vingroup (VIC) tất toán 2 lô trái phiếu 4.000 tỷ đồng  Nhịp sống nhà đất
Source: Google News RSS
Matched keywords: VIC, Vingroup, VinGroup

[Thu, 14 May 2026 15:10:00 GMT] Cổ phiếu họ Vingroup kéo VN-Index lập đỉnh kỷ lục - cafeland.vn
Summary: Cổ phiếu họ Vingroup kéo VN-Index lập đỉnh kỷ lục  cafeland.vn
Source: Google News RSS
Matched keywords: Vingroup, VinGroup, họ Vingroup, ho Vingroup

[Thu, 14 May 2026 07:58:00 GMT] Nước cờ tái cấu trúc của VinFast: Gánh nặng trăm ngàn tỷ được trút bỏ, bức tranh tài chính Vingroup sẽ bừng sáng? - CafeF
Summary: Nước cờ tái cấu trúc của VinFast: Gánh nặng trăm ngàn tỷ được trút bỏ, bức tranh tài chính Vingroup sẽ bừng sáng?  CafeF
Source: Google News RSS
Matched keywords: Vingroup, VinGroup, VinFast

[Thu, 14 May 2026 10:25:14 GMT] Chứng khoán tăng 27 điểm, lực đẩy nhờ nhóm cổ phiếu Vingroup - Báo Dân trí
Summary: Chứng khoán tăng 27 điểm, lực đẩy nhờ nhóm cổ phiếu Vingroup  Báo Dân trí
Source: Google News RSS
Matched keywords: Vingroup, VinGroup
```

## Governance
- Decision: **KEEP_CHAMPION**
- Final model: **champion**
- Accepted challenger: **False**
- Reason: Challenger rejected because governance gates failed: directional_accuracy_ok, pinball_loss_ok.

## Final Recommendation
Final research action is MANUAL_REVIEW. Final model=champion; risk_level=EXTREME_RISK; drift_severity=HIGH; governance_decision=KEEP_CHAMPION. This is research output only, intended for paper-trading review and not financial advice.

## Audit Trail
- `2026-05-15T05:44:48Z` | pipeline_initialization | **PASS** | Initial champion forecast prepared.
- `2026-05-15T05:44:48Z` | validate_forecast | **PASS** | Champion forecast validated; quantile crossing fixed=False.
- `2026-05-15T05:44:48Z` | evaluate_monitoring | **PASS** | Champion walk-forward, drift, regime, and risk monitoring completed.
- `2026-05-15T05:44:51Z` | search_news_context | **PASS** | Google News context search completed with status=NEWS_FOUND_GOOGLE_NEWS.
- `2026-05-15T05:44:54Z` | technical_retrain_check | **PASS** | Technical retrain required=True because 95% interval coverage 0.8356 below minimum 0.8500.; High drift severity with concept-drift evidence.; Risk level is EXTREME_RISK with drift severity HIGH.
- `2026-05-15T05:44:54Z` | plan_retrain | **PASS** | The model exhibits high feature drift (score 85) and interval under-coverage (83.56% vs 85% target). Significant news events regarding Vingroup's restructuring and bond settlement suggest a structural shift in the data generating process, necessitating a retrain on a more recent, relevant window to capture current market dynamics.
- `2026-05-15T05:44:54Z` | validate_config_patch | **PASS** | Config patch validation completed successfully.
- `2026-05-15T05:44:54Z` | train_challenger | **PASS** | Challenger model trained successfully.
- `2026-05-15T05:44:54Z` | evaluate_challenger | **PASS** | Challenger monitoring completed.
- `2026-05-15T05:44:54Z` | governance_review | **KEEP_CHAMPION** | Challenger rejected because governance gates failed: directional_accuracy_ok, pinball_loss_ok.
- `2026-05-15T05:44:54Z` | final_recommendation | **MANUAL_REVIEW** | Extreme forecast risk requires manual review.
