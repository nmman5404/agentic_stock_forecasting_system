# Pipeline Walkthrough

## Mục tiêu

Pipeline tạo forecast nghiên cứu cho ticker, kiểm tra độ tin cậy bằng walk-forward validation, đo drift/regime/risk, sau đó dùng LangGraph để quyết định có cần train challenger hay không.

## Luồng chạy

```text
1. Ingest dữ liệu thị trường
2. Lưu raw và processed data vào SQLite
3. Train LightGBM Quantile forecast 7 ngày
4. Validate forecast quantiles
5. Đánh giá walk-forward
6. Detect drift
7. Detect regime
8. Measure risk
9. Nếu model OK: sinh report
10. Nếu degraded: tìm Google News RSS context
11. Agent đề xuất retrain plan và config_patch
12. Validator kiểm tra hoặc repair patch
13. Nếu patch valid: train challenger
14. Evaluate challenger
15. Governance so sánh champion/challenger
16. Recommendation node sinh final research action
17. Ghi 4 output report
```

## AgentState mới

State được gom theo nhóm:

- `workflow`
- `champion`
- `challenger`
- `news`
- `improvement`
- `governance`
- `recommendation`
- `audit`

## Nguyên tắc quyết định

- LLM chỉ đề xuất diagnosis, strategy, và config patch.
- Deterministic policy quyết định có mở retrain planning hay không.
- Config patch validator chặn key lạ, value không numeric, value ngoài range, và key không an toàn.
- Challenger chỉ được promote nếu governance pass.
- Final action là output nghiên cứu, không phải lệnh giao dịch.

## Report outputs

- Executive JSON: summary gọn cho reviewer.
- Technical JSON: full debug state.
- Markdown: readable research report.
- HTML: report có chart forecast.
