# Metrics Guide

Tài liệu này giải thích các metric quan trọng trong project bằng tiếng Việt, theo hướng dễ hiểu cho người mới học quantitative research / machine learning.

Lưu ý: một số metric trong danh sách assessment chưa được implement trong code hiện tại. Những metric đó được ghi rõ là "chưa implement".

## 1. MAE

### Tên metric

Mean Absolute Error.

### Công thức đơn giản

```text
MAE = trung bình(|giá thực tế - giá dự báo|)
```

### Ý nghĩa

MAE đo trung bình model dự báo sai bao nhiêu đơn vị giá.

Ví dụ nếu MAE = 5,000 VND, nghĩa là trung bình forecast lệch khoảng 5,000 VND so với giá thực tế.

### Dùng ở đâu trong project

- `src/modeling/trainer.py`: holdout evaluation.
- `src/modeling/validation.py`: walk-forward validation.

### Ví dụ dễ hiểu

Giá thực tế là 100, 110, 120. Model dự báo 98, 115, 118.

Sai số tuyệt đối là 2, 5, 2. MAE = 3.

### Khi nào metric báo động

MAE cao so với mức giá cổ phiếu nghĩa là model đang lệch nhiều. Tuy nhiên MAE phụ thuộc scale giá, nên nên đọc cùng MAPE.

## 2. RMSE

### Tên metric

Root Mean Squared Error.

### Công thức đơn giản

```text
RMSE = sqrt(trung bình((giá thực tế - giá dự báo)^2))
```

### Ý nghĩa

RMSE giống MAE nhưng phạt lỗi lớn mạnh hơn.

### Dùng ở đâu trong project

- `src/modeling/trainer.py`: holdout evaluation.
- `src/modeling/validation.py`: walk-forward validation.
- `src/agent/nodes.py`: governance gate, không cho challenger tệ hơn quá nhiều về RMSE.

### Ví dụ dễ hiểu

Nếu model sai nhỏ đều đều thì RMSE gần MAE. Nếu có vài ngày sai cực lớn, RMSE sẽ tăng mạnh.

### Khi nào metric báo động

RMSE tăng nhanh so với MAE thường cho thấy model có các lỗi lớn bất thường.

## 3. MAPE

### Tên metric

Mean Absolute Percentage Error.

### Công thức đơn giản

```text
MAPE = trung bình(|giá thực tế - giá dự báo| / |giá thực tế|)
```

### Ý nghĩa

MAPE đo lỗi theo phần trăm, dễ so sánh hơn MAE khi giá cổ phiếu có scale khác nhau.

### Dùng ở đâu trong project

- `src/modeling/trainer.py`: holdout evaluation.
- `src/modeling/validation.py`: walk-forward validation.
- `src/agent/nodes.py`: `node_evaluate` so sánh holdout MAPE với `configs/agent_config.yaml`.
- `src/agent/nodes.py`: governance gate yêu cầu challenger phải cải thiện MAPE.

### Ví dụ dễ hiểu

Giá thực tế 100, dự báo 95. Lỗi là 5%. MAPE càng thấp càng tốt.

### Khi nào metric báo động

Khi MAPE vượt threshold trong `agent_config.yaml`, agent đánh dấu model là `ABNORMAL` và có thể kích hoạt retrain loop.

Current implementation note: file config hiện tại là source of truth. Nếu assessment cần demo threshold 1%, hãy kiểm tra lại `configs/agent_config.yaml` trước khi chạy.

## 4. SMAPE

### Tên metric

Symmetric Mean Absolute Percentage Error.

### Công thức đơn giản

```text
SMAPE = trung bình(2 * |actual - predicted| / (|actual| + |predicted|))
```

### Ý nghĩa

SMAPE là phiên bản phần trăm cân bằng hơn giữa giá thực tế và giá dự báo.

### Dùng ở đâu trong project

- `src/modeling/validation.py`: walk-forward validation.

### Ví dụ dễ hiểu

Nếu actual = 100 và predicted = 110, SMAPE dùng cả 100 và 110 ở mẫu số, thay vì chỉ dùng actual như MAPE.

### Khi nào metric báo động

SMAPE cao nghĩa là forecast kém ổn định theo phần trăm. Nên đọc cùng MAPE.

## 5. Directional Accuracy

### Tên metric

Directional Accuracy.

### Công thức đơn giản

```text
Directional Accuracy =
  tỷ lệ ngày model dự báo đúng chiều tăng/giảm
```

Trong code:

```text
sign(actual_price - current_price) == sign(predicted_price - current_price)
```

### Ý nghĩa

Metric này không hỏi "dự báo đúng bao nhiêu VND", mà hỏi "dự báo đúng hướng không?".

### Dùng ở đâu trong project

- `src/modeling/validation.py`: walk-forward validation.
- `src/agent/nodes.py`: governance gate.
- `src/risk/risk_engine.py`: signal confidence và signal cap.

### Ví dụ dễ hiểu

Nếu cổ phiếu thực tế tăng và model cũng dự báo tăng, đó là đúng hướng.

### Khi nào metric báo động

Nếu directional accuracy dưới khoảng 50%, model đang yếu trong việc gọi đúng chiều. Risk engine có thể hạ signal về `WATCH` hoặc `HOLD`.

## 6. Interval Coverage

### Tên metric

Interval Coverage 80% và Interval Coverage 95%.

### Công thức đơn giản

```text
Coverage = tỷ lệ giá thực tế nằm trong khoảng dự báo
```

80% interval:

```text
q_0.1 <= actual <= q_0.9
```

95% interval:

```text
q_0.025 <= actual <= q_0.975
```

### Ý nghĩa

Coverage đo xem khoảng dự báo có "ôm" được giá thực tế hay không.

### Dùng ở đâu trong project

- `src/modeling/validation.py`: walk-forward validation.
- `src/agent/nodes.py`: governance gate dùng 95% interval coverage.
- `src/risk/risk_engine.py`: signal confidence.

### Ví dụ dễ hiểu

Nếu model nói khoảng 95% là 90-110, và giá thực tế là 105, lần đó được tính là covered.

### Khi nào metric báo động

Coverage quá thấp nghĩa là interval quá tự tin hoặc model không calibrated. Coverage cao nhưng interval quá rộng cũng có thể không hữu ích, nhưng project hiện chưa đo interval width riêng.

## 7. Pinball Loss

### Tên metric

Pinball Loss / Quantile Loss.

### Công thức đơn giản

Với quantile `q`:

```text
loss = max(q * residual, (q - 1) * residual)
residual = actual - predicted_quantile
```

### Ý nghĩa

Pinball loss là metric chuẩn cho quantile regression. Nó đánh giá chất lượng từng quantile, không chỉ median.

### Dùng ở đâu trong project

- `src/modeling/validation.py`: walk-forward validation.

### Ví dụ dễ hiểu

Với quantile thấp như 2.5%, model không bị phạt giống nhau khi dự báo trên hoặc dưới actual. Cách phạt bất đối xứng này giúp model học đúng vị trí quantile.

### Khi nào metric báo động

Pinball loss tăng nghĩa là các quantile forecast đang kém hơn. Nên so sánh theo thời gian hoặc giữa champion/challenger.

## 8. Prediction Bias

### Tên metric

Prediction Bias.

### Công thức đơn giản

```text
Prediction Bias = trung bình(predicted_price - actual_price)
```

### Ý nghĩa

Bias cho biết model có xu hướng dự báo cao hơn hay thấp hơn thực tế.

### Dùng ở đâu trong project

- `src/modeling/validation.py`: walk-forward validation.
- Report Markdown/JSON.

### Ví dụ dễ hiểu

Nếu bias = -500, model thường dự báo thấp hơn thực tế khoảng 500 VND.

### Khi nào metric báo động

Bias lớn theo một chiều cho thấy model có lỗi hệ thống, không chỉ nhiễu ngẫu nhiên.

## 9. VaR

### Tên metric

Value at Risk 95%.

### Công thức đơn giản trong project

```text
VaR_95 = max(0, (current_price - q_0.025_T+7) / current_price)
```

### Ý nghĩa

VaR 95% ước lượng mức lỗ downside ở vùng tail 5% theo forecast quantile.

### Dùng ở đâu trong project

- `src/risk/risk_engine.py`: risk report và signal logic.

### Ví dụ dễ hiểu

Nếu current price = 100 và q_0.025 = 92, VaR 95% = 8%.

### Khi nào metric báo động

VaR cao cho thấy downside tail risk lớn. Trong code, VaR cao có thể nâng `risk_level` lên `MEDIUM` hoặc `HIGH`.

## 10. Expected Shortfall

### Tên metric

Expected Shortfall.

### Công thức đơn giản trong project

```text
Expected Shortfall = VaR_95 * 1.15
```

### Ý nghĩa

Expected Shortfall cố gắng mô tả mức lỗ trung bình khi tình huống xấu hơn VaR xảy ra. Trong project hiện tại, đây là approximation đơn giản, không phải expected shortfall tính từ full distribution.

### Dùng ở đâu trong project

- `src/risk/risk_engine.py`: risk report và risk level.

### Ví dụ dễ hiểu

Nếu VaR 95% = 10%, expected shortfall trong project = 11.5%.

### Khi nào metric báo động

Expected shortfall cao nghĩa là tail risk nghiêm trọng. Code có thể phân loại risk là `HIGH`.

## 11. Risk/Reward Ratio

### Tên metric

Risk/Reward Ratio.

### Công thức đơn giản trong project

```text
risk_reward_ratio = upside_potential_95 / abs(downside_risk_95)
```

### Ý nghĩa

Metric này so sánh upside tiềm năng với downside risk.

### Dùng ở đâu trong project

- `src/risk/risk_engine.py`: quyết định preliminary signal.

### Ví dụ dễ hiểu

Nếu upside = 12% và downside = 6%, risk/reward = 2.0.

### Khi nào metric báo động

Risk/reward thấp nghĩa là upside không đủ hấp dẫn so với downside.

## 12. Sharpe Ratio

### Tên metric

Sharpe Ratio.

### Công thức đơn giản

```text
Sharpe = (portfolio_return - risk_free_rate) / return_volatility
```

### Ý nghĩa

Sharpe đo lợi nhuận điều chỉnh theo rủi ro.

### Dùng ở đâu trong project

Chưa implement. Không tìm thấy module backtest hoặc portfolio return dùng Sharpe ratio.

### Ví dụ dễ hiểu

Hai chiến lược có return 10%, nhưng chiến lược biến động thấp hơn sẽ có Sharpe tốt hơn.

### Khi nào metric báo động

Sharpe thấp hoặc âm báo hiệu chiến lược không tạo đủ return so với rủi ro.

## 13. Max Drawdown

### Tên metric

Maximum Drawdown.

### Công thức đơn giản

```text
Max Drawdown = mức giảm lớn nhất từ đỉnh vốn xuống đáy vốn
```

### Ý nghĩa

Max drawdown đo khoản lỗ tệ nhất nếu đi từ peak đến trough.

### Dùng ở đâu trong project

Chưa implement. Project hiện chưa có portfolio-level backtest.

### Ví dụ dễ hiểu

Nếu equity curve tăng lên 100 rồi giảm xuống 70 trước khi hồi phục, drawdown là 30%.

### Khi nào metric báo động

Drawdown lớn cho thấy chiến lược có rủi ro chịu lỗ sâu.

## 14. Win Rate

### Tên metric

Win Rate.

### Công thức đơn giản

```text
Win Rate = số giao dịch thắng / tổng số giao dịch
```

### Ý nghĩa

Win rate đo tỷ lệ signal/trade có kết quả dương.

### Dùng ở đâu trong project

Chưa implement. Project hiện chỉ output research signal, không có trade execution hoặc trade ledger.

### Ví dụ dễ hiểu

Nếu 60 trên 100 trade lời, win rate = 60%.

### Khi nào metric báo động

Win rate thấp có thể xấu, nhưng phải đọc cùng payoff ratio. Một chiến lược win rate thấp vẫn có thể tốt nếu lãi mỗi lần thắng lớn hơn lỗ mỗi lần thua.

## 15. Drift Severity

### Tên metric

Drift Severity.

### Công thức / logic

Project gán nhãn:

```text
LOW / MEDIUM / HIGH
```

dựa trên:

- feature drift
- target drift
- concept drift
- số feature bị drift

### Dùng ở đâu trong project

- `src/monitoring/drift_detector.py`
- `src/risk/risk_engine.py`
- report JSON/Markdown/HTML

### Ví dụ dễ hiểu

Nếu nhiều feature hiện tại khác mạnh so với lịch sử, severity có thể là `HIGH`.

### Khi nào metric báo động

`HIGH` thường dẫn tới `MANUAL_REVIEW` trong risk engine.

## 16. Regime Confidence

### Tên metric

Regime Confidence.

### Công thức / logic

Heuristic score dựa trên:

- số dòng dữ liệu
- volatility regime có rõ không
- trend regime có rõ không
- liquidity regime có bất thường không

### Dùng ở đâu trong project

- `src/monitoring/regime_detector.py`
- report output

### Ví dụ dễ hiểu

Nếu dữ liệu đủ dài và trend/volatility rõ, confidence cao hơn.

### Khi nào metric báo động

Confidence thấp nghĩa là regime label nên được đọc thận trọng.

## 17. Signal Confidence

### Tên metric

Signal Confidence.

### Công thức / logic

Risk engine bắt đầu từ base confidence rồi điều chỉnh theo:

- directional accuracy
- interval coverage
- risk level
- drift severity
- loại signal

### Dùng ở đâu trong project

- `src/risk/risk_engine.py`
- `AgentState`
- final report

### Ví dụ dễ hiểu

Nếu model có directional accuracy tốt và coverage tốt, confidence tăng. Nếu drift cao hoặc risk extreme, confidence giảm.

### Khi nào metric báo động

Confidence thấp nghĩa là signal nên được xem là yếu hoặc cần manual review.
