# Pipeline Walkthrough VI

Tài liệu này giải thích toàn bộ pipeline bằng tiếng Việt, theo kiểu hướng dẫn lại cho người mới đọc project.

## 1. Dự án này làm gì?

Project **Agentic Vingroup Quant Forecasting System** là một hệ thống research pipeline cho cổ phiếu nhóm Vingroup.

Nó làm các việc chính:

1. Lấy dữ liệu giá/khối lượng từ vnstock.
2. Lưu dữ liệu vào SQLite.
3. Tạo feature kỹ thuật.
4. Train model LightGBM Quantile.
5. Forecast 7 ngày tiếp theo.
6. Đánh giá model bằng holdout và walk-forward validation.
7. Kiểm tra market regime.
8. Kiểm tra drift.
9. Tính risk report.
10. Cho LangGraph Agent đánh giá model và governance.
11. Nếu cần, thử retrain model challenger.
12. Nếu challenger tốt hơn theo governance gate thì accept, nếu không thì rollback/giữ champion.
13. Xuất report JSON, Markdown, HTML.

Điểm quan trọng: hệ thống này chỉ tạo **research signal**, không đặt lệnh thật.

## 2. Dữ liệu đi từ đâu tới đâu?

### Bước 1: Lấy dữ liệu

File:

```text
src/ingestion/vnstock_api.py
```

Hàm chính:

```python
get_vingroup_and_context_data(start_date, end_date)
```

Hàm này lấy:

- `VIC`
- `VHM`
- `VRE`
- `VPL`
- `VN30`
- `VN30F1M`

Dữ liệu là daily OHLCV:

```text
open, high, low, close, volume, ticker
```

### Bước 2: Lưu raw data

File:

```text
src/processing/cleaner.py
src/processing/db_manager.py
```

Raw data được lưu vào SQLite:

```text
data/database.sqlite
```

Các bảng raw:

```text
raw_VIC
raw_VHM
raw_VRE
raw_VPL
raw_VN30
raw_VN30F1M
```

### Bước 3: Tạo feature

File:

```text
src/processing/features.py
```

Hàm:

```python
generate_technical_features(df)
generate_context_features(context_df, prefix)
```

Feature tạo ra gồm:

- daily return
- volume change
- moving average
- volatility
- RSI
- MACD
- ATR
- ROC
- calendar features
- lag features
- VN30 context
- VN30F1M context

### Bước 4: Lưu processed data

Processed data được lưu vào:

```text
processed_VIC
processed_VHM
processed_VRE
processed_VPL
```

Đây là dữ liệu chính model sử dụng.

## 3. Model học cái gì?

Model nằm ở:

```text
src/modeling/trainer.py
src/modeling/predictor.py
```

Class chính:

```python
QuantileLightGBM
```

Hàm quan trọng:

```python
prepare_data(df, step)
```

Model không học trực tiếp raw price tương lai. Nó học **future return**:

```text
target = (close[t + step] - close[t]) / close[t]
```

Ví dụ:

- hôm nay close = 100
- 7 ngày sau close = 110
- target return = 10%

Khi model predict xong return, project convert lại thành price:

```text
predicted_price = current_close * (1 + predicted_return)
```

## 4. Forecast 7 ngày hoạt động ra sao?

File:

```text
src/modeling/predictor.py
```

Hàm:

```python
generate_7_day_forecast(df)
```

Pipeline forecast:

1. Tạo trainer `QuantileLightGBM`.
2. Chạy holdout evaluation.
3. Chạy walk-forward validation.
4. Lấy dòng dữ liệu mới nhất làm input.
5. Lặp `step=1` tới `step=7`.
6. Với mỗi step, train nhiều model quantile.
7. Trả ra các forecast:

```text
q_0.025
q_0.1
q_0.5
q_0.9
q_0.975
```

Ý nghĩa:

- `q_0.5`: median forecast
- `q_0.1` và `q_0.9`: khoảng 80%
- `q_0.025` và `q_0.975`: khoảng 95%

## 5. Agent làm gì?

Agent nằm trong:

```text
src/agent/
```

Các file chính:

```text
state.py
graph.py
nodes.py
tools.py
```

Agent không phải trader tự động. Nó đóng vai trò:

- Quantitative Risk Committee Agent
- Model Governance Reviewer
- Market Context Analyst

Agent đọc:

- forecast data
- holdout metrics
- walk-forward metrics
- regime report
- drift report
- risk report
- news evidence
- governance decision

Agent output:

- model status
- shock type
- news assessment
- governance decision
- final research signal
- final recommendation text
- audit trail

## 6. Agent workflow chạy như thế nào?

File:

```text
src/agent/graph.py
src/agent/nodes.py
```

Graph gồm 5 node:

```text
node_validate
node_evaluate
node_contextualize
node_improve
node_recommend
```

Luồng cơ bản:

```text
node_validate
  -> node_evaluate
      -> nếu PASS: node_recommend
      -> nếu ABNORMAL lần đầu: node_contextualize
      -> nếu ABNORMAL sau retry: node_improve
```

Sau contextualize:

```text
BLACK_SWAN hoặc DATA_ISSUE -> node_recommend
các trường hợp khác -> node_improve
```

Sau improve:

```text
nếu còn retry -> quay lại node_validate
nếu hết retry -> node_recommend
```

## 7. Vì sao cần retrain?

Model có thể xuống chất lượng vì:

- thị trường đổi regime
- distribution của feature thay đổi
- volatility tăng
- dữ liệu gần đây khác dữ liệu lịch sử
- model hiện tại underfit/overfit

Trong project, retrain được kích hoạt khi:

```text
holdout MAPE > thresholds.max_mape
```

Threshold nằm ở:

```text
configs/agent_config.yaml
```

Current implementation note: file config hiện tại là source of truth. Nếu bài assessment yêu cầu demo threshold 1%, hãy kiểm tra file này trước khi chạy.

## 8. Vì sao có rollback?

Không phải model retrain là tốt hơn.

Một challenger có thể:

- MAPE tốt hơn một chút
- nhưng directional accuracy tệ hơn
- hoặc interval coverage tệ hơn
- hoặc RMSE tệ hơn

Vì vậy project dùng governance gate:

```text
MAPE phải cải thiện
Directional accuracy không được giảm quá nhiều
95% interval coverage không được giảm quá nhiều
RMSE không được xấu hơn quá nhiều
```

Nếu challenger fail, project rollback config và giữ champion.

File:

```text
src/agent/nodes.py
```

Hàm:

```python
_compare_champion_challenger(...)
```

## 9. Regime detection là gì?

File:

```text
src/monitoring/regime_detector.py
```

Hàm:

```python
detect_regime(processed_df)
```

Regime detector trả ra:

```text
volatility_regime
trend_regime
liquidity_regime
regime_confidence
regime_notes
```

Ví dụ:

```text
NORMAL_VOLATILITY
UPTREND
NORMAL_LIQUIDITY
```

Regime giúp risk engine hiểu forecast đang nằm trong môi trường thị trường nào.

## 10. Drift detection là gì?

File:

```text
src/monitoring/drift_detector.py
```

Hàm:

```python
detect_drift(reference_df, current_df, validation_metrics)
```

Drift gồm 3 loại:

### Feature drift

Feature hiện tại khác mạnh so với feature lịch sử.

Ví dụ:

- volume tăng bất thường
- RSI distribution khác trước
- volatility feature khác trước

### Target drift

Return của giá close thay đổi distribution.

### Concept drift

Quan hệ giữa feature và target có thể đã thay đổi. Project suy luận concept drift từ metric validation:

- MAPE quá cao
- directional accuracy yếu
- interval coverage thấp

Nếu drift severity là `HIGH`, risk engine có thể đưa signal về `MANUAL_REVIEW`.

## 11. Risk engine làm gì?

File:

```text
src/risk/risk_engine.py
```

Hàm:

```python
calculate_risk_report(...)
```

Risk engine đọc forecast quantiles và tính:

- expected return 7 ngày
- downside risk 95%
- upside potential 95%
- VaR 95%
- expected shortfall
- risk/reward ratio
- risk level
- preliminary signal
- signal confidence

Signal có thể là:

```text
BUY
SELL
HOLD
WATCH
MANUAL_REVIEW
```

Nếu drift cao hoặc volatility extreme, signal thường bị đưa về `MANUAL_REVIEW`.

## 12. Báo cáo output đọc như thế nào?

Reports nằm ở:

```text
reports/json/
reports/markdown/
reports/html/
```

### JSON report

Dành cho máy đọc hoặc debug sâu. Nó chứa gần như toàn bộ `AgentState`.

Nên xem JSON khi bạn muốn biết:

- forecast data đầy đủ
- risk_report
- drift_report
- regime_report
- governance_decision
- audit_trail

### Markdown report

Dành cho người đọc. Có cấu trúc:

1. Executive Summary
2. Forecast Performance
3. Walk-forward Validation
4. Market Regime
5. Drift Detection
6. Risk Assessment
7. News & Event Context
8. Model Governance Decision
9. Final Research Signal
10. Audit Trail

### HTML report

Dành cho xem nhanh trong browser. Có summary panel và forecast fan chart bằng Plotly.

## 13. Nếu tôi muốn debug thì xem file nào trước?

### Pipeline không chạy

Xem:

```text
main.py
logs/YYYY-MM-DD_system.log
```

### Không lấy được data

Xem:

```text
src/ingestion/vnstock_api.py
```

Kiểm tra:

- network
- vnstock API
- symbol
- date range

### Data đã lấy nhưng feature lỗi

Xem:

```text
src/processing/features.py
src/processing/cleaner.py
data/database.sqlite
```

### Model forecast lỗi

Xem:

```text
src/modeling/trainer.py
src/modeling/predictor.py
src/modeling/validation.py
```

### Drift/regime/risk lạ

Xem:

```text
src/monitoring/regime_detector.py
src/monitoring/drift_detector.py
src/risk/risk_engine.py
```

### Agent routing lạ

Xem:

```text
src/agent/graph.py
src/agent/nodes.py
src/agent/state.py
```

### News luôn NO_NEWS

Xem:

```text
src/agent/tools.py
```

Có thể RSS parse lỗi hoặc keyword không match.

### Report thiếu field

Xem:

```text
src/reporting/generator.py
reports/json/
```

JSON report là nơi tốt nhất để kiểm tra field thực tế.

## 14. File đọc theo thứ tự đề xuất

Nếu bạn mới quay lại project sau một thời gian, nên đọc theo thứ tự:

1. `README.md`
2. `docs/CODEBASE_REVIEW.md`
3. `main.py`
4. `src/processing/features.py`
5. `src/modeling/trainer.py`
6. `src/modeling/predictor.py`
7. `src/modeling/validation.py`
8. `src/monitoring/regime_detector.py`
9. `src/monitoring/drift_detector.py`
10. `src/risk/risk_engine.py`
11. `src/agent/graph.py`
12. `src/agent/nodes.py`
13. `src/reporting/generator.py`

## 15. Cách hiểu final signal

Final signal không phải lời khuyên đầu tư.

Ý nghĩa thực tế:

| Signal | Cách hiểu |
|---|---|
| `BUY` | Forecast/risk tương đối thuận lợi theo rule hiện tại. |
| `SELL` | Forecast/risk nghiêng tiêu cực. |
| `HOLD` | Không đủ edge rõ ràng để hành động. |
| `WATCH` | Có tín hiệu cần theo dõi nhưng chưa đủ mạnh. |
| `MANUAL_REVIEW` | Model/data/risk có vấn đề, cần con người kiểm tra. |

Trong môi trường assessment, `MANUAL_REVIEW` không phải lỗi. Nó có thể là output đúng nếu drift hoặc tail risk cao.
