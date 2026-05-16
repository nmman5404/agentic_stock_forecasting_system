# 📈 **AGENTIC STOCK FORECASTING SYSTEM**

Dự án này là một **hệ thống nghiên cứu định lượng được hỗ trợ bởi AI Agent**, kết hợp giữa mô hình Machine Learning truyền thống và khả năng suy luận của Large Language Model.

Thay vì chỉ dựa vào các luật `if/else` cố định, hệ thống sử dụng **LightGBM Quantile Regression** để tạo dự báo xác suất, đồng thời dùng **Agent** để phân tích bối cảnh, đánh giá chất lượng mô hình, đọc tin tức và đề xuất cấu hình tinh chỉnh phù hợp.

Mục tiêu của dự án không phải là xây dựng một bot giao dịch tự động, mà là tạo ra một pipeline nghiên cứu có khả năng:

- Thu thập dữ liệu thị trường
- Tạo đặc trưng kỹ thuật
- Dự báo giá theo nhiều phân vị
- Đánh giá mô hình bằng walk-forward validation
- Giám sát drift, regime và rủi ro
- Dùng AI Agent (Gemini) để đề xuất cải thiện cấu hình mô hình,
- Sinh báo cáo nghiên cứu cuối cùng.


## 📂 **CẤU TRÚC THƯ MỤC**

```text
.
├── configs/
│   └── model_config.yaml         Lưu siêu tham số (hyperparameters) LightGBM và thiết lập walk-forward validation.
├── data/
│   ├── database.sqlite           # CSDL SQLite lưu dữ liệu raw và processed của các mã cổ phiếu.
│   └── raw/csv/                  # Lưu các file CSV snapshot ngay sau khi lấy dữ liệu từ API.
├── reports/                      # Lưu báo cáo đầu ra ở các định dạng JSON, HTML và Markdown.
├── src/
│   ├── agent/                    
│   │   ├── graph.py              # Định nghĩa State Machine bằng LangGraph để quản lý luồng chạy của Agent.
│   │   ├── nodes.py              # Chứa logic thực thi của từng Node như sửa quantile, gọi LLM, retrain, so sánh config.
│   │   ├── prompts.py            # Chứa các prompt template dùng để giao tiếp với LLM.
│   │   ├── state.py              # Định nghĩa cấu trúc bộ nhớ State xuyên suốt workflow của Agent.
│   │   └── tools.py              # Công cụ cho Agent, hiện gồm Google News RSS search, lọc trùng và lọc theo ngày.
│   ├── ingestion/                
│   │   └── vnstock_api.py        # Gọi API vnstock để lấy dữ liệu OHLCV của cổ phiếu mục tiêu và VN30.
│   ├── modeling/                
│   │   ├── predictor.py          # Chạy walk-forward validation và xuất dự báo 7 ngày.
│   │   ├── trainer.py            # Khởi tạo và huấn luyện LightGBM Quantile Regression.
│   │   └── validation.py         # Chạy walk-forward validation để đánh giá mô hình khách quan theo thời gian.
│   ├── monitoring/               
│   │   ├── drift_detector.py     # Phát hiện sự thay đổi trong phân phối dữ liệu và chất lượng mô hình.
│   │   ├── regime_detector.py    # Phân loại trạng thái thị trường theo trend, volatility và volume.
│   │   └── risk_engine.py        # Tính toán rủi ro tài chính như VaR, Expected Shortfall từ forecast quantiles.
│   ├── processing/              
│   │   ├── cleaner.py            # Xử lý missing values, merge dữ liệu cổ phiếu với bối cảnh VN30.
│   │   ├── db_manager.py         # Quản lý đọc/ghi DataFrame vào SQLite.
│   │   └── features.py           # Tính toán các chỉ báo phân tích kỹ thuật.
│   └── reporting/                
│       └── generator.py          # Render báo cáo Markdown, JSON, HTML và biểu đồ forecast bằng Plotly.
├── utils/
│   ├── helpers.py                # Các hàm tiện ích dùng chung như format số, thời gian.
│   └── logger.py                 # Cấu hình logger cho terminal và file log hằng ngày.
├── main.py                       # Entry point để khởi chạy toàn bộ hệ thống.
└── requirements.txt              
```

## 🔄 **WORKFLOW CHI TIẾT CỦA DỰ ÁN**

Hệ thống được khởi chạy bằng lệnh:

```bash
python main.py
```

Lệnh này gọi hàm `run_daily_pipeline(target_ticker)`. Toàn bộ pipeline gồm 6 bước chính.


### **Bước 1: Data Ingestion — Thu thập dữ liệu**

Hệ thống gọi API `vnstock` để lấy dữ liệu OHLCV lịch sử trong khoảng 2 năm cho mã cổ phiếu mục tiêu.

Ngoài cổ phiếu chính, hệ thống cũng lấy thêm dữ liệu bối cảnh thị trường như:

- VN30,
- VN30F1M,

Dữ liệu thô sau đó được lưu vào:

```text
data/raw/
data/database.sqlite
```

Việc lưu snapshot giúp quá trình kiểm tra, debug và tái lập kết quả dễ dàng hơn.

---

### **Bước 2: Feature Engineering — Tạo đặc trưng dữ liệu**

Dữ liệu thô được biến đổi thành các feature phục vụ mô hình dự báo.

Các feature chính gồm:

- Tỷ suất lợi nhuận
- Biến động khối lượng
- Moving averages (MA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Rate of Change (ROC)
- Average True Range (ATR)
- Lag features
- Tỷ suất thay đổi của chỉ số/hợp đồng phái sinh và khối lượng giao dịch của VN30.

Sau khi tạo feature, hệ thống:

1. merge dữ liệu cổ phiếu với dữ liệu bối cảnh,
2. xóa các dòng bị NaN do rolling window hoặc lag,
3. lưu kết quả vào bảng trong SQLite:

```text
processed_{ticker}
```

---

### **Bước 3: Modeling & Validation — Mô hình hóa và đánh giá**

Hệ thống không lưu model `.pkl` cố định, mà train lại mô hình theo dữ liệu mới mỗi lần chạy.

Quy trình gồm hai phần:

#### 1. Walk-forward validation

Hệ thống sử dụng walk-forward validation để đánh giá mô hình theo cách gần với thực tế thời gian.

Thay vì chia train/test ngẫu nhiên, dữ liệu được chia theo cửa sổ thời gian:

```text
train quá khứ → validate tương lai gần
```

Cách này phù hợp hơn với dữ liệu tài chính vì tránh rò rỉ thông tin tương lai.

Các metric chính gồm:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Symmetric Mean Absolute Percentage Error (sMAPE)
- Directional Accuracy (độ chính xác xu hướng tăng/giảm)
- Prediction Interval Coverage (80%, 95%)
- Pinball Loss cho Quantile Regression

#### 2. Prediction

Sau khi đánh giá, mô hình được train trên toàn bộ dữ liệu hiện tại để dự báo 7 ngày tiếp theo.

Dự báo được xuất ra dưới dạng 5 phân vị:

```text
q_0.025
q_0.1
q_0.5
q_0.9
q_0.975
```

Trong đó `q_0.5` là dự báo trung vị, còn các phân vị còn lại tạo thành dải bất định của forecast.

---

### **Bước 4: Monitoring — Giám sát bối cảnh**

Sau khi có forecast và validation metrics, hệ thống chạy các monitoring engine.

Các engine này chỉ có vai trò **mô tả trạng thái**, không trực tiếp ra quyết định giao dịch.

#### Regime Detector

Xác định thị trường đang ở trạng thái nào:

- Uptrend hay downtrend
- Biến động cao hay thấp (volitality)
- Volume bình thường hay đột biến

#### Drift Detector

Đánh giá dữ liệu hiện tại có khác biệt đáng kể so với quá khứ không.

Drift được chia thành:

- Feature drift: phân phối của các feature đầu vào thay đổi như thế nào so với dữ liệu lịch sử dùng để train mô hình.
- Target drift: phân phối của biến mục tiêu (giá hoặc return tương lai) thay đổi như thế nào theo thời gian.
- Concept drift: mối quan hệ giữa feature và target thay đổi như thế nào, các pattern cũ còn dự báo tốt như trước không.

#### Risk Engine

Tính toán các rủi ro tài chính từ forecast quantiles:

- Expected return,
- Downside risk,
- VaR,
- Expected shortfall,
- Risk/ reward ratio.

---

### **Bước 5: LLM Agentic Workflow — Vòng lặp AI bằng LangGraph**

Sau khi có đầy đủ dữ liệu, forecast, metrics và monitoring reports, hệ thống đóng gói tất cả vào `AgentState`.

`AgentState` sau đó được đưa vào LangGraph để AI Agent xử lý theo từng node.

#### 1. `node_fix_quantiles`

Node này kiểm tra các dải phân vị dự báo.

Nếu xảy ra lỗi quantile crossing, ví dụ:

```text
q_0.1 > q_0.5
```

thì node sẽ sửa lại thứ tự các quantile để đảm bảo forecast hợp lệ về mặt toán học.

#### 2. `node_llm_evaluate`

LLM đóng vai trò đánh giá ban đầu.

Nó đọc các metrics và monitoring context để xác định mô hình hiện tại đang ở trạng thái:

```text
OK
BAD
```

Nếu trạng thái là `OK`, workflow đi thẳng đến bước báo cáo.

Nếu trạng thái là `BAD`, workflow chuyển sang bước đọc tin tức.

#### 3. `node_search_news`

Node này dùng Google News RSS để lấy các tin tức liên quan đến mã cổ phiếu.

Quy trình gồm:

- Tạo query theo ticker
- Lấy RSS entries
- Lọc trùng
- Lọc theo thời gian
- Match keyword
- Lấy title và summary làm context

Tin tức đóng vai trò cung cấp bối cảnh cho Agent giải thích lý do mô hình kém hiệu quả.

#### 4. `node_llm_retrain_compare`

LLM đề xuất bộ tham số LightGBM mới dựa trên:

- Validation metrics,
- Drift,
- Regime,
- Risk,
- News context.

Sau đó hệ thống dùng Python để:

1. Train lại mô hình với bộ tham số mới,
2. Đánh giá lại bằng walk-forward validation,
3. So sánh metrics mới và cũ.

Nếu cấu hình mới tốt hơn, hệ thống ghi đè vào `model_config.yaml`.

Nếu cấu hình mới không tốt hơn, hệ thống từ chối cấu hình đó và kích hoạt cơ chế Self-Reflection (Tự phản tỉnh). Nó lưu lại cấu hình thất bại vào bộ nhớ (rejected_configs), giúp LLM tự rút kinh nghiệm và không lặp lại sai lầm trong lần thử tiếp theo.

Workflow có thể lặp lại cho đến khi mô hình đạt trạng thái tốt hơn hoặc hết số lần thử tối đa.

---

### **Bước 6: Final Reporting — Sinh báo cáo cuối cùng**

Ở bước cuối, hệ thống tổng hợp toàn bộ kết quả để sinh báo cáo.

Báo cáo gồm:

- Forecast,
- Validation metrics,
- Drift/ regime/ risk monitoring,
- News context,
- Quyết định config,
- Final research action,
- Audit trail của workflow.

Kết quả được render ra nhiều định dạng:

```text
Markdown
HTML
JSON
```

Final action có thể là:

```text
BUY
SELL
HOLD
WATCH
```

Kết quả này chỉ nên được hiểu là **research output / paper-trading signal**, không phải khuyến nghị đầu tư hay tín hiệu giao dịch thật.



## 🧮 **CÁC CÔNG THỨC TOÁN HỌC VÀ LOGIC**


### **Công thức trong Feature Engineering (`features.py`)**

Mô hình không chỉ học trực tiếp trên giá tuyệt đối, mà còn sử dụng nhiều đặc trưng dựa trên tỷ suất lợi nhuận và chỉ báo kỹ thuật để tăng khả năng mô tả trạng thái thị trường.

#### Daily Return — Tỷ suất lợi nhuận ngày

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

Trong đó:

- $P_t$ là giá đóng cửa tại ngày $t$,
- $P_{t-1}$ là giá đóng cửa ngày trước đó.

---

#### Volume Change — Biến động khối lượng

$$
V\_change_t = \frac{V_t - V_{t-1}}{V_{t-1}}
$$

Chỉ số này cho biết khối lượng giao dịch thay đổi bao nhiêu so với ngày trước đó.

---

#### Moving Average — Trung bình động

$$
MA_n = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}
$$

Moving average giúp mô hình nắm bắt xu hướng ngắn hạn và trung hạn của giá.

---

#### Volatility — Độ biến động

Volatility được tính bằng độ lệch chuẩn của tỷ suất lợi nhuận trong một cửa sổ thời gian.

$$
\sigma_n = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(R_i - \bar{R})^2}
$$

Trong đó:

- $R_i$ là tỷ suất lợi nhuận tại thời điểm $i$,
- $\bar{R}$ là tỷ suất lợi nhuận trung bình trong cửa sổ,
- $n$ là số phiên được xét.

Trong code, hệ thống sử dụng rolling volatility với cửa sổ 7 ngày.

---

#### Các chỉ báo kỹ thuật khác

Các chỉ báo như:

- Relative Strength Index (RSI),
- Moving Average Convergence Divergence (MACD),
- Average True Range (ATR),
- Rate of Change (ROC),

được tính theo công thức chuẩn trong phân tích kỹ thuật thông qua thư viện `ta`.

---

#### RSI — Relative Strength Index

RSI đo động lượng tăng giảm của giá:

$$
RSI = 100 - \frac{100}{1 + RS}
$$

Trong đó:

$$
RS = \frac{\text{Average Gain}}{\text{Average Loss}}
$$

---

#### MACD — Moving Average Convergence Divergence

MACD đo sự khác biệt giữa hai đường EMA:

$$
MACD = EMA_{12} - EMA_{26}
$$

Signal line:

$$
Signal = EMA_9(MACD)
$$

---

#### ATR — Average True Range

ATR đo mức độ biến động giá:

$$
TR = \max(H-L,\ |H-C_{prev}|,\ |L-C_{prev}|)
$$

$$
ATR = \frac{1}{n}\sum_{i=1}^{n} TR_i
$$

Trong đó:
- $H$ là giá cao nhất,
- $L$ là giá thấp nhất,
- $C_{prev}$ là giá đóng cửa phiên trước.

---

#### ROC — Rate of Change

ROC đo tốc độ thay đổi của giá:

$$
ROC = \frac{P_t - P_{t-n}}{P_{t-n}} \times 100
$$

Trong đó:
- $P_t$ là giá hiện tại,
- $P_{t-n}$ là giá cách đó $n$ phiên.

---

### **Toán học trong Modeling và Validation (`validation.py`)**

Hệ thống sử dụng **Quantile Regression**.

Thay vì chỉ dự báo một giá trị duy nhất, mô hình dự báo nhiều phân vị của phân phối giá tương lai:

$$
q \in \{0.025, 0.1, 0.5, 0.9, 0.975\}
$$

Trong đó:

- $q = 0.5$ là dự báo trung vị,
- $q = 0.025$ và $q = 0.975$ tạo thành dải dự báo 95%,
- $q = 0.1$ và $q = 0.9$ tạo thành dải dự báo 80%.

Gọi:

- $y$ là giá trị thực tế,
- $\hat{y}$ là giá trị dự báo trung vị.

---

#### MAE — Mean Absolute Error

$$
MAE = \frac{1}{n} \sum |y - \hat{y}|
$$

MAE đo sai số tuyệt đối trung bình giữa giá trị thật và giá trị dự báo.

---

#### RMSE — Root Mean Squared Error

$$
RMSE = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}
$$

RMSE phạt mạnh hơn đối với các sai số lớn.

---

#### MAPE — Mean Absolute Percentage Error

$$
MAPE = \frac{1}{n} \sum \left| \frac{y - \hat{y}}{y} \right|
$$

MAPE cho biết sai số trung bình dưới dạng phần trăm.

---

#### sMAPE — Symmetric MAPE

$$
sMAPE = \frac{1}{n} \sum \frac{2|y - \hat{y}|}{|y| + |\hat{y}|}
$$

sMAPE giúp giảm bớt thiên lệch khi giá trị thực tế quá nhỏ.

---

#### Directional Accuracy — Độ chính xác xu hướng

$$
DA = \frac{1}{n} \sum \mathbb{I}(\text{sign}(y - P_0) = \text{sign}(\hat{y} - P_0))
$$

Metric này đo tỷ lệ mô hình dự báo đúng hướng tăng hoặc giảm.

---

#### Interval Coverage — Độ bao phủ khoảng dự báo

$$
Coverage = \frac{1}{n} \sum \mathbb{I}(q_{lower} \le y \le q_{upper})
$$

Metric này cho biết tỷ lệ giá thực tế rơi vào trong khoảng dự báo.

Với dải 95%, coverage lý tưởng nên gần 95%.

---

#### Pinball Loss

Pinball Loss là hàm mất mát chính cho Quantile Regression.

$$
L_q(y, \hat{y}) = \max(q \cdot (y - \hat{y}), (q - 1) \cdot (y - \hat{y}))
$$

Nếu giá trị thực nằm dưới dự báo, mô hình bị phạt theo tỷ lệ $1-q$.

Nếu giá trị thực nằm trên dự báo, mô hình bị phạt theo tỷ lệ $q$.

---

### **Giám sát bối cảnh — Regime Detector (`regime_detector.py`)**

Regime Detector mô tả trạng thái thị trường theo ba nhóm:

```text
Volatility
Trend
Volume
```

---

#### Volatility Regime

Hệ thống tính độ lệch chuẩn lợi nhuận 20 ngày gần nhất:


$$
\sigma_{20} = std(R_{t-19}, ..., R_t)
$$

Sau đó so sánh với lịch sử rolling volatility bằng percentile rank.

- `LOW_VOLATILITY`: Percentile < 0.25
- `NORMAL_VOLATILITY`: 0.25 ≤ Percentile < 0.75
- `HIGH_VOLATILITY`: 0.75 ≤ Percentile < 0.92
- `EXTREME_VOLATILITY`: Percentile ≥ 0.92

---

#### Trend Regime

Trend được xác định bằng khoảng cách giữa MA7 và MA21:

$$
MA\ Gap = \frac{MA_7 - MA_{21}}{MA_{21}}
$$

Nếu:

```text
MA Gap > 1.5%
và return 20 ngày > 3%
```

thì hệ thống gắn nhãn:

```text
UPTREND
```

Ngược lại, nếu điều kiện giảm tương ứng xảy ra, hệ thống có thể gắn nhãn:

```text
DOWNTREND
```

Ngoài ra, hệ thống cũng gắn nhãn:
- `SIDEWAYS`: Nếu khoảng cách $|MA\ Gap| \le 1.5\%$ và $|return\ 20d| \le 4\%$.
- `MIXED_TREND`: Khi các điều kiện trên không đồng nhất.

---

#### Volume Regime

Volume regime sử dụng Z-score và tỷ lệ thanh khoản ngắn hạn để phát hiện đột biến.

$$
Z = \frac{V_{current} - \mu_{60\_days}}{\sigma_{60\_days}}
$$
$$
Ratio = \frac{\mu_{5\_days}}{\mu_{60\_days}}
$$

Hệ thống gắn nhãn dựa trên các ngưỡng:
- `VOLUME_SPIKE`: Nếu $Z \ge 2.5$ hoặc $Ratio \ge 1.8$ (Thanh khoản tăng vọt).
- `LOW_VOLUME`: Nếu $Ratio \le 0.45$ (Thanh khoản cạn kiệt).
- `NORMAL_VOLUME`: Các trường hợp còn lại.

---

### **Giám sát sai lệch — Drift Detector (`drift_detector.py`)**

Drift Detector so sánh dữ liệu hiện tại (`cur`) với dữ liệu quá khứ (`ref`).

Mục tiêu là xác định liệu phân phối dữ liệu, target hoặc chất lượng mô hình có đang thay đổi không.

#### Cách xác định tập dữ liệu để đo Drift

Để tính toán sự sai lệch, hệ thống chia toàn bộ chuỗi thời gian đã qua xử lý (`df_processed`) thành hai tập: **tập hiện tại (Current)** và **tập tham chiếu (Reference)** theo quy tắc động:

- **Tập hiện tại (Current Dataset):** Mặc định lấy **60 phiên giao dịch gần nhất** (khoảng 3 tháng). Nếu tổng dữ liệu có ít hơn 60 phiên, lấy toàn bộ dữ liệu.
- **Tập tham chiếu (Reference Dataset):** 
  - Nếu tổng số phiên giao dịch lớn hơn 120: Lấy toàn bộ dữ liệu từ đầu cho đến trước 60 ngày cuối cùng.
  - Nếu tổng số phiên giao dịch nhỏ hơn hoặc bằng 120: Lấy **50% dữ liệu đầu tiên** (nửa đầu của chuỗi thời gian).

Cách chia này đảm bảo tập `Current` luôn đại diện cho "trạng thái bình thường mới" của thị trường trong ngắn hạn, còn tập `Reference` đủ dài để đóng vai trò là "chuẩn mực lịch sử".

---

#### Mean Shift Z-Score

$$
Z = \frac{|\mu_{cur} - \mu_{ref}|}{\sigma_{ref}}
$$

Công thức này đo mức độ dịch chuyển của trung bình hiện tại so với trung bình quá khứ.

Ngưỡng phân loại Z-score (Độ lệch trung bình):
- `NONE`: $Z < 0.1$
- `LOW`: $1.0 \le Z < 2.0$
- `MEDIUM`: $2.0 \le Z < 3.0$
- `HIGH`: $Z \ge 3.0$

---

#### Standard Deviation Ratio

$$
Ratio = \frac{\sigma_{cur}}{\sigma_{ref}}
$$

Metric này đo sự thay đổi về độ biến động giữa dữ liệu hiện tại và dữ liệu tham chiếu.

Ngưỡng phân loại Ratio (Độ lệch biến động): 
- NONE: $0.8 \le Ratio \le 1.2$
- LOW: $0.6 \le Ratio < 0.8$ hoặc $1.2 < Ratio \le 1.5$
- MEDIUM: $0.5 \le Ratio < 0.6$ hoặc $1.5 < Ratio \le 2.0$
- HIGH: $Ratio < 0.5$ hoặc $Ratio > 2.0$

---

#### PSI — Population Stability Index

PSI đo mức độ thay đổi của phân phối xác suất.

Dữ liệu được chia thành nhiều bin, sau đó so sánh tỷ lệ mẫu rơi vào từng bin giữa tập hiện tại và tập tham chiếu.

$$
PSI = \sum_{i=1}^{10} (Pct_{cur, i} - Pct_{ref, i}) \cdot \ln\left(\frac{Pct_{cur, i}}{Pct_{ref, i}}\right)
$$

Trong đó:

- $Pct_{cur, i}$ là tỷ lệ mẫu hiện tại trong bin $i$,
- $Pct_{ref, i}$ là tỷ lệ mẫu tham chiếu trong bin $i$.

Ngưỡng đánh giá độ lệch phân phối (Drift Level):
- `NONE`: PSI < 0.1
- `LOW`: 0.1 ≤ PSI < 0.2
- `MEDIUM`: 0.2 ≤ PSI < 0.35
- `HIGH`: PSI ≥ 0.35 (Lệch phân phối cực nặng)

---
#### Concept Drift (Suy giảm chất lượng mô hình)

Hệ thống theo dõi sự sụt giảm các metrics của mô hình trên tập validation. Việc đánh giá dựa trên cơ chế phân lớp (thỏa mãn điều kiện tệ nhất sẽ lấy nhãn đó):

Ngưỡng MAPE (Càng cao càng tệ)

- `NONE`: $MAPE < 0.03$
- `LOW`: $0.03 \le MAPE < 0.05$
- `MEDIUM`: $0.05 \le MAPE < 0.08$
- `HIGH`: $MAPE \ge 0.08$

---

Ngưỡng Directional Accuracy (Càng thấp càng tệ)

- `NONE`: $DA \ge 0.60$
- `LOW`: $0.55 \le DA < 0.60$
- `MEDIUM`: $0.48 \le DA < 0.55$
- `HIGH`: $DA < 0.48$

---

Ngưỡng Interval Coverage 95% (Càng thấp càng tệ)
- `NONE`: $Coverage \ge 0.80$
- `LOW`: $0.70 \le Coverage < 0.80$
- `MEDIUM`: $0.55 \le Coverage < 0.70$
- `HIGH`: $Coverage < 0.55$

---

### **Giám sát rủi ro — Risk Engine (`risk_engine.py`)**

Risk Engine tính toán rủi ro tài chính dựa trên các dải phân vị dự báo.

---

#### Expected Return — Lợi nhuận kỳ vọng

$$
Expected\ Return = \frac{q_{0.5} - P_{current}}{P_{current}}
$$

Trong đó:

- $q_{0.5}$ là dự báo trung vị,
- $P_{current}$ là giá hiện tại.

---

#### Downside Risk 95%

$$
Downside = \min\left(0, \frac{q_{0.025} - P_{current}}{P_{current}}\right)
$$

Metric này đo mức giảm giá cực hạn theo biên dưới của dải dự báo 95%.

---

#### Upside Potential 95%

$$
Upside = \max\left(0, \frac{q_{0.975} - P_{current}}{P_{current}}\right)
$$

Metric này đo tiềm năng tăng giá theo biên trên của dải dự báo 95%.

---

#### Value at Risk — VaR 95%

VaR 95% ước tính mức tổn thất tối đa trong điều kiện bình thường với độ tin cậy 95%. Trong hệ thống, VaR tương ứng với phần giá trị tuyệt đối của downside risk.

$$
VaR_{95} = |Downside|
$$


---

#### Expected Shortfall — CVaR

Expected Shortfall ước lượng rủi ro trong vùng tail risk xấu nhất.

Hệ thống sử dụng công thức heuristic đơn giản:

$$
CVaR = VaR \cdot 1.15
$$

Dựa vào hai chỉ số này, hệ thống phân loại rủi ro:
- `EXTREME_RISK`: Nếu $VaR_{95} \ge 20\%$ hoặc $CVaR \ge 25\%$.
- `HIGH_RISK`: Nếu $VaR_{95} \ge 12\%$ hoặc $CVaR \ge 15\%$.
- `MEDIUM_RISK`: Nếu $VaR_{95} \ge 7\%$.
- `LOW_RISK`: Mức độ rủi ro bình thường.

---

#### Risk/Reward Ratio

$$
Risk/Reward = \frac{Upside}{|Downside|}
$$

Metric này cho biết tiềm năng lợi nhuận so với rủi ro giảm giá.

---