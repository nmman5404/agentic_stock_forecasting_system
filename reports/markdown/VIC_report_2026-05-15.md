# Quant Research Report: VIC (2026-05-15)

## 1. Executive Summary
- **Final Action:** WATCH
- **Model Status:** BAD (Retries: 2)
- **Summary:** VIC đang trong trạng thái kỹ thuật phân kỳ với xu hướng tăng ngắn hạn nhưng đối mặt với rủi ro điều chỉnh cao và hiện tượng lệch dữ liệu (drift) nghiêm trọng.

### Luận điểm đầu tư (Reasoning):
Mặc dù các tin tức cơ bản cho thấy Vingroup đang tích cực tái cấu trúc tài chính (tất toán trái phiếu) và mở rộng sang các lĩnh vực công nghệ cao, dữ liệu định lượng lại phát đi tín hiệu cảnh báo. Thứ nhất, mức rủi ro hiện tại là HIGH_RISK với dự báo lợi nhuận kỳ vọng âm (-1.77%) trong 7 ngày tới. Thứ hai, hiện tượng 'Feature Drift' ở mức cao (16 đặc trưng bị lệch) cho thấy mô hình dự báo đang gặp khó khăn trong việc thích nghi với biến động thị trường hiện tại. Với tỷ lệ rủi ro/lợi nhuận (risk_reward_ratio) là 1.21 và khả năng giảm giá sâu (downside risk 95% là -18.5%), việc mở vị thế mua mới lúc này là thiếu an toàn. Khuyến nghị đứng ngoài quan sát để chờ đợi sự ổn định của các chỉ báo kỹ thuật và xác nhận xu hướng sau khi các thông tin về tái cấu trúc được thị trường hấp thụ hoàn toàn.

## 2. Forecast & Risk
- Current Price: 228.0000
- Expected Return: -1.77%
- Risk Level: **HIGH_RISK**
- Value at Risk (95%): 18.59%
- Downside 95%: -18.59%

## 3. Monitoring Context
- **Regime:** NORMAL_VOLATILITY__UPTREND__NORMAL_VOLUME
- **Drift:** FEATURE_HIGH__TARGET_LOW__CONCEPT_LOW

## 4. Model Validation (Walk-forward)
- MAPE: 2.88%
- Directional Accuracy: 55.48%
- Interval Coverage 80%: **67.12%**
- Interval Coverage 95%: **83.56%**

## 5. Agent Workflow Logs
- **LLM Evaluation Reason:** Mô hình đang gặp tình trạng Feature Drift nghiêm trọng (16/16 tính năng bị drift cao với chỉ số PSI rất lớn), cho thấy dữ liệu đầu vào đã thay đổi cấu trúc so với tập huấn luyện. Mặc dù Concept Drift hiện tại thấp, nhưng với mức độ Feature Drift cao và rủi ro thị trường được đánh giá là HIGH_RISK, độ tin cậy của dự báo trong tương lai gần là rất thấp. Cần thực hiện tái huấn luyện (retraining) hoặc hiệu chỉnh lại các đặc trưng đầu vào.
- **News Context Found:** Yes
- **Retries Attempted:** 2
- **Reflection & Self-Correction:**
  - Thử nghiệm bị loại: Mô hình MỚI không tốt hơn mô hình CŨ. Mặc dù mô hình MỚI cải thiện được độ chệch (prediction_bias) và tỷ lệ giao cắt phân vị (quantile_crossing_rate), nhưng các chỉ số đo lường sai số chính (MAE, RMSE, Pinball Loss) đều tăng, cho thấy độ chính xác tổng thể giảm. Quan trọng hơn, độ chính xác hướng (directional_accuracy) giảm đáng kể từ 55.48% xuống 51.37% và khả năng bao phủ khoảng tin cậy (interval coverage) cũng thấp hơn, cho thấy mô hình MỚI kém tin cậy hơn trong việc dự báo xu hướng và ước lượng khoảng biến động.
  - Thử nghiệm bị loại: Mặc dù mô hình MỚI cải thiện nhẹ về sai số điểm (MAE, MAPE, SMAPE) và độ bao phủ khoảng tin cậy (Interval Coverage), nhưng nó suy giảm đáng kể về khả năng dự báo xu hướng (Directional Accuracy giảm từ 0.55 xuống 0.49). Ngoài ra, mô hình MỚI có độ chệch (Prediction Bias) cao hơn và Pinball Loss kém hơn so với mô hình CŨ. Việc đạt được Quantile Crossing Rate bằng 0 là một điểm cộng về tính nhất quán, nhưng không đủ để bù đắp cho sự sụt giảm về khả năng dự báo hướng đi của dữ liệu.
