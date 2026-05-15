# Quant Research Report: VIC (2026-05-15)

## 1. Executive Summary
- **Final Action:** HOLD
- **Model Status:** BAD (Retries: 2)
- **Summary:** VIC đang trong xu hướng tăng (uptrend) với các tin tức cơ bản tích cực về tái cấu trúc tài chính, tuy nhiên mô hình định lượng cảnh báo rủi ro cao và sự lệch pha dữ liệu (drift) đáng kể.

### Luận điểm đầu tư (Reasoning):
Về mặt định tính, Vingroup đang có những bước đi chiến lược quan trọng như tất toán trái phiếu và mở rộng sang lĩnh vực y tế công nghệ cao, tạo tâm lý tích cực cho thị trường. Tuy nhiên, về mặt định lượng, hệ thống ghi nhận mức độ 'Feature Drift' ở mức CAO (HIGH) trên 16 chỉ báo, cho thấy dữ liệu hiện tại đang biến động mạnh và không còn khớp hoàn toàn với mô hình dự báo cũ. Mặc dù tỷ lệ Risk/Reward là 1.37, nhưng mức rủi ro (Risk Level) đang ở ngưỡng HIGH_RISK với VaR 95% lên tới 17.56%. Với sự không chắc chắn từ dữ liệu (drift) và rủi ro đuôi (tail loss) cao, chiến lược thận trọng là nắm giữ (HOLD) để quan sát sự ổn định của xu hướng sau khi các tin tức tích cực đã phản ánh vào giá, thay vì mở vị thế mua mới trong giai đoạn biến động này.

## 2. Forecast & Risk
- Current Price: 228.0000
- Expected Return: 0.96%
- Risk Level: **HIGH_RISK**
- Value at Risk (95%): 17.57%
- Downside 95%: -17.57%

## 3. Monitoring Context
- **Regime:** NORMAL_VOLATILITY__UPTREND__NORMAL_VOLUME
- **Drift:** FEATURE_HIGH__TARGET_LOW__CONCEPT_MEDIUM

## 4. Model Validation (Walk-forward)
- MAPE: 2.89%
- Directional Accuracy: 53.42%
- Interval Coverage 95%: 83.56%

## 5. Agent Workflow Logs
- LLM Evaluation Reason: Mô hình đang gặp vấn đề nghiêm trọng về tính ổn định. Mặc dù các chỉ số sai số (MAPE ~2.8%) có vẻ thấp, nhưng hệ thống ghi nhận 'Feature Drift' ở mức CAO trên toàn bộ 16/16 đặc trưng (PSI cao, Mean shift đáng kể) và 'Concept Drift' ở mức TRUNG BÌNH. Điều này cho thấy mô hình đã mất khả năng nắm bắt phân phối dữ liệu hiện tại. Ngoài ra, Directional Accuracy chỉ đạt 53.4% (gần mức ngẫu nhiên) và rủi ro được đánh giá là 'HIGH_RISK' với các ước tính đuôi (tail loss) cao, cho thấy mô hình không còn tin cậy trong điều kiện thị trường hiện tại.
- News Context Found: Yes
