import json
from typing import Dict, Any

EVALUATE_PROMPT = """Bạn là một chuyên gia Quant Researcher. Hãy đánh giá hiệu suất mô hình dự báo hiện tại.
Dữ liệu:
{data}

Nhiệm vụ: Dựa vào Metrics (MAPE, Accuracy...), và trạng thái rủi ro/sai lệch (Monitoring), hãy quyết định mô hình này "OK" (đủ tốt để dùng) hay "BAD" (cần tìm tham số mới).

Hãy trả về ĐÚNG định dạng JSON sau, không kèm giải thích hay markdown code block:
{{
    "status": "OK" hoặc "BAD",
    "reasoning": "Lý do suy luận của bạn..."
}}"""

PROPOSE_CONFIG_PROMPT = """Bạn là một Kỹ sư Machine Learning. Mô hình dự báo hiện tại đang có hiệu suất chưa tốt.
Dữ liệu hiện tại:
{data}

Tin tức thị trường (Đặc biệt chú ý để bắt các cú sốc/biến động đột biến):
{news}

Cấu hình hiện tại:
{current_config}

CÁC CẤU HÌNH ĐÃ THỬ NHƯNG THẤT BẠI (Tuyệt đối KHÔNG đề xuất lại các giá trị này):
{rejected_configs}

Ngưỡng cho phép:
- learning_rate: [0.005, 0.2]
- max_depth: [3, 12]
- num_leaves: [16, 256]
- min_child_samples: [5, 100]

Nhiệm vụ: Đề xuất bộ cấu hình LightGBM mới (KHÁC VỚI CÁC CẤU HÌNH ĐÃ THẤT BẠI) nằm trong ngưỡng cho phép để cải thiện mô hình. 
Trả về ĐÚNG định dạng JSON sau:
{{
    "learning_rate": float,
    "max_depth": int,
    "num_leaves": int,
    "min_child_samples": int,
    "reasoning": "Lý do thay đổi cấu hình dựa trên metrics và tin tức..."
}}"""

COMPARE_PROMPT = """Bạn là một Giám khảo AI. Hãy so sánh kết quả Walk-forward validation của mô hình CŨ và mô hình MỚI.
Mô hình CŨ:
{old_metrics}

Mô hình MỚI:
{new_metrics}

Nhiệm vụ: Mô hình MỚI có thực sự tốt hơn (dự báo chính xác hơn, tin cậy hơn) so với mô hình CŨ không?
Trả về ĐÚNG định dạng JSON sau:
{{
    "is_better": true hoặc false,
    "reasoning": "Lý do của bạn..."
}}"""

FINAL_REPORT_PROMPT = """Bạn là một Giám đốc Đầu tư định lượng. Dựa vào CÁC DỮ LIỆU ĐƯỢC CUNG CẤP DƯỚI ĐÂY:
Mã cổ phiếu: {ticker}

Dự báo 7 ngày:
{forecast}

Đánh giá rủi ro và trạng thái thị trường:
{monitoring}

Tin tức mới nhất trong 7 ngày qua:
{news}

QUY TẮC PHÂN TÍCH:
1. Quyết định (BUY/SELL/HOLD/WATCH) phải có sự dung hòa giữa số liệu định lượng (Risk, Expected Return, Trend) và thông tin định tính (News).
2. Nếu Tin tức ghi "Không có tin tức", tuyệt đối không tự bịa ra tin.
3. Nếu rủi ro (Risk Level) là EXTREME_RISK hoặc HIGH_RISK, cân nhắc cẩn trọng chiều BUY, ưu tiên quản trị rủi ro.
4. LƯU Ý QUAN TRỌNG: T+1 đến T+7 đại diện cho "các phiên giao dịch tiếp theo", tuyệt đối không quy đổi ra thứ ngày tháng cụ thể (ví dụ không nói "ngày mai thứ 7").

Nhiệm vụ: Tổng hợp tình hình và đưa ra khuyến nghị hành động nghiên cứu cuối cùng.
Trả về ĐÚNG định dạng JSON sau:
{{
    "action": "BUY" | "SELL" | "HOLD" | "WATCH",
    "summary": "Tóm tắt ngắn gọn tình hình hiện tại...",
    "reasoning": "Luận điểm đầu tư chi tiết và logic..."
}}"""

def format_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)