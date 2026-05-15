from typing import TypedDict, Dict, Any

class AgentState(TypedDict, total=False):
    ticker: str
    
    # Đầu vào từ các Layer trước
    forecast_data: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    monitoring: Dict[str, Any]  # Chứa drift, regime, risk
    current_config: Dict[str, Any]
    
    # State của Agent Workflow
    evaluation: Dict[str, Any]     # LLM đánh giá {status, reasoning}
    news_context: str              # Kết quả tìm tin tức
    retry_count: int               # Đếm số lần retrain (0, 1, 2...)
    
    # Báo cáo cuối cùng
    final_report: Dict[str, Any]   # LLM tổng hợp {action, summary, reasoning}