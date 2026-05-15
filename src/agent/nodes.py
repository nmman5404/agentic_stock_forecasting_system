import json
import re
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from src.agent.state import AgentState
from src.agent.tools import tool_search_google_news
from src.agent.prompts import (
    EVALUATE_PROMPT, PROPOSE_CONFIG_PROMPT, COMPARE_PROMPT, FINAL_REPORT_PROMPT, format_json
)
from src.modeling.predictor import generate_7_day_forecast
from src.processing.db_manager import load_from_sqlite
from utils.logger import get_logger

logger = get_logger("AgentNodes")
_LLM = None

def _get_llm() -> ChatGoogleGenerativeAI:
    global _LLM
    if _LLM is None:
        load_dotenv()
        _LLM = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
    return _LLM

def _extract_text(content: Any) -> str:
    """Trích xuất text an toàn từ Langchain AIMessage content"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)

def _ask_gemini(prompt: str) -> Dict[str, Any]:
    try:
        response_msg = _get_llm().invoke([
            SystemMessage(content="Return pure JSON only. No markdown, no chain-of-thought."),
            HumanMessage(content=prompt)
        ])
        
        # Bước 1: Trích xuất text an toàn (Fix lỗi list)
        raw_text = _extract_text(response_msg.content)
        
        # Bước 2: Xóa thẻ markdown json
        clean_text = re.sub(r"```(?:json)?\n?|```", "", raw_text).strip()
        
        return json.loads(clean_text)
    except Exception as exc:
        logger.error("Gemini request failed | error=%s", exc)
        return {}

def node_fix_quantiles(state: AgentState) -> AgentState:
    """Đảm bảo các dải lượng tử (quantiles) không bị cắt chéo nhau."""
    logger.info("Node execution | fix_quantiles")
    forecast_data = state.get("forecast_data", {})
    keys = ["q_0.025", "q_0.1", "q_0.5", "q_0.9", "q_0.975"]
    
    for step in forecast_data.get("forecasts", []):
        if all(k in step for k in keys):
            values = sorted([step[k] for k in keys])
            for k, v in zip(keys, values):
                step[k] = v
                
    state["forecast_data"] = forecast_data
    return state

def node_llm_evaluate(state: AgentState) -> AgentState:
    """LLM tự đánh giá mô hình dựa trên Metrics và Monitoring."""
    logger.info("Node execution | llm_evaluate")
    data_context = {
        "metrics": state.get("validation_metrics", {}).get("metrics", {}),
        "monitoring": state.get("monitoring", {})
    }
    prompt = EVALUATE_PROMPT.format(data=format_json(data_context))
    
    result = _ask_gemini(prompt)
    # Default to OK if parsing fails to avoid infinite loops
    state["evaluation"] = {
        "status": result.get("status", "OK").upper(),
        "reasoning": result.get("reasoning", "No reasoning provided.")
    }
    logger.info("LLM Evaluation | status=%s", state["evaluation"]["status"])
    return state

def node_search_news(state: AgentState) -> AgentState:
    """Dùng tool để tìm tin tức nếu có lỗi/mô hình chưa tốt."""
    logger.info("Node execution | search_news")
    ticker = state.get("ticker", "UNKNOWN")
    state["news_context"] = tool_search_google_news(ticker)
    return state

def node_llm_retrain_compare(state: AgentState) -> AgentState:
    """LLM đề xuất config -> Code Retrain -> LLM So sánh -> Cập nhật State."""
    logger.info("Node execution | llm_retrain_compare")
    ticker = state.get("ticker", "UNKNOWN")
    
    # 1. Xin config mới
    prompt_propose = PROPOSE_CONFIG_PROMPT.format(
        data=format_json(state.get("validation_metrics", {}).get("metrics")),
        news=state.get("news_context", "NO_NEWS"),
        current_config=format_json(state.get("current_config", {}))
    )
    new_params = _ask_gemini(prompt_propose)
    
    # Nếu LLM không trả về đủ tham số chuẩn thì bỏ qua, tăng retry
    if not all(k in new_params for k in ["learning_rate", "max_depth", "num_leaves"]):
        logger.warning("LLM proposed invalid config parameters.")
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state

    merged_config = {**state.get("current_config", {}), **new_params}
    if "reasoning" in merged_config:
        del merged_config["reasoning"]

    # 2. Retrain
    df_processed = load_from_sqlite(f"processed_{ticker}")
    new_forecast = generate_7_day_forecast(df_processed, model_params=merged_config)
    new_forecast["ticker"] = ticker
    new_metrics = new_forecast.get("validation_metrics", {})
    
    # 3. LLM So sánh
    prompt_compare = COMPARE_PROMPT.format(
        old_metrics=format_json(state.get("validation_metrics", {}).get("metrics")),
        new_metrics=format_json(new_metrics.get("metrics"))
    )
    compare_result = _ask_gemini(prompt_compare)
    is_better = compare_result.get("is_better", False)
    
    
    # 4. Quyết định
    logger.info("Retrain attempt complete | is_better=%s | reasoning=%s", is_better, compare_result.get("reasoning"))
    if is_better:
        state["forecast_data"] = new_forecast
        state["validation_metrics"] = new_metrics
        state["current_config"] = merged_config

        import yaml
        with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f) or {}
            
        full_config["lightgbm_params"] = merged_config
        
        with open("configs/model_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(full_config, f, sort_keys=False)

    state["retry_count"] = state.get("retry_count", 0) + 1
    return state

def node_generate_final_report(state: AgentState) -> AgentState:
    """LLM tổng hợp thông tin và ra quyết định hành động cuối cùng."""
    logger.info("Node execution | generate_final_report")
    prompt = FINAL_REPORT_PROMPT.format(
        ticker=state.get("ticker", "UNKNOWN"),
        forecast=format_json(state.get("forecast_data", {}).get("forecasts")),
        monitoring=format_json(state.get("monitoring", {})),
        news=state.get("news_context", "NO_NEWS")
    )
    
    result = _ask_gemini(prompt)
    state["final_report"] = {
        "action": result.get("action", "MANUAL_REVIEW").upper(),
        "summary": result.get("summary", "N/A"),
        "reasoning": result.get("reasoning", "N/A")
    }
    logger.info("Final Report | action=%s", state["final_report"]["action"])
    return state