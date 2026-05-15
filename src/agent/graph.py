from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    node_fix_quantiles,
    node_llm_evaluate,
    node_search_news,
    node_llm_retrain_compare,
    node_generate_final_report
)
from utils.logger import get_logger

logger = get_logger("AgentGraph")

MAX_RETRIES = 2

def route_evaluate(state: AgentState) -> str:
    """Phân luồng sau khi LLM đánh giá hiệu suất."""
    status = state.get("evaluation", {}).get("status", "OK")
    retry_count = state.get("retry_count", 0)
    
    if status == "OK" or retry_count >= MAX_RETRIES:
        logger.info("Routing | To: generate_final_report | status=%s | retry=%d", status, retry_count)
        return "node_generate_final_report"
        
    if not state.get("news_context"):
        logger.info("Routing | To: search_news | news=empty")
        return "node_search_news"
        
    logger.info("Routing | To: llm_retrain_compare | news=exist")
    return "node_llm_retrain_compare"

def build_agent_graph():
    logger.info("Building Agent Graph (LLM-driven workflow)")
    workflow = StateGraph(AgentState)

    # Thêm các Node
    workflow.add_node("node_fix_quantiles", node_fix_quantiles)
    workflow.add_node("node_llm_evaluate", node_llm_evaluate)
    workflow.add_node("node_search_news", node_search_news)
    workflow.add_node("node_llm_retrain_compare", node_llm_retrain_compare)
    workflow.add_node("node_generate_final_report", node_generate_final_report)

    # Định nghĩa các Edge (Đường nối)
    workflow.set_entry_point("node_fix_quantiles")
    
    workflow.add_edge("node_fix_quantiles", "node_llm_evaluate")
    
    # Conditional Routing
    workflow.add_conditional_edges(
        "node_llm_evaluate",
        route_evaluate,
        {
            "node_generate_final_report": "node_generate_final_report",
            "node_search_news": "node_search_news",
            "node_llm_retrain_compare": "node_llm_retrain_compare"
        }
    )
    
    workflow.add_edge("node_search_news", "node_llm_retrain_compare")
    
    # Vòng lặp: Sau khi retrain và so sánh, quay lại bước fix quantiles
    workflow.add_edge("node_llm_retrain_compare", "node_fix_quantiles")
    
    # Kết thúc
    workflow.add_edge("node_generate_final_report", END)

    return workflow.compile()