from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    node_compare_models,
    node_evaluate_challenger,
    node_evaluate_monitoring,
    node_generate_report,
    node_plan_retrain,
    node_search_news_context,
    node_train_challenger,
    node_validate_forecast,
    node_validate_or_repair_patch,
)
from src.agent.state import AgentState
from utils.logger import get_logger

logger = get_logger("AgentGraph")


def route_after_evaluate(state: AgentState) -> str:
    health = state.get("workflow", {}).get("model_health_status", "MANUAL_REVIEW")
    if health == "OK":
        logger.info("Graph routing | from=evaluate_monitoring | to=generate_report | health=OK")
        return "node_generate_report"
    logger.info("Graph routing | from=evaluate_monitoring | to=search_news_context | health=%s", health)
    return "node_search_news_context"


def route_after_patch(state: AgentState) -> str:
    improvement = state.get("improvement", {})
    if improvement.get("config_patch_valid") is True and improvement.get("validated_config_patch"):
        logger.info("Graph routing | from=validate_or_repair_patch | to=train_challenger | valid=True")
        return "node_train_challenger"
    logger.info("Graph routing | from=validate_or_repair_patch | to=generate_report | valid=False")
    return "node_generate_report"


def build_agent_graph():
    logger.info("Graph build started | workflow=champion_optional_challenger")
    workflow = StateGraph(AgentState)

    workflow.add_node("node_validate_forecast", node_validate_forecast)
    workflow.add_node("node_evaluate_monitoring", node_evaluate_monitoring)
    workflow.add_node("node_search_news_context", node_search_news_context)
    workflow.add_node("node_plan_retrain", node_plan_retrain)
    workflow.add_node("node_validate_or_repair_patch", node_validate_or_repair_patch)
    workflow.add_node("node_train_challenger", node_train_challenger)
    workflow.add_node("node_evaluate_challenger", node_evaluate_challenger)
    workflow.add_node("node_compare_models", node_compare_models)
    workflow.add_node("node_generate_report", node_generate_report)

    workflow.set_entry_point("node_validate_forecast")
    workflow.add_edge("node_validate_forecast", "node_evaluate_monitoring")
    workflow.add_conditional_edges(
        "node_evaluate_monitoring",
        route_after_evaluate,
        {
            "node_generate_report": "node_generate_report",
            "node_search_news_context": "node_search_news_context",
        },
    )
    workflow.add_edge("node_search_news_context", "node_plan_retrain")
    workflow.add_edge("node_plan_retrain", "node_validate_or_repair_patch")
    workflow.add_conditional_edges(
        "node_validate_or_repair_patch",
        route_after_patch,
        {
            "node_train_challenger": "node_train_challenger",
            "node_generate_report": "node_generate_report",
        },
    )
    workflow.add_edge("node_train_challenger", "node_evaluate_challenger")
    workflow.add_edge("node_evaluate_challenger", "node_compare_models")
    workflow.add_edge("node_compare_models", "node_generate_report")
    workflow.add_edge("node_generate_report", END)
    return workflow.compile()
