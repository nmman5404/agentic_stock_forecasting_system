from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from langgraph.graph import END, StateGraph

from src.agent.nodes import node_contextualize, node_evaluate, node_improve, node_recommend, node_validate
from src.agent.state import AgentState
from utils.logger import get_logger

logger = get_logger("AgentGraph")


def load_config() -> Dict[str, Any]:
    with Path("configs/agent_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def route_after_evaluate(state: AgentState) -> str:
    if state.get("evaluation_status") == "PASS":
        return "node_recommend"
    if state.get("retry_count", 0) > 0:
        logger.info("Graph routing | from=evaluate_model | to=model_governance | reason=revalidation_after_retry")
        return "node_improve"
    logger.info("Graph routing | from=evaluate_model | to=contextualize_news | reason=abnormal_model_status")
    return "node_contextualize"


def route_after_contextualize(state: AgentState) -> str:
    shock_type = state.get("shock_type", "NO_NEWS")
    if shock_type in {"BLACK_SWAN", "DATA_ISSUE"}:
        logger.warning("Graph routing | from=contextualize_news | to=research_signal | shock_type=%s", shock_type)
        return "node_recommend"
    logger.info("Graph routing | from=contextualize_news | to=model_governance | shock_type=%s", shock_type)
    return "node_improve"


def route_after_improve(state: AgentState) -> str:
    max_retries = load_config()["thresholds"]["max_retries"]
    if state.get("retry_count", 0) >= max_retries:
        logger.warning(
            "Graph routing | from=model_governance | to=research_signal | reason=max_retries_reached | max_retries=%s",
            max_retries,
        )
        return "node_recommend"
    logger.info("Graph routing | from=model_governance | to=validate_quantiles | reason=revalidate_challenger")
    return "node_validate"


def build_agent_graph():
    logger.info("Graph build started | workflow=agentic_stock_forecasting")
    workflow = StateGraph(AgentState)
    workflow.add_node("node_validate", node_validate)
    workflow.add_node("node_evaluate", node_evaluate)
    workflow.add_node("node_contextualize", node_contextualize)
    workflow.add_node("node_improve", node_improve)
    workflow.add_node("node_recommend", node_recommend)

    workflow.set_entry_point("node_validate")
    workflow.add_edge("node_validate", "node_evaluate")
    workflow.add_conditional_edges(
        "node_evaluate",
        route_after_evaluate,
        {
            "node_recommend": "node_recommend",
            "node_contextualize": "node_contextualize",
            "node_improve": "node_improve",
        },
    )
    workflow.add_conditional_edges(
        "node_contextualize",
        route_after_contextualize,
        {
            "node_recommend": "node_recommend",
            "node_improve": "node_improve",
        },
    )
    workflow.add_conditional_edges(
        "node_improve",
        route_after_improve,
        {
            "node_validate": "node_validate",
            "node_recommend": "node_recommend",
        },
    )
    workflow.add_edge("node_recommend", END)
    return workflow.compile()
