from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    ticker: str
    run_id: str
    forecast_data: Dict[str, Any]
    quantiles_fixed: bool

    evaluation_status: str
    evaluation_reason: str
    validation_metrics: Dict[str, Any]
    regime_report: Dict[str, Any]
    drift_report: Dict[str, Any]
    risk_report: Dict[str, Any]

    news_context: str
    news_found: bool
    news_items_count: int
    news_items: List[Dict[str, Any]]
    evidence_level: str
    shock_type: str
    news_analysis: str
    news_evidence: Dict[str, Any]

    trading_signal: str
    signal_confidence: float
    governance_decision: Dict[str, Any]
    agent_assessment_summary: str
    committee_assessment: Dict[str, Any]
    assessment_summary: str
    interpretation: Any
    decision_rationale: str
    evidence_used: List[str]

    retry_count: int
    action_taken: str
    final_recommendation: str
    audit_trail: List[Dict[str, Any]]
