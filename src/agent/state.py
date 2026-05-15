from __future__ import annotations

from typing import Any, Dict, TypedDict


class AgentState(TypedDict, total=False):
    ticker: str
    run_id: str

    workflow: Dict[str, Any]
    champion: Dict[str, Any]
    challenger: Dict[str, Any]
    news: Dict[str, Any]
    improvement: Dict[str, Any]
    governance: Dict[str, Any]
    recommendation: Dict[str, Any]
    audit: Dict[str, Any]
