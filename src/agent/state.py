from typing import TypedDict, Dict, Any, List

class AgentState(TypedDict, total=False):
    ticker: str
    forecast_data: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    monitoring: Dict[str, Any]  
    current_config: Dict[str, Any]
    
    evaluation: Dict[str, Any]     
    news_context: str              
    retry_count: int               
    
    rejected_configs: List[Dict[str, Any]] 
    
    final_report: Dict[str, Any]