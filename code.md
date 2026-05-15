configs/model_config.yaml

```yaml
lightgbm_params:
  boosting_type: gbdt
  learning_rate: 0.019999999999999997
  max_depth: 2
  n_estimators: 40
  num_leaves: 3
  random_state: 42
  verbose: -1
walk_forward_validation:
  horizon: 1
  initial_train_size: 252
  max_windows: 8
  quantiles:
  - 0.025
  - 0.1
  - 0.5
  - 0.9
  - 0.975
  step_size: 20
  validation_window: 20
```

main.py

```python
from src.orchestration.daily_pipeline import run_daily_pipeline


if __name__ == "__main__":
    run_daily_pipeline()
```

requirements.txt

```text
pandas
numpy
scikit-learn
lightgbm
sqlalchemy
vnstock
ta
langchain
langchain-core
langgraph
langchain-google-genai
python-dotenv
pyyaml
plotly
feedparser
```

src/__init__.py

```python

```

src/agent/__init__.py

```python

```

src/agent/graph.py

```python
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
```

src/agent/nodes.py

```python
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
```

src/agent/prompts.py

```python
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

Ngưỡng cho phép:
- learning_rate: [0.005, 0.2]
- max_depth: [3, 12]
- num_leaves: [16, 256]
- min_child_samples: [5, 100]

Nhiệm vụ: Đề xuất bộ cấu hình LightGBM mới nằm trong ngưỡng cho phép để mô hình thích ứng tốt hơn với trạng thái thị trường hiện tại. 
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

FINAL_REPORT_PROMPT = """Bạn là một Giám đốc Đầu tư (CIO) định lượng. Dựa vào CÁC DỮ LIỆU ĐƯỢC CUNG CẤP DƯỚI ĐÂY:
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

Nhiệm vụ: Tổng hợp tình hình và đưa ra khuyến nghị hành động nghiên cứu cuối cùng.
Trả về ĐÚNG định dạng JSON sau:
{{
    "action": "BUY" | "SELL" | "HOLD" | "WATCH",
    "summary": "Tóm tắt ngắn gọn tình hình hiện tại...",
    "reasoning": "Luận điểm đầu tư chi tiết và logic..."
}}"""

def format_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
```

src/agent/state.py

```python
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
```

src/agent/tools.py

```python
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from utils.logger import get_logger

logger = get_logger("AgentTools")

def _get_queries_for_ticker(ticker: str) -> list[str]:
    """Map mã cổ phiếu với các truy vấn liên quan để tối đa hóa độ phủ tin tức."""
    query_map = {
        "VIC": ['"VIC" "Vingroup"', '"VinFast" "Vingroup"', '"cổ phiếu VIC"'],
        "VHM": ['"VHM" "Vinhomes"', '"Vinhomes" "cổ phiếu"'],
        "VRE": ['"VRE" "Vincom Retail"', '"Vincom Retail" "cổ phiếu"'],
        "VPL": ['"VPL" "Vinpearl"', '"Vinpearl" "Vingroup"'],
    }
    return query_map.get(ticker, [f'"{ticker}" "cổ phiếu"', f'"{ticker}" "chứng khoán"'])

def tool_search_google_news(ticker: str, max_items: int = 5, lookback_days: int = 7) -> str:
    """Tìm kiếm tin tức trên Google News RSS, có lọc trùng lặp và giới hạn thời gian."""
    logger.info("Calling Google News RSS Tool | ticker=%s", ticker)
    queries = _get_queries_for_ticker(ticker)
    
    all_items = []
    seen_titles = set()
    
    # Tính thời điểm cắt đứt (cutoff) để không lấy tin quá cũ
    cutoff_date = datetime.now().astimezone() - timedelta(days=lookback_days)

    for q in queries:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(q)}&hl=vi&gl=VN&ceid=VN:vi"
        try:
            feed = feedparser.parse(url)
            for entry in getattr(feed, "entries", []):
                title = entry.get('title', '').strip()
                if not title or title in seen_titles:
                    continue
                
                # Xử lý thời gian và lọc tin cũ
                pub_str = entry.get('published', entry.get('updated', ''))
                try:
                    pub_dt = parsedate_to_datetime(pub_str)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.astimezone()
                    if pub_dt < cutoff_date:
                        continue  # Bỏ qua tin cũ hơn 7 ngày
                except Exception:
                    pass # Nếu không parse được ngày thì vẫn giữ lại cho an toàn
                
                seen_titles.add(title)
                all_items.append(f"- [{pub_str}] {title}")
                
        except Exception as exc:
            logger.warning("Google News Query failed | query=%s | error=%s", q, exc)

    if not all_items:
        logger.info("Google News RSS Tool | No recent news found for %s", ticker)
        return "Không có tin tức nào mới trên thị trường trong 7 ngày qua."
    
    logger.info("Google News RSS Tool | Found %d unique recent items for %s", len(all_items), ticker)
    
    # Trả về tối đa max_items bài báo mới nhất
    return "\n".join(all_items[:max_items])
```

src/ingestion/__init__.py

```python

```

src/ingestion/vnstock_api.py

```python
import pandas as pd
from pathlib import Path
from vnstock import Quote
from utils.logger import get_logger

logger = get_logger("DataIngestion")
RAW_CSV_DIR = Path("data/raw/csv")

def fetch_historical_data(symbols: list, start_date: str, end_date: str, data_type: str = "stock") -> dict:
    """
    Lấy dữ liệu lịch sử cho một danh sách mã chứng khoán/chỉ số.
    
    Args:
        symbols: list các mã (VD: ['VIC', 'VHM'] hoặc ['VN30'])
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
        data_type: 'stock', 'index', hoặc 'derivative'
        
    Returns:
        dict: Dictionary chứa Pandas DataFrame của từng mã. VD: {'VIC': df, 'VHM': df}
    """
    data_dict = {}
    
    for sym in symbols:
        try:
            logger.info(
                "Data fetch started | symbol=%s | data_type=%s | start_date=%s | end_date=%s",
                sym,
                data_type,
                start_date,
                end_date,
            )
            
            # Gọi hàm của vnstock
            df = Quote(source="VCI", symbol=sym, show_log=False).history(
                start=start_date,
                end=end_date,
                interval="1D",
            )
            
            if df is not None and not df.empty:
                # vnstock trả về cột 'time', ta đổi tên thành 'date' 
                if 'time' in df.columns:
                    df.rename(columns={'time': 'date'}, inplace=True)
                
                # Ép kiểu dữ liệu ngày tháng và đặt làm Index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Sắp xếp lại theo thời gian từ cũ đến mới
                df.sort_index(inplace=True)
                
                data_dict[sym] = df
                logger.info("Data fetch completed | symbol=%s | rows=%s", sym, len(df))
            else:
                logger.warning("Data fetch returned no rows | symbol=%s", sym)
                
        except Exception as e:
            logger.error("Data fetch failed | symbol=%s | error=%s", sym, str(e))
            
    return data_dict

def get_vingroup_and_context_data(start_date: str, end_date: str) -> dict:
    """
    Hàm tổng hợp: Kéo dữ liệu cổ phiếu Vingroup + Context (VN30, Phái sinh).
    """
    logger.info("Ingestion phase started | source=vnstock")
    
    # 1. Kéo cổ phiếu họ Vingroup
    vingroup_symbols = ["VIC", "VHM", "VRE", "VPL"]
    stock_data = fetch_historical_data(vingroup_symbols, start_date, end_date, data_type="stock")
    
    # 2. Kéo Context: Chỉ số VN30
    vn30_data = fetch_historical_data(["VN30"], start_date, end_date, data_type="index")
    
    # 3. Kéo Context: Phái sinh VN30F1M
    derivative_data = fetch_historical_data(["VN30F1M"], start_date, end_date, data_type="derivative")
    
    # Gộp tất cả vào 1 dictionary duy nhất
    all_data = {**stock_data, **vn30_data, **derivative_data}
    save_raw_csv_snapshots(all_data, RAW_CSV_DIR)

    logger.info("Ingestion phase completed | symbols_loaded=%s", len(all_data))
    return all_data

def save_raw_csv_snapshots(all_data: dict, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for symbol, df in all_data.items():
        if df is None or df.empty:
            logger.warning("Raw CSV snapshot skipped | symbol=%s | reason=empty_dataframe", symbol)
            continue

        csv_df = df.copy()
        if csv_df.index.name == "date" or "date" not in csv_df.columns:
            csv_df = csv_df.reset_index()
        if "date" in csv_df.columns:
            csv_df["date"] = pd.to_datetime(csv_df["date"]).dt.strftime("%Y-%m-%d")
        csv_df["ticker"] = symbol

        preferred_columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
        ordered_columns = [col for col in preferred_columns if col in csv_df.columns]
        ordered_columns.extend(col for col in csv_df.columns if col not in ordered_columns)
        csv_df = csv_df[ordered_columns]

        path = output_dir / f"{symbol}_raw.csv"
        csv_df.to_csv(path, index=False)
        saved_paths.append(path)

    if saved_paths:
        logger.info(
            "Raw CSV snapshots saved:\n%s", saved_paths
        )
    else:
        logger.warning("Raw CSV snapshots skipped | reason=no_dataframes_saved")

    return saved_paths
```

src/modeling/__init__.py

```python

```

src/modeling/predictor.py

```python
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.modeling.trainer import QuantileLightGBM
from src.modeling.validation import run_walk_forward_validation
from utils.logger import get_logger

logger = get_logger("Predictor")


def generate_7_day_forecast(
    df: pd.DataFrame,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a 7-day direct multi-step quantile forecast.

    Walk-forward validation is the only model-evaluation source of truth.
    """
    ticker = df["ticker"].iloc[0] if "ticker" in df.columns and not df.empty else "Stock"
    logger.info("Forecast generation started | ticker=%s | horizon_days=7", ticker)

    trainer = QuantileLightGBM(model_params=model_params)
    validation_metrics = run_walk_forward_validation(
        df,
        target_col=trainer.target_col,
        model_params=trainer.config,
    )

    last_row = df.iloc[[-1]]
    x_last = last_row[[col for col in df.columns if col not in {"ticker", "date", "target"}]]

    forecasts = []
    logger.info("Quantile forecast training started | ticker=%s | steps=7 | quantiles=5", ticker)
    for step in range(1, 8):
        forecasts.append(trainer.train_and_predict_step(df, x_last, step))

    logger.info("Forecast generation completed | ticker=%s | horizon_days=7", ticker)
    return {
        "ticker": ticker,
        "current_price": float(df["close"].iloc[-1]) if "close" in df.columns and not df.empty else None,
        "as_of_date": df.index[-1].strftime("%Y-%m-%d") if len(df.index) else None,
        "evaluation_method": "walk_forward",
        "validation_metrics": validation_metrics,
        "forecasts": forecasts,
    }
```

src/modeling/trainer.py

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from utils.logger import get_logger

logger = get_logger("ModelTrainer")


def load_config() -> Dict[str, Any]:
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class QuantileLightGBM:
    """LightGBM quantile forecaster that predicts future returns, then maps them to prices."""

    def __init__(self, target_col: str = "close", model_params: Optional[Dict[str, Any]] = None):
        self.target_col = target_col
        full_config = load_config()
        self.config = model_params.copy() if model_params is not None else full_config.get("lightgbm_params", {})
        
        # Đọc động quantiles từ config thay vì hardcode
        wf_config = full_config.get("walk_forward_validation", {})
        self.quantiles = wf_config.get("quantiles", [0.025, 0.1, 0.5, 0.9, 0.975])
        
        self.features: List[str] = []

    def prepare_data(self, df: pd.DataFrame, step: int = 1):
        """Create supervised data for direct multi-step return forecasting."""
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' is missing from training data.")

        ordered = df.copy()
        if isinstance(ordered.index, pd.DatetimeIndex):
            ordered = ordered.sort_index()

        excluded = {"ticker", "date", "target"}
        candidate_features = [col for col in ordered.columns if col not in excluded]
        self.features = [
            col for col in candidate_features if pd.api.types.is_numeric_dtype(ordered[col])
        ]

        supervised = ordered[self.features].copy()
        supervised["target"] = (
            ordered[self.target_col].shift(-step) - ordered[self.target_col]
        ) / ordered[self.target_col]
        supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])
        supervised = supervised.dropna(axis=0)
        return supervised[self.features], supervised["target"]

    def train_and_predict_step(self, df: pd.DataFrame, x_last: pd.DataFrame, step: int) -> Dict[str, float]:
        X, y = self.prepare_data(df, step=step)
        if X.empty or y.empty:
            raise ValueError(f"Insufficient training rows for forecast step {step}.")

        current_close = float(df[self.target_col].iloc[-1])
        x_last = x_last.reindex(columns=self.features).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        step_forecast: Dict[str, float] = {"step": int(step)}
        for quantile in self.quantiles:
            params = self.config.copy()
            params["objective"] = "quantile"
            params["alpha"] = float(quantile)

            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)

            predicted_return = float(model.predict(x_last)[0])
            step_forecast[f"q_{quantile}"] = current_close * (1.0 + predicted_return)

        return step_forecast
```

src/modeling/validation.py

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from utils.logger import get_logger

logger = get_logger("WalkForwardValidation")


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_size: int = 252
    validation_window: int = 20
    step_size: int = 20
    max_windows: Optional[int] = 8
    horizon: int = 1
    quantiles: Tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)


@dataclass(frozen=True)
class ValidationMetrics:
    mae: float
    rmse: float
    mape: float
    smape: float
    directional_accuracy: float
    interval_80_coverage: float
    interval_95_coverage: float
    interval_coverage: float
    pinball_loss: float
    prediction_bias: float
    prediction_bias_pct: float
    quantile_crossing_rate: float


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    train_size: int
    validation_size: int
    metrics: ValidationMetrics


@dataclass(frozen=True)
class ValidationReport:
    evaluation_method: str
    status: str
    target_col: str
    horizon: int
    sample_count: int
    feature_count: int
    fold_count: int
    config: WalkForwardConfig
    metrics: Optional[ValidationMetrics]
    folds: List[FoldMetrics]
    notes: List[str]
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_walk_forward_config(config_path: Path = Path("configs/model_config.yaml")) -> WalkForwardConfig:
    if not config_path.exists():
        logger.warning("Walk-forward config file not found. Using defaults.")
        return WalkForwardConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    raw_validation = raw_config.get("walk_forward_validation", {})
    if not isinstance(raw_validation, dict):
        return WalkForwardConfig()

    defaults = WalkForwardConfig()
    quantiles = tuple(raw_validation.get("quantiles", defaults.quantiles))
    return WalkForwardConfig(
        initial_train_size=int(raw_validation.get("initial_train_size", defaults.initial_train_size)),
        validation_window=int(raw_validation.get("validation_window", defaults.validation_window)),
        step_size=int(raw_validation.get("step_size", defaults.step_size)),
        max_windows=raw_validation.get("max_windows", defaults.max_windows),
        horizon=int(raw_validation.get("horizon", defaults.horizon)),
        quantiles=quantiles,
    )


def load_model_params(config_path: Path = Path("configs/model_config.yaml")) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    params = raw_config.get("lightgbm_params", {})
    if not isinstance(params, dict):
        raise ValueError("configs/model_config.yaml must contain a lightgbm_params mapping.")

    return params.copy()


def run_walk_forward_validation(
    df: pd.DataFrame,
    target_col: str = "close",
    config: Optional[WalkForwardConfig] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report = walk_forward_validate(
        df=df,
        target_col=target_col,
        config=config or load_walk_forward_config(),
        model_params=model_params or load_model_params(),
    )
    return report.to_dict()


def walk_forward_validate(
    df: pd.DataFrame,
    target_col: str = "close",
    config: Optional[WalkForwardConfig] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> ValidationReport:
    config = config or load_walk_forward_config()
    model_params = model_params or load_model_params()

    X, y_return, current_close, actual_price, feature_cols = _prepare_supervised_data(
        df=df,
        target_col=target_col,
        horizon=config.horizon,
    )

    sample_count = len(X)
    feature_count = len(feature_cols)
    min_required = max(30, config.validation_window + 2)
    if sample_count < min_required or feature_count == 0:
        message = (
            f"Insufficient validation data: samples={sample_count}, "
            f"features={feature_count}, required_samples>={min_required}."
        )
        logger.warning(message)
        return ValidationReport(
            evaluation_method="walk_forward",
            status="INSUFFICIENT_DATA",
            target_col=target_col,
            horizon=config.horizon,
            sample_count=sample_count,
            feature_count=feature_count,
            fold_count=0,
            config=config,
            metrics=None,
            folds=[],
            notes=["Walk-forward evaluation could not run because the dataset is too small."],
            message=message,
        )

    initial_train_size = _resolve_initial_train_size(config, sample_count)
    fold_starts = list(range(initial_train_size, sample_count, config.step_size))
    if config.max_windows is not None and len(fold_starts) > config.max_windows:
        fold_starts = fold_starts[-int(config.max_windows):]

    folds: List[FoldMetrics] = []
    all_predictions: List[pd.DataFrame] = []

    for fold_number, train_end in enumerate(fold_starts, start=1):
        validation_start = train_end
        validation_end = min(train_end + config.validation_window, sample_count)
        if validation_end <= validation_start:
            continue

        X_train = X.iloc[:train_end]
        y_train = y_return.iloc[:train_end]
        X_val = X.iloc[validation_start:validation_end]

        prediction_frame = _predict_quantile_prices(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            current_close=current_close.iloc[validation_start:validation_end],
            actual_price=actual_price.iloc[validation_start:validation_end],
            quantiles=config.quantiles,
            model_params=model_params,
        )
        fold_metric_values = _calculate_metrics(prediction_frame, config.quantiles)

        folds.append(
            FoldMetrics(
                fold=fold_number,
                train_start=_format_index_value(X.index[0]),
                train_end=_format_index_value(X.index[train_end - 1]),
                validation_start=_format_index_value(X.index[validation_start]),
                validation_end=_format_index_value(X.index[validation_end - 1]),
                train_size=len(X_train),
                validation_size=len(X_val),
                metrics=fold_metric_values,
            )
        )
        all_predictions.append(prediction_frame)

    if not all_predictions:
        message = "No walk-forward folds were generated."
        logger.warning(message)
        return ValidationReport(
            evaluation_method="walk_forward",
            status="INSUFFICIENT_DATA",
            target_col=target_col,
            horizon=config.horizon,
            sample_count=sample_count,
            feature_count=feature_count,
            fold_count=0,
            config=config,
            metrics=None,
            folds=[],
            notes=["Walk-forward evaluation generated no validation folds."],
            message=message,
        )

    combined_predictions = pd.concat(all_predictions, axis=0)
    aggregate_metrics = _calculate_metrics(combined_predictions, config.quantiles)
    logger.info(
        "Walk-forward validation complete: folds=%s, MAE=%.4f, RMSE=%.4f, MAPE=%.4f, "
        "directional_accuracy=%.4f, interval_95_coverage=%.4f",
        len(folds),
        aggregate_metrics.mae,
        aggregate_metrics.rmse,
        aggregate_metrics.mape,
        aggregate_metrics.directional_accuracy,
        aggregate_metrics.interval_95_coverage,
    )

    return ValidationReport(
        evaluation_method="walk_forward",
        status="COMPLETED",
        target_col=target_col,
        horizon=config.horizon,
        sample_count=sample_count,
        feature_count=feature_count,
        fold_count=len(folds),
        config=config,
        metrics=aggregate_metrics,
        folds=folds,
        notes=["Walk-forward evaluation is the source of truth for model validation."],
    )


def _prepare_supervised_data(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from validation data.")

    ordered = df.copy()
    if isinstance(ordered.index, pd.DatetimeIndex):
        ordered = ordered.sort_index()

    excluded = {"ticker", "date", "target"}
    candidate_features = [col for col in ordered.columns if col not in excluded]
    numeric_features = [
        col for col in candidate_features if pd.api.types.is_numeric_dtype(ordered[col])
    ]

    target_return = (ordered[target_col].shift(-horizon) - ordered[target_col]) / ordered[target_col]
    supervised = ordered[numeric_features].copy()
    supervised["target"] = target_return
    supervised["current_close"] = ordered[target_col]
    supervised["actual_price"] = ordered[target_col].shift(-horizon)
    supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna()

    X = supervised[numeric_features]
    y_return = supervised["target"]
    current_close = supervised["current_close"]
    actual_price = supervised["actual_price"]
    return X, y_return, current_close, actual_price, numeric_features


def _resolve_initial_train_size(config: WalkForwardConfig, sample_count: int) -> int:
    requested = max(1, int(config.initial_train_size))
    minimum_train = max(30, int(sample_count * 0.5))
    maximum_train = max(1, sample_count - max(1, config.validation_window))
    return min(max(requested, minimum_train), maximum_train)


def _predict_quantile_prices(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    current_close: pd.Series,
    actual_price: pd.Series,
    quantiles: Sequence[float],
    model_params: Dict[str, Any],
) -> pd.DataFrame:
    prediction_data: Dict[str, Any] = {
        "current_close": current_close.astype(float).to_numpy(),
        "actual_price": actual_price.astype(float).to_numpy(),
    }
    raw_quantile_prices: List[np.ndarray] = []

    for quantile in quantiles:
        params = model_params.copy()
        params["objective"] = "quantile"
        params["alpha"] = float(quantile)

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        predicted_return = model.predict(X_val)
        predicted_price = current_close.to_numpy(dtype=float) * (1.0 + predicted_return)
        raw_quantile_prices.append(predicted_price.astype(float))

    raw_matrix = np.vstack(raw_quantile_prices).T
    sorted_matrix = np.sort(raw_matrix, axis=1)
    crossed_rows = np.any(np.diff(raw_matrix, axis=1) < 0, axis=1)
    crossing_rate = float(np.mean(crossed_rows))

    for position, quantile in enumerate(quantiles):
        prediction_data[_quantile_column(quantile)] = sorted_matrix[:, position]
    prediction_data["quantile_crossed"] = crossed_rows
    prediction_data["quantile_crossing_rate"] = crossing_rate

    return pd.DataFrame(prediction_data, index=X_val.index)


def _calculate_metrics(predictions: pd.DataFrame, quantiles: Sequence[float]) -> ValidationMetrics:
    actual = predictions["actual_price"].to_numpy(dtype=float)
    current = predictions["current_close"].to_numpy(dtype=float)
    median = predictions[_quantile_column(0.5)].to_numpy(dtype=float)

    error = median - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mape = float(np.mean(_safe_divide(np.abs(error), np.abs(actual))))
    smape = float(np.mean(_safe_divide(2.0 * np.abs(error), np.abs(actual) + np.abs(median))))
    directional_accuracy = float(np.mean(np.sign(actual - current) == np.sign(median - current)))

    lower_80 = predictions[_quantile_column(0.1)].to_numpy(dtype=float)
    upper_80 = predictions[_quantile_column(0.9)].to_numpy(dtype=float)
    lower_95 = predictions[_quantile_column(0.025)].to_numpy(dtype=float)
    upper_95 = predictions[_quantile_column(0.975)].to_numpy(dtype=float)
    interval_80_coverage = float(np.mean((actual >= lower_80) & (actual <= upper_80)))
    interval_95_coverage = float(np.mean((actual >= lower_95) & (actual <= upper_95)))

    pinball_values = []
    for quantile in quantiles:
        q_pred = predictions[_quantile_column(quantile)].to_numpy(dtype=float)
        pinball_values.append(_pinball_loss(actual, q_pred, float(quantile)))

    prediction_bias = float(np.mean(error))
    prediction_bias_pct = float(np.mean(_safe_divide(error, actual)))
    quantile_crossing_rate = float(np.mean(predictions["quantile_crossed"].astype(float)))

    return ValidationMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        smape=smape,
        directional_accuracy=directional_accuracy,
        interval_80_coverage=interval_80_coverage,
        interval_95_coverage=interval_95_coverage,
        interval_coverage=interval_95_coverage,
        pinball_loss=float(np.mean(pinball_values)),
        prediction_bias=prediction_bias,
        prediction_bias_pct=prediction_bias_pct,
        quantile_crossing_rate=quantile_crossing_rate,
    )


def _pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    residual = actual - predicted
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
    result = numerator / denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _quantile_column(quantile: float) -> str:
    return f"q_{quantile:g}"


def _format_index_value(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
```

src/monitoring/__init__.py

```python

```

src/monitoring/drift_detector.py

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional
from utils.helpers import safe_float

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("DriftDetector")

DRIFT_LEVELS = ("NONE", "LOW", "MEDIUM", "HIGH")
LEVEL_ORDER = {level: idx for idx, level in enumerate(DRIFT_LEVELS)}
EVIDENCE_TO_LEVEL = {
    "NONE": "NONE",
    "WEAK": "LOW",
    "MODERATE": "MEDIUM",
    "STRONG": "HIGH",
}


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    validation_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(reference_df, pd.DataFrame) or not isinstance(current_df, pd.DataFrame):
        raise TypeError("Drift detection requires pandas DataFrame inputs.")
    if reference_df.empty or current_df.empty:
        raise ValueError("Drift detection requires non-empty reference and current datasets.")

    drift_notes: List[str] = []
    feature_drift = _feature_drift(reference_df, current_df, drift_notes)
    target_drift = _target_drift(reference_df, current_df, drift_notes)
    concept_drift = _concept_drift(validation_metrics, drift_notes)

    feature_level = feature_drift["level"]
    target_level = target_drift["level"]
    concept_level = concept_drift["level"]
    overall_level = _max_level([feature_level, target_level, concept_level])
    final_drift_label = f"FEATURE_{feature_level}__TARGET_{target_level}__CONCEPT_{concept_level}"

    evidence_summary = _evidence_summary(feature_drift, target_drift, concept_drift)
    if not drift_notes:
        drift_notes.extend(evidence_summary or ["No material drift evidence detected."])

    report = {
        "feature_drift_level": feature_level,
        "target_drift_level": target_level,
        "concept_drift_level": concept_level,
        "overall_drift_level": overall_level,
        "final_drift_label": final_drift_label,
        "feature_drift_detected": feature_level != "NONE",
        "target_drift_detected": target_level != "NONE",
        "concept_drift_detected": concept_level != "NONE",
        "feature_drift": feature_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "drifted_features": feature_drift.get("drifted_features", []),
        "evidence_summary": evidence_summary,
        "drift_notes": drift_notes,
    }
    logger.info(
        "Drift report generated | label=%s | feature=%s | target=%s | concept=%s",
        final_drift_label,
        feature_level,
        target_level,
        concept_level,
    )
    return report


def _feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_notes: List[str],
) -> Dict[str, Any]:
    numeric_cols = [
        col
        for col in reference_df.columns
        if col in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[col])
    ]
    if not numeric_cols:
        drift_notes.append("Feature drift not evaluated because no shared numeric columns were available.")
        return {
            "level": "NONE",
            "detected": False,
            "features": [],
            "drifted_features": [],
            "evaluated_feature_count": 0,
        }

    features: List[Dict[str, Any]] = []
    skipped_features = 0
    for col in _priority_features(numeric_cols):
        ref = reference_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        cur = current_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref) < 20 or len(cur) < 5:
            skipped_features += 1
            continue

        ref_std = float(ref.std() or 0.0)
        mean_shift_z = abs(float(cur.mean() - ref.mean())) / ref_std if ref_std > 0 else 0.0
        std_ratio = float(cur.std() / ref_std) if ref_std > 0 else 1.0
        psi = _psi(ref, cur)

        mean_evidence = _mean_shift_evidence(mean_shift_z)
        std_evidence = _std_ratio_evidence(std_ratio)
        psi_evidence = _psi_evidence(psi)
        feature_level = _max_level(
            [
                _evidence_to_level(mean_evidence),
                _evidence_to_level(std_evidence),
                _evidence_to_level(psi_evidence),
            ]
        )

        features.append(
            {
                "feature": col,
                "mean_shift_z": round(mean_shift_z, 6),
                "mean_shift_evidence": mean_evidence,
                "std_ratio": round(std_ratio, 6),
                "std_ratio_evidence": std_evidence,
                "psi": round(psi, 6),
                "psi_evidence": psi_evidence,
                "feature_drift_level": feature_level,
            }
        )

    if not features:
        drift_notes.append("Feature drift not evaluated because shared numeric columns had insufficient samples.")

    drifted_features = [item for item in features if item["feature_drift_level"] != "NONE"]
    level = _max_level([item["feature_drift_level"] for item in features])
    return {
        "level": level,
        "detected": level != "NONE",
        "features": features,
        "drifted_features": drifted_features,
        "evaluated_feature_count": len(features),
        "skipped_feature_count": skipped_features,
    }


def _target_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_notes: List[str],
) -> Dict[str, Any]:
    if "close" not in reference_df.columns or "close" not in current_df.columns:
        drift_notes.append("Target drift not evaluated because close price is unavailable.")
        return {"level": "NONE", "detected": False, "metrics": {}}

    ref_returns = reference_df["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    cur_returns = current_df["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref_returns) < 20 or len(cur_returns) < 5:
        drift_notes.append("Target drift not evaluated because return samples are insufficient.")
        return {"level": "NONE", "detected": False, "metrics": {}}

    ref_std = float(ref_returns.std() or 0.0)
    mean_shift_z = abs(float(cur_returns.mean() - ref_returns.mean())) / ref_std if ref_std > 0 else 0.0
    volatility_ratio = float(cur_returns.std() / ref_std) if ref_std > 0 else 1.0
    mean_evidence = _target_mean_evidence(mean_shift_z)
    vol_evidence = _target_vol_evidence(volatility_ratio)
    level = _max_level([_evidence_to_level(mean_evidence), _evidence_to_level(vol_evidence)])
    return {
        "level": level,
        "detected": level != "NONE",
        "metrics": {
            "return_mean_shift_z": round(mean_shift_z, 6),
            "return_mean_shift_evidence": mean_evidence,
            "return_volatility_ratio": round(volatility_ratio, 6),
            "return_volatility_evidence": vol_evidence,
        },
    }


def _concept_drift(
    validation_metrics: Optional[Dict[str, Any]],
    drift_notes: List[str],
) -> Dict[str, Any]:
    metrics = validation_metrics.get("metrics") if isinstance(validation_metrics, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        drift_notes.append("Concept drift not evaluated because validation metrics are unavailable.")
        return {"level": "NONE", "detected": False, "metrics": {}}

    mape = safe_float(metrics.get("mape"))
    directional_accuracy = safe_float(metrics.get("directional_accuracy"))
    interval_95_coverage = safe_float(metrics.get("interval_95_coverage", metrics.get("interval_coverage")))

    metric_payload: Dict[str, Any] = {}
    levels: List[str] = []
    if mape is not None:
        mape_evidence = _concept_mape_evidence(mape)
        metric_payload.update({"mape": round(mape, 6), "mape_evidence": mape_evidence})
        levels.append(_evidence_to_level(mape_evidence))
    if directional_accuracy is not None:
        da_evidence = _directional_accuracy_evidence(directional_accuracy)
        metric_payload.update(
            {
                "directional_accuracy": round(directional_accuracy, 6),
                "directional_accuracy_evidence": da_evidence,
            }
        )
        levels.append(_evidence_to_level(da_evidence))
    if interval_95_coverage is not None:
        coverage_evidence = _interval_coverage_evidence(interval_95_coverage)
        metric_payload.update(
            {
                "interval_95_coverage": round(interval_95_coverage, 6),
                "interval_coverage_evidence": coverage_evidence,
            }
        )
        levels.append(_evidence_to_level(coverage_evidence))

    if not levels:
        drift_notes.append("Concept drift not evaluated because validation metrics were present but unusable.")

    level = _max_level(levels)
    return {"level": level, "detected": level != "NONE", "metrics": metric_payload}


def _priority_features(numeric_cols: List[str]) -> List[str]:
    preferred = [
        "close",
        "volume",
        "daily_return",
        "vol_change",
        "ma_7",
        "ma_14",
        "volatility_7",
        "rsi_14",
        "macd",
        "atr_14",
        "roc_7",
        "vn30_return",
        "vn30f_return",
    ]
    ordered = [col for col in preferred if col in numeric_cols]
    ordered.extend([col for col in numeric_cols if col not in ordered][:8])
    return ordered[:16]


def _mean_shift_evidence(value: float) -> str:
    if value < 1.0:
        return "NONE"
    if value < 2.0:
        return "WEAK"
    if value < 3.0:
        return "MODERATE"
    return "STRONG"


def _std_ratio_evidence(value: float) -> str:
    if 0.8 <= value <= 1.2:
        return "NONE"
    if 0.6 <= value < 0.8 or 1.2 < value <= 1.5:
        return "WEAK"
    if 0.5 <= value < 0.6 or 1.5 < value < 2.0:
        return "MODERATE"
    return "STRONG"


def _psi_evidence(value: float) -> str:
    if value < 0.10:
        return "NONE"
    if value < 0.20:
        return "WEAK"
    if value < 0.35:
        return "MODERATE"
    return "STRONG"


def _target_mean_evidence(value: float) -> str:
    if value < 0.75:
        return "NONE"
    if value < 1.5:
        return "WEAK"
    if value < 2.5:
        return "MODERATE"
    return "STRONG"


def _target_vol_evidence(value: float) -> str:
    if 0.8 <= value <= 1.2:
        return "NONE"
    if 0.6 <= value < 0.8 or 1.2 < value <= 1.5:
        return "WEAK"
    if 0.5 <= value < 0.6 or 1.5 < value < 1.8:
        return "MODERATE"
    return "STRONG"


def _concept_mape_evidence(value: float) -> str:
    if value < 0.03:
        return "NONE"
    if value < 0.05:
        return "WEAK"
    if value < 0.08:
        return "MODERATE"
    return "STRONG"


def _directional_accuracy_evidence(value: float) -> str:
    if value >= 0.60:
        return "NONE"
    if value >= 0.55:
        return "WEAK"
    if value >= 0.48:
        return "MODERATE"
    return "STRONG"


def _interval_coverage_evidence(value: float) -> str:
    if value >= 0.80:
        return "NONE"
    if value >= 0.70:
        return "WEAK"
    if value >= 0.55:
        return "MODERATE"
    return "STRONG"


def _evidence_summary(
    feature_drift: Dict[str, Any],
    target_drift: Dict[str, Any],
    concept_drift: Dict[str, Any],
) -> List[str]:
    summary: List[str] = []
    if feature_drift["detected"]:
        summary.append(
            f"Feature drift level={feature_drift['level']} across "
            f"{len(feature_drift.get('drifted_features', []))} drifted features."
        )
    if target_drift["detected"]:
        summary.append(f"Target drift level={target_drift['level']}.")
    if concept_drift["detected"]:
        summary.append(f"Concept drift level={concept_drift['level']}.")
    return summary


def _psi(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    edges = np.unique(np.quantile(reference.to_numpy(dtype=float), np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    eps = 1e-6
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), eps)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _evidence_to_level(evidence: str) -> str:
    return EVIDENCE_TO_LEVEL.get(str(evidence).upper(), "NONE")


def _max_level(levels: List[str]) -> str:
    if not levels:
        return "NONE"
    return max((str(level).upper() for level in levels), key=lambda level: LEVEL_ORDER.get(level, 0))
```

src/monitoring/regime_detector.py

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("RegimeDetector")


def detect_regime(processed_df: pd.DataFrame) -> Dict[str, Any]:
    _validate_regime_input(processed_df)

    df = processed_df.sort_index().copy()
    notes: List[str] = []
    warnings: List[str] = []

    volatility_regime, vol_metrics = _volatility_regime(df)
    trend_regime, trend_metrics, trend_warnings = _trend_regime(df)
    volume_regime, volume_metrics = _volume_regime(df)
    warnings.extend(trend_warnings)

    final_regime_label = f"{volatility_regime}__{trend_regime}__{volume_regime}"
    metrics = {
        **vol_metrics,
        **trend_metrics,
        **volume_metrics,
    }
    notes.append(
        "Regime components computed independently: volatility=%s, trend=%s, volume=%s."
        % (volatility_regime, trend_regime, volume_regime)
    )

    report = {
        "volatility_regime": volatility_regime,
        "trend_regime": trend_regime,
        "volume_regime": volume_regime,
        "final_regime_label": final_regime_label,
        "metrics": metrics,
        "warnings": warnings,
        "regime_notes": notes,
        "liquidity_regime": volume_regime,
    }
    logger.info(
        "Regime report generated | volatility=%s | trend=%s | volume=%s | label=%s",
        volatility_regime,
        trend_regime,
        volume_regime,
        final_regime_label,
    )
    return report


def _validate_regime_input(processed_df: pd.DataFrame) -> None:
    if not isinstance(processed_df, pd.DataFrame):
        raise TypeError("Regime detection requires a pandas DataFrame input.")
    if processed_df.empty:
        raise ValueError("Regime detection requires a non-empty processed dataset.")
    if "close" not in processed_df.columns:
        raise ValueError("Regime detection requires a close column.")
    close = processed_df["close"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(close) < 22:
        raise ValueError("Regime detection requires at least 22 valid close observations.")


def _volatility_regime(df: pd.DataFrame) -> tuple[str, Dict[str, Optional[float]]]:
    returns = df["daily_return"] if "daily_return" in df.columns else df["close"].pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 20:
        raise ValueError("Volatility regime requires at least 20 return observations.")

    current_vol = float(returns.tail(20).std() or 0.0)
    rolling_vol = returns.rolling(20).std().dropna()
    vol_percentile = _percentile_rank(rolling_vol, current_vol)

    if vol_percentile < 0.25:
        label = "LOW_VOLATILITY"
    elif vol_percentile < 0.75:
        label = "NORMAL_VOLATILITY"
    elif vol_percentile < 0.92:
        label = "HIGH_VOLATILITY"
    else:
        label = "EXTREME_VOLATILITY"

    return label, {
        "vol_percentile": round(vol_percentile, 6),
        "current_vol_20d": round(current_vol, 6),
    }


def _trend_regime(df: pd.DataFrame) -> tuple[str, Dict[str, float], List[str]]:
    ma7 = df["close"].rolling(7).mean()
    ma21 = df["close"].rolling(21).mean()
    latest_ma21 = float(ma21.iloc[-1]) if not pd.isna(ma21.iloc[-1]) else 0.0
    if latest_ma21 <= 0:
        raise ValueError("Trend regime requires a positive 21-day moving average.")

    ma_gap = float((ma7.iloc[-1] - ma21.iloc[-1]) / latest_ma21)
    return_20d = _safe_float(df["close"].pct_change(20).iloc[-1])
    return_5d = _safe_float(df["close"].pct_change(5).iloc[-1])

    if ma_gap > 0.015 and return_20d > 0.03:
        label = "UPTREND"
    elif ma_gap < -0.015 and return_20d < -0.03:
        label = "DOWNTREND"
    elif abs(ma_gap) <= 0.015 and abs(return_20d) <= 0.04:
        label = "SIDEWAYS"
    else:
        label = "MIXED_TREND"

    warnings: List[str] = []
    if label == "UPTREND" and return_5d < -0.02:
        warnings.append("SHORT_TERM_PULLBACK")
    if label == "DOWNTREND" and return_5d > 0.02:
        warnings.append("SHORT_TERM_REBOUND")

    return label, {
        "ma_gap": round(ma_gap, 6),
        "return_20d": round(return_20d, 6),
        "return_5d": round(return_5d, 6),
    }, warnings


def _volume_regime(df: pd.DataFrame) -> tuple[str, Dict[str, Optional[float]]]:
    if "volume" not in df.columns:
        return "NORMAL_VOLUME", {"volume_zscore": None, "recent_volume_ratio": None}

    volume = df["volume"].replace([np.inf, -np.inf], np.nan).dropna()
    if volume.empty:
        return "NORMAL_VOLUME", {"volume_zscore": None, "recent_volume_ratio": None}

    trailing = volume.tail(60)
    volume_mean = float(trailing.mean() or 0.0)
    volume_std = float(trailing.std() or 0.0)
    volume_zscore = (float(volume.iloc[-1]) - volume_mean) / volume_std if volume_std > 0 else 0.0
    recent_volume_ratio = float(volume.tail(5).mean() / volume_mean) if volume_mean > 0 else 1.0

    if volume_zscore >= 2.5 or recent_volume_ratio >= 1.8:
        label = "VOLUME_SPIKE"
    elif recent_volume_ratio <= 0.45:
        label = "LOW_VOLUME"
    else:
        label = "NORMAL_VOLUME"

    return label, {
        "volume_zscore": round(volume_zscore, 6),
        "recent_volume_ratio": round(recent_volume_ratio, 6),
    }


def _percentile_rank(values: pd.Series, current_value: float) -> float:
    if values.empty:
        raise ValueError("Volatility percentile requires rolling volatility observations.")
    return float((values <= current_value).mean())


def _safe_float(value: Any) -> float:
    if pd.isna(value):
        raise ValueError("Regime metric calculation produced NaN.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Regime metric calculation produced non-numeric value: {value}") from exc
```

src/monitoring/risk_engine.py

```python
from __future__ import annotations

from typing import Any, Dict, List
from utils.logger import get_logger

logger = get_logger("RiskEngine")

def calculate_risk_report(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """Measure forecast risk strictly from validated quantile forecasts."""
    forecasts = _validated_forecasts(forecast_data)
    current_price = _required_positive_float(forecast_data.get("current_price"), "current_price")

    horizon_forecast = forecasts[-1]
    expected_price = _required_positive_float(horizon_forecast.get("q_0.5"), "q_0.5")
    lower_95 = _required_positive_float(horizon_forecast.get("q_0.025"), "q_0.025")
    upper_95 = _required_positive_float(horizon_forecast.get("q_0.975"), "q_0.975")
    
    if lower_95 > expected_price or expected_price > upper_95:
        raise ValueError("Risk calculation requires ordered 95% forecast quantiles.")

    expected_return = (expected_price - current_price) / current_price
    downside_risk_95 = min(0.0, (lower_95 - current_price) / current_price)
    upside_potential_95 = max(0.0, (upper_95 - current_price) / current_price)
    var_95 = max(0.0, (current_price - lower_95) / current_price)
    expected_shortfall = var_95 * 1.15
    risk_reward_ratio = upside_potential_95 / abs(downside_risk_95) if downside_risk_95 < 0 else 0.0

    risk_notes: List[str] = []
    risk_level = _risk_level(var_95=var_95, expected_shortfall=expected_shortfall, risk_notes=risk_notes)

    report = {
        "expected_return": round(expected_return, 6),
        "expected_return_7d": round(expected_return, 6),
        "downside_risk_95": round(downside_risk_95, 6),
        "upside_potential_95": round(upside_potential_95, 6),
        "risk_reward_ratio": round(risk_reward_ratio, 4),
        "var_95": round(var_95, 6),
        "expected_shortfall": round(expected_shortfall, 6),
        "risk_level": risk_level,
        "risk_inputs": {
            "current_price": current_price,
            "expected_price": expected_price,
            "lower_95": lower_95,
            "upper_95": upper_95,
        },
        "risk_notes": risk_notes,
    }
    logger.info(
        "Risk report generated | expected_return=%.4f | risk_level=%s | var_95=%.4f",
        expected_return,
        risk_level,
        var_95,
    )
    return report

def _validated_forecasts(forecast_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(forecast_data, dict):
        raise TypeError("Risk calculation requires forecast_data as a dict.")
    forecasts = forecast_data.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        raise ValueError("Risk calculation requires non-empty forecast quantiles.")
    if not all(isinstance(item, dict) for item in forecasts):
        raise ValueError("Risk calculation requires forecast rows as dictionaries.")
    return forecasts

def _risk_level(var_95: float, expected_shortfall: float, risk_notes: List[str]) -> str:
    if var_95 >= 0.20 or expected_shortfall >= 0.25:
        risk_notes.append("Tail loss estimate is extreme.")
        return "EXTREME_RISK"
    if var_95 >= 0.12 or expected_shortfall >= 0.15:
        risk_notes.append("Tail loss estimate is elevated.")
        return "HIGH_RISK"
    if var_95 >= 0.07:
        risk_notes.append("Tail loss estimate is moderate.")
        return "MEDIUM_RISK"
    risk_notes.append("Tail risk is within normal monitoring range.")
    return "LOW_RISK"

def _required_positive_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Risk calculation requires {field_name}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Risk calculation requires numeric {field_name}.") from exc
    if numeric <= 0:
        raise ValueError(f"Risk calculation requires positive {field_name}.")
    return numeric
```

src/orchestration/__init__.py

```python
"""Pipeline orchestration package."""
```

src/orchestration/daily_pipeline.py

```python
from __future__ import annotations

import time
import uuid
import warnings
from datetime import datetime, timedelta

from dotenv import load_dotenv

from src.agent.graph import build_agent_graph
from src.ingestion.vnstock_api import get_vingroup_and_context_data
from src.modeling.predictor import generate_7_day_forecast
from src.processing.cleaner import process_and_save_data
from src.processing.db_manager import load_from_sqlite
from src.reporting.generator import generate_reports

# --- IMPORT CÁC HÀM MONITORING & MODELING ---
from src.monitoring.regime_detector import detect_regime
from src.monitoring.drift_detector import detect_drift
from src.monitoring.risk_engine import calculate_risk_report
from src.modeling.validation import load_model_params

from utils.logger import get_logger

warnings.filterwarnings("ignore")
load_dotenv()
logger = get_logger("MainPipeline")


def run_daily_pipeline(target_ticker: str = "VIC"):
    run_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    logger.info("Pipeline started | run_id=%s | ticker=%s", run_id, target_ticker)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")

    logger.info(
        "Phase started | run_id=%s | ticker=%s | phase=ingestion | start_date=%s | end_date=%s",
        run_id,
        target_ticker,
        start_date,
        end_date,
    )
    # Lấy dữ liệu raw (Không áp dụng bất kỳ heuristic scale giá nào)
    raw_data = get_vingroup_and_context_data(start_date, end_date)
    if target_ticker not in raw_data:
        logger.error("Pipeline aborted | run_id=%s | ticker=%s | reason=missing_target_data", run_id, target_ticker)
        return None

    logger.info("Phase started | run_id=%s | ticker=%s | phase=feature_engineering", run_id, target_ticker)
    process_and_save_data(raw_data)

    logger.info("Phase started | run_id=%s | ticker=%s | phase=modeling", run_id, target_ticker)
    df_processed = load_from_sqlite(f"processed_{target_ticker}")
    forecast_result = generate_7_day_forecast(df_processed)
    forecast_result["ticker"] = target_ticker

    # Tính các monitoring trước khi nhét vào agent 
    logger.info("Phase started | run_id=%s | ticker=%s | phase=monitoring", run_id, target_ticker)
    regime = detect_regime(df_processed)
    risk = calculate_risk_report(forecast_result)
    
    # Chia dữ liệu ra làm 2 (ví dụ: 60 ngày cuối làm current, phần còn lại làm reference) để tính drift
    current_df = df_processed.iloc[-60:] if len(df_processed) > 60 else df_processed
    reference_df = df_processed.iloc[:-60] if len(df_processed) > 120 else df_processed.iloc[:len(df_processed)//2]
    drift = detect_drift(reference_df, current_df, forecast_result.get("validation_metrics"))

    # Khởi tạo state tĩnh gọn gàng theo chuẩn Agent mới
    initial_state = {
        "ticker": target_ticker,
        "forecast_data": forecast_result,
        "validation_metrics": forecast_result.get("validation_metrics", {}),
        "monitoring": {
            "regime": regime,
            "risk": risk,
            "drift": drift
        },
        "current_config": load_model_params(), 
        "retry_count": 0,
        "news_context": ""
    }

    logger.info("Phase started | run_id=%s | ticker=%s | phase=agent_workflow", run_id, target_ticker)
    agent_app = build_agent_graph()
    final_state = agent_app.invoke(initial_state)

    logger.info("Phase started | run_id=%s | ticker=%s | phase=reporting", run_id, target_ticker)
    
    # Bỏ try...except để báo lỗi một cách tường minh
    report_paths = generate_reports(final_state)

    duration = time.perf_counter() - started
    
    # Log theo chuẩn state mới
    eval_status = final_state.get("evaluation", {}).get("status", "UNKNOWN")
    retries = final_state.get("retry_count", 0)
    final_action = final_state.get("final_report", {}).get("action", "UNKNOWN")

    logger.info(
        "Pipeline completed | run_id=%s | ticker=%s | duration_sec=%.2f | eval_status=%s | "
        "retrain_count=%d | final_action=%s",
        run_id,
        target_ticker,
        duration,
        eval_status,
        retries,
        final_action,
    )
    logger.info("Reports saved | run_id=%s | paths=%s", run_id, report_paths)
        
    return final_state
```

src/processing/__init__.py

```python

```

src/processing/cleaner.py

```python
import pandas as pd
from src.processing.features import generate_technical_features, generate_context_features
from src.processing.db_manager import save_to_sqlite
from utils.logger import get_logger

logger = get_logger("DataProcessing")

def process_and_save_data(raw_data_dict: dict):
    """
    Quy trình làm sạch, thêm tính năng (Feature Engineering) và lưu trữ.
    """
    logger.info("Feature engineering phase started")
    
    # 1. Xử lý Context Data (VN30 và VN30F)
    vn30_features = None
    vn30f_features = None
    
    if "VN30" in raw_data_dict:
        vn30_features = generate_context_features(raw_data_dict["VN30"], prefix="vn30")
        # Lưu raw context vào DB
        save_to_sqlite(raw_data_dict["VN30"], "raw_VN30")
        
    if "VN30F1M" in raw_data_dict:
        vn30f_features = generate_context_features(raw_data_dict["VN30F1M"], prefix="vn30f")
        save_to_sqlite(raw_data_dict["VN30F1M"], "raw_VN30F1M")

    # Merge hai bảng context lại với nhau theo index (date)
    context_combined = pd.DataFrame()
    if vn30_features is not None and vn30f_features is not None:
        context_combined = vn30_features.join(vn30f_features, how='outer')
    
    processed_stocks = {}
    stock_symbols = [sym for sym in raw_data_dict.keys() if sym not in ["VN30", "VN30F1M"]]
    
    # 2. Xử lý từng mã cổ phiếu
    for sym in stock_symbols:
        df = raw_data_dict[sym].copy()
        
        # Lưu raw data vào DB trước khi biến đổi
        save_to_sqlite(df, f"raw_{sym}")
        
        # Tạo technical features
        df_featured = generate_technical_features(df)
        
        # Nhúng Context Features (Thị trường chung) vào cổ phiếu
        if not context_combined.empty:
            df_featured = df_featured.join(context_combined, how='left')
            
        # Xóa các dòng bị NaN do quá trình tính lag/rolling/merge
        # Đối với cổ phiếu mới như VPL, việc rớt vài dòng đầu là bình thường
        initial_len = len(df_featured)
        df_featured.dropna(inplace=True)
        final_len = len(df_featured)
        
        logger.info(
            "Feature engineering completed | symbol=%s | rows_dropped=%s | rows_final=%s",
            sym,
            initial_len - final_len,
            final_len,
        )
        
        # Lưu dữ liệu ĐÃ XỬ LÝ (Processed) vào Database
        save_to_sqlite(df_featured, f"processed_{sym}")
        processed_stocks[sym] = df_featured
        
    logger.info("Feature engineering phase completed | database=data/database.sqlite")
    return processed_stocks
```

src/processing/db_manager.py

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from utils.logger import get_logger

logger = get_logger("DBManager")

DB_PATH = Path("data/database.sqlite")


def get_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")


def save_to_sqlite(df: pd.DataFrame, table_name: str, replace: bool = True):
    try:
        engine = get_engine()
        if_exists_behavior = "replace" if replace else "append"
        df.to_sql(table_name, con=engine, if_exists=if_exists_behavior, index=True)
        logger.info("SQLite write completed | table=%s | rows=%s | mode=%s", table_name, len(df), if_exists_behavior)
    except Exception as e:
        logger.error("SQLite write failed | table=%s | error=%s", table_name, str(e))


def load_from_sqlite(table_name: str) -> pd.DataFrame:
    try:
        engine = get_engine()
        df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine, index_col="date")
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logger.error("SQLite read failed | table=%s | error=%s", table_name, str(e))
        return pd.DataFrame()
```

src/processing/features.py

```python
import pandas as pd
import numpy as np
import ta  # Technical Analysis

def generate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng kỹ thuật & Calendar cho cổ phiếu."""
    df = df.copy().sort_index()
    
    # 1. Cơ bản: Returns & Volume
    df['daily_return'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    
    # 2. Moving Averages & Volatility
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_14'] = df['close'].rolling(window=14).mean()
    df['volatility_7'] = df['daily_return'].rolling(window=7).std()
    
    # 3. Chỉ báo kỹ thuật (RSI, MACD, ATR, ROC)
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['roc_7'] = ta.momentum.ROCIndicator(df['close'], window=7).roc()
    
    # 4. Calendar Features (Thời gian)
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # 5. Độ trễ (Lags) của cả Giá và Return
    for lag in [1, 3, 7, 14]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        
    return df

def generate_context_features(context_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Tạo các đặc trưng từ thị trường chung (VN30, VN30F)."""
    df = context_df.copy()
    context_features = pd.DataFrame(index=df.index)
    context_features[f'{prefix}_return'] = df['close'].pct_change()
    context_features[f'{prefix}_close'] = df['close']
    return context_features
```

src/reporting/__init__.py

```python

```

src/reporting/generator.py

```python
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go

from src.agent.state import AgentState
from utils.logger import get_logger

logger = get_logger("ReportGenerator")

def ensure_directories() -> Dict[str, Path]:
    base = Path("reports")
    folders = {
        "json": base / "json",
        "markdown": base / "markdown",
        "html": base / "html",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders

def generate_reports(state: AgentState) -> Dict[str, str]:
    folders = ensure_directories()
    ticker = state.get("ticker", "UNKNOWN")
    today_str = datetime.now().strftime("%Y-%m-%d")

    executive_json_path = folders["json"] / f"{ticker}_executive_report_{today_str}.json"
    technical_json_path = folders["json"] / f"{ticker}_technical_pipeline_report_{today_str}.json"
    md_path = folders["markdown"] / f"{ticker}_report_{today_str}.md"
    html_path = folders["html"] / f"{ticker}_report_{today_str}.html"

    _write_json(executive_json_path, _build_json_payload(state, "executive"))
    _write_json(technical_json_path, _build_json_payload(state, "technical"))
    md_path.write_text(_build_markdown_report(state, today_str), encoding="utf-8")
    html_path.write_text(_build_html_report(state, today_str), encoding="utf-8")

    return {
        "executive_json": str(executive_json_path),
        "technical_pipeline_json": str(technical_json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }

def _build_json_payload(state: AgentState, report_type: str) -> Dict[str, Any]:
    return {
        "metadata": {
            "ticker": state.get("ticker", "UNKNOWN"),
            "report_type": report_type,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "retrain_count": state.get("retry_count", 0),
        },
        "forecast_data": state.get("forecast_data", {}),
        "validation_metrics": state.get("validation_metrics", {}),
        "monitoring": state.get("monitoring", {}),
        "current_config": state.get("current_config", {}),
        "evaluation": state.get("evaluation", {}),
        "news_context": state.get("news_context", ""),
        "final_report": state.get("final_report", {}),
    }

def _safe_pct(val: Any) -> str:
    try: return f"{float(val)*100:.2f}%"
    except: return "N/A"

def _safe_num(val: Any) -> str:
    try: return f"{float(val):.4f}"
    except: return "N/A"

def _build_markdown_report(state: AgentState, today_str: str) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    final = state.get("final_report", {})
    eval_dict = state.get("evaluation", {})
    mon = state.get("monitoring", {})
    risk = mon.get("risk", {})
    regime = mon.get("regime", {})
    drift = mon.get("drift", {})
    val = state.get("validation_metrics", {}).get("metrics", {})
    fc = state.get("forecast_data", {})
    
    return f"""# Quant Research Report: {ticker} ({today_str})

## 1. Executive Summary
- **Final Action:** {final.get('action', 'N/A')}
- **Model Status:** {eval_dict.get('status', 'N/A')} (Retries: {state.get('retry_count', 0)})
- **Summary:** {final.get('summary', 'N/A')}

### Luận điểm đầu tư (Reasoning):
{final.get('reasoning', 'N/A')}

## 2. Forecast & Risk
- Current Price: {_safe_num(fc.get('current_price'))}
- Expected Return: {_safe_pct(risk.get('expected_return'))}
- Risk Level: **{risk.get('risk_level', 'N/A')}**
- Value at Risk (95%): {_safe_pct(risk.get('var_95'))}
- Downside 95%: {_safe_pct(risk.get('downside_risk_95'))}

## 3. Monitoring Context
- **Regime:** {regime.get('final_regime_label', 'N/A')}
- **Drift:** {drift.get('final_drift_label', 'N/A')}

## 4. Model Validation (Walk-forward)
- MAPE: {_safe_pct(val.get('mape'))}
- Directional Accuracy: {_safe_pct(val.get('directional_accuracy'))}
- Interval Coverage 95%: {_safe_pct(val.get('interval_95_coverage'))}

## 5. Agent Workflow Logs
- LLM Evaluation Reason: {eval_dict.get('reasoning', 'N/A')}
- News Context Found: {"Yes" if state.get('news_context') else "No"}
"""

def _build_html_report(state: AgentState, today_str: str) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    final = state.get("final_report", {})
    action = final.get("action", "MANUAL_REVIEW")
    
    bg_color = "#f3f4f6"
    text_color = "#1f2937"
    if action == "BUY": bg_color, text_color = "#def7ec", "#03543f"
    elif action == "SELL": bg_color, text_color = "#fde8e8", "#9b1c1c"
    elif action == "WATCH": bg_color, text_color = "#fef3c7", "#92400e"

    # Chart
    forecasts = state.get("forecast_data", {}).get("forecasts", [])
    chart_div = "<p>No forecast data available.</p>"
    if forecasts:
        steps = [f"T+{item.get('step')}" for item in forecasts]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps + steps[::-1],
            y=[item.get("q_0.975") for item in forecasts] + [item.get("q_0.025") for item in forecasts][::-1],
            fill="toself", fillcolor="rgba(47, 128, 237, 0.14)", line=dict(color="rgba(255,255,255,0)"), name="95% interval"))
        fig.add_trace(go.Scatter(x=steps, y=[item.get("q_0.5") for item in forecasts], mode="lines+markers", name="Median forecast"))
        fig.update_layout(title=f"7-day quantile forecast for {ticker}", template="plotly_white")
        chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return f"""
    <html>
        <head><style>body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; }} .signal {{ padding: 15px; border-radius: 8px; font-size: 20px; font-weight: bold; background: {bg_color}; color: {text_color}; text-align: center; }} .section {{ margin-top: 20px; padding: 15px; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb; }}</style></head>
        <body>
            <h1>Quant Research Report: {ticker} ({today_str})</h1>
            <div class="signal">Final Action: {action}</div>
            <div class="section">
                <h3>Assessment Summary</h3>
                <p>{html.escape(final.get('summary', ''))}</p>
                <p><b>Reasoning:</b> {html.escape(final.get('reasoning', ''))}</p>
            </div>
            {chart_div}
        </body>
    </html>
    """

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4, default=str)
    logger.info("Report saved | format=json | path=%s", path)
```

tests/test_refactor_contracts.py

```python
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.agent.config_patch_validator import validate_config_patch
from src.agent.governance import compare_model_candidates
from src.monitoring.drift_detector import detect_drift
from src.monitoring.regime_detector import detect_regime
from src.monitoring.risk_engine import calculate_risk_report


def _market_frame(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(100, 120, rows)
    df = pd.DataFrame({"close": close, "volume": np.full(rows, 1_000_000.0)}, index=idx)
    df["daily_return"] = df["close"].pct_change().fillna(0.0)
    return df


class RefactorContractsTest(unittest.TestCase):
    def test_drift_report_uses_labels(self):
        reference = _market_frame().tail(60)
        validation = {"metrics": {"mape": 0.09, "directional_accuracy": 0.42, "interval_95_coverage": 0.50}}

        report = detect_drift(reference, reference.copy(), validation)

        self.assertEqual(report["concept_drift_level"], "HIGH")
        self.assertIn("final_drift_label", report)
        self.assertNotIn("recommended_action", report)

    def test_regime_report_has_no_confidence_heuristic(self):
        report = detect_regime(_market_frame())

        self.assertIn("final_regime_label", report)
        self.assertNotIn("regime_confidence", report)

    def test_risk_engine_raises_on_invalid_forecast(self):
        with self.assertRaises(ValueError):
            calculate_risk_report({})

    def test_config_validator_is_range_only(self):
        patch = {"learning_rate": 0.03, "max_depth": 6, "num_leaves": 64, "min_child_samples": 25}
        valid_patch, warnings, valid = validate_config_patch(patch, {}, {})

        self.assertTrue(valid)
        self.assertEqual(warnings, [])
        self.assertEqual(valid_patch["num_leaves"], 64)

    def test_metric_comparison_prefers_lower_mape(self):
        current = {"mape": 0.03, "directional_accuracy": 0.55, "interval_95_coverage": 0.82}
        candidate = {"mape": 0.025, "directional_accuracy": 0.52, "interval_95_coverage": 0.80}

        result = compare_model_candidates(current, candidate)

        self.assertEqual(result["decision"], "SAVE_CANDIDATE_CONFIG")
        self.assertTrue(result["accepted_candidate"])


if __name__ == "__main__":
    unittest.main()
```

utils/__init__.py

```python

```

utils/helpers.py

```python
from typing import Any, Optional

def format_vnd(amount: float) -> str:
    """
    Định dạng số thực thành chuỗi tiền tệ VNĐ.
    Ví dụ: 220000.5 -> '220,000 VNĐ'
    """
    try:
        return f"{int(amount):,} VNĐ"
    except (ValueError, TypeError):
        return str(amount)

def get_current_timestamp() -> str:
    """Trả về thời gian hiện tại chuẩn ISO"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Ép kiểu an toàn sang số thực, trả về default nếu lỗi."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
```

utils/logger.py

```python
import logging
import os
from datetime import datetime

def get_logger(name="AgenticStockSystem"):
    """
    Khởi tạo và cấu hình Logger.
    Log sẽ được in ra màn hình (Console) và lưu vào file trong thư mục logs/
    """
    logger = logging.getLogger(name)
    
    # Tránh việc add handler nhiều lần nếu gọi lại hàm
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Format của log: [Thời gian] - [Tên module] - [Mức độ] - [Nội dung]
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-22s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 1. Bắn log ra Console (Terminal)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. Lưu log vào file theo ngày
        # Đảm bảo thư mục logs/ tồn tại
        os.makedirs("logs", exist_ok=True)
        log_filename = datetime.now().strftime("%Y-%m-%d") + "_system.log"
        file_handler = logging.FileHandler(os.path.join("logs", log_filename), encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
```

