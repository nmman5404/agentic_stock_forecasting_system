from __future__ import annotations

import time
import uuid
import warnings
from datetime import datetime, timedelta

from dotenv import load_dotenv

from src.agent.graph import build_agent_graph

from src.ingestion.vnstock_api import get_market_data
from src.modeling.predictor import generate_7_day_forecast

from src.processing.cleaner import process_and_save_data
from src.processing.db_manager import load_from_sqlite
from src.reporting.generator import generate_reports


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
    
    raw_data = get_market_data(target_ticker, start_date, end_date)
    if target_ticker not in raw_data:
        logger.error("Pipeline aborted | run_id=%s | ticker=%s | reason=missing_target_data", run_id, target_ticker)
        return None

    logger.info("Phase started | run_id=%s | ticker=%s | phase=feature_engineering", run_id, target_ticker)
    process_and_save_data(raw_data) # hàm này tạo features luôn

    logger.info("Phase started | run_id=%s | ticker=%s | phase=modeling", run_id, target_ticker)
    df_processed = load_from_sqlite(f"processed_{target_ticker}")
    forecast_result = generate_7_day_forecast(df_processed, target_ticker) # hàm này chạy walk-forward validation, trả về cả metrics và forecast

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