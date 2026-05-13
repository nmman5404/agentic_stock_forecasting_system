from __future__ import annotations

import os
import time
import uuid
import warnings
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

from src.agent.graph import build_agent_graph
from src.ingestion.vnstock_api import get_vingroup_and_context_data
from src.modeling.predictor import generate_7_day_forecast
from src.monitoring.drift_detector import detect_drift
from src.monitoring.regime_detector import detect_regime
from src.processing.cleaner import process_and_save_data
from src.processing.db_manager import load_from_sqlite
from src.reporting.generator import generate_reports
from src.risk.risk_engine import calculate_risk_report
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
    raw_data = get_vingroup_and_context_data(start_date, end_date)
    if target_ticker not in raw_data:
        logger.error("Pipeline aborted | run_id=%s | ticker=%s | reason=missing_target_data", run_id, target_ticker)
        return None

    logger.info("Phase started | run_id=%s | ticker=%s | phase=feature_engineering", run_id, target_ticker)
    process_and_save_data(raw_data)

    logger.info("Phase started | run_id=%s | ticker=%s | phase=modeling", run_id, target_ticker)
    df_processed = load_from_sqlite(f"processed_{target_ticker}")
    forecast_result = generate_7_day_forecast(df_processed)

    logger.info("Phase started | run_id=%s | ticker=%s | phase=monitoring", run_id, target_ticker)
    regime_report = detect_regime(df_processed)
    reference_df, current_df = _split_reference_current(df_processed)
    drift_report = detect_drift(reference_df, current_df, forecast_result.get("validation_metrics"))
    risk_report = calculate_risk_report(
        forecast_result,
        forecast_result.get("validation_metrics"),
        regime_report=regime_report,
        drift_report=drift_report,
    )

    initial_state = {
        "run_id": run_id,
        "ticker": target_ticker,
        "forecast_data": forecast_result,
        "validation_metrics": forecast_result.get("validation_metrics", {}),
        "regime_report": regime_report,
        "drift_report": drift_report,
        "risk_report": risk_report,
        "signal_confidence": risk_report.get("signal_confidence", 0.0),
        "retry_count": 0,
        "audit_trail": [
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "phase": "pipeline_initialization",
                "status": "PASS",
                "message": "Initial forecast, monitoring, and risk reports prepared.",
                "details": {"run_id": run_id},
            }
        ],
    }

    logger.info("Phase started | run_id=%s | ticker=%s | phase=agent_workflow", run_id, target_ticker)
    agent_app = build_agent_graph()
    final_state = agent_app.invoke(initial_state)

    logger.info("Phase started | run_id=%s | ticker=%s | phase=reporting", run_id, target_ticker)
    report_paths = generate_reports(final_state)

    duration = time.perf_counter() - started
    logger.info(
        "Pipeline completed successfully | run_id=%s | ticker=%s | duration_sec=%.2f | model_status=%s | "
        "retrain_attempts=%s | final_signal=%s | confidence=%.2f",
        run_id,
        target_ticker,
        duration,
        final_state.get("evaluation_status", "N/A"),
        final_state.get("retry_count", 0),
        final_state.get("trading_signal", "HOLD"),
        final_state.get("signal_confidence", 0.0),
    )
    logger.info("Reports saved | run_id=%s | paths=%s", run_id, report_paths)
    return final_state


def _split_reference_current(df: pd.DataFrame, current_window: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= current_window * 2:
        split_idx = max(1, int(len(df) * 0.7))
        return df.iloc[:split_idx], df.iloc[split_idx:]
    return df.iloc[:-current_window], df.iloc[-current_window:]


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Missing GOOGLE_API_KEY. Set it in .env before running the pipeline.")
        raise SystemExit(1)

    run_daily_pipeline(target_ticker="VIC")
