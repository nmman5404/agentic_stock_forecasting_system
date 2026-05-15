from __future__ import annotations

import time
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.agent.graph import build_agent_graph
from src.ingestion.vnstock_api import get_vingroup_and_context_data
from src.modeling.predictor import generate_7_day_forecast
from src.processing.cleaner import process_and_save_data
from src.processing.db_manager import load_from_sqlite
from src.reporting.generator import generate_reports
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
    price_scale_metadata = _check_and_apply_price_scale(raw_data)

    logger.info("Phase started | run_id=%s | ticker=%s | phase=feature_engineering", run_id, target_ticker)
    process_and_save_data(raw_data)

    logger.info("Phase started | run_id=%s | ticker=%s | phase=modeling", run_id, target_ticker)
    df_processed = load_from_sqlite(f"processed_{target_ticker}")
    forecast_result = generate_7_day_forecast(df_processed)
    forecast_result["ticker"] = target_ticker

    initial_state = {
        "run_id": run_id,
        "ticker": target_ticker,
        "workflow": {
            "current_phase": "initialized",
            "model_health_status": "UNKNOWN",
            "retrain_count": 0,
            "max_retries": _configured_max_retries(),
            "retrain_attempted": False,
            "active_candidate": "champion",
            "price_scale_metadata": price_scale_metadata,
            "price_unit_detected": price_scale_metadata.get(target_ticker, {}).get("detected_unit", "unknown"),
            "price_scale_note": price_scale_metadata.get(target_ticker, {}).get("note", "Price scale metadata unavailable."),
        },
        "champion": {
            "forecast_data": forecast_result,
            "validation_metrics": forecast_result.get("validation_metrics", {}),
        },
        "challenger": {},
        "news": {},
        "improvement": {},
        "governance": {"final_model": "champion"},
        "recommendation": {},
        "audit": {
            "trail": [
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "phase": "pipeline_initialization",
                    "node_execution_status": "PASS",
                    "message": "Initial champion forecast prepared.",
                    "details": {"run_id": run_id},
                }
            ],
            "debug_paths": {},
            "errors": [],
        },
    }

    logger.info("Phase started | run_id=%s | ticker=%s | phase=agent_workflow", run_id, target_ticker)
    agent_app = build_agent_graph()
    final_state = agent_app.invoke(initial_state)

    logger.info("Phase started | run_id=%s | ticker=%s | phase=reporting", run_id, target_ticker)
    report_paths = generate_reports(final_state)

    duration = time.perf_counter() - started
    logger.info(
        "Pipeline completed successfully | run_id=%s | ticker=%s | duration_sec=%.2f | health=%s | "
        "technical_retrain_required=%s | retrain_attempted=%s | config_decision=%s | final_action=%s | confidence=%.2f",
        run_id,
        target_ticker,
        duration,
        final_state.get("workflow", {}).get("model_health_status", "UNKNOWN"),
        final_state.get("improvement", {}).get("technical_retrain_required", False),
        final_state.get("workflow", {}).get("retrain_attempted", False),
        final_state.get("governance", {}).get("decision", "NOT_REQUIRED"),
        final_state.get("recommendation", {}).get("final_action", "MANUAL_REVIEW"),
        final_state.get("recommendation", {}).get("confidence", 0.0),
    )
    logger.info("Reports saved | run_id=%s | paths=%s", run_id, report_paths)
    return final_state


def _check_and_apply_price_scale(raw_data: dict) -> dict:
    config = _load_data_config().get("price", {})
    unit = str(config.get("unit", "auto")).lower()
    normalize_to = str(config.get("normalize_to", "raw")).lower()
    warn_if_price_below = float(config.get("warn_if_price_below", 1000) or 1000)
    metadata = {}

    for ticker, df in raw_data.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        median_close = float(pd.to_numeric(df["close"], errors="coerce").dropna().median())
        detected_unit = unit if unit in {"vnd", "thousand_vnd"} else ("thousand_vnd" if median_close < warn_if_price_below else "vnd")
        note = (
            f"Median close is {median_close:.4f}; detected unit={detected_unit}; "
            f"normalize_to={normalize_to}. Model input scale was kept raw."
        )
        logger.info(
            "Price unit detected | ticker=%s | median_close=%.4f | detected_unit=%s | configured_unit=%s | normalize_to=%s",
            ticker,
            median_close,
            detected_unit,
            unit,
            normalize_to,
        )

        if normalize_to in {"vnd", "thousand_vnd"} and normalize_to != detected_unit:
            factor = 1000.0 if normalize_to == "vnd" and detected_unit == "thousand_vnd" else 0.001
            for column in ["open", "high", "low", "close"]:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce") * factor
            note = (
                f"Median close was {median_close:.4f}; detected unit={detected_unit}; "
                f"explicit config normalized OHLC to {normalize_to}."
            )
            logger.info("Price scale normalized | ticker=%s | from=%s | to=%s | factor=%s", ticker, detected_unit, normalize_to, factor)

        metadata[ticker] = {
            "median_close": median_close,
            "detected_unit": detected_unit,
            "configured_unit": unit,
            "normalize_to": normalize_to,
            "warn_if_price_below": warn_if_price_below,
            "note": note,
        }
    return metadata


def _load_data_config() -> dict:
    config_path = Path("configs/data_config.yaml")
    if not config_path.exists():
        return {"price": {"unit": "auto", "normalize_to": "raw", "warn_if_price_below": 1000}}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _configured_max_retries() -> int:
    config_path = Path("configs/agent_config.yaml")
    if not config_path.exists():
        return 1
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    try:
        return int((data.get("retrain") or {}).get("max_retries", (data.get("thresholds") or {}).get("max_retries", 1)))
    except (TypeError, ValueError):
        return 1
