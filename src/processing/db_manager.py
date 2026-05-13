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
