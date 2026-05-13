import pandas as pd
from src.modeling.trainer import QuantileLightGBM
from src.modeling.validation import run_walk_forward_validation
from utils.logger import get_logger

logger = get_logger("Predictor")

def generate_7_day_forecast(df: pd.DataFrame):
    """
    Tạo dự báo cho 7 ngày tiếp theo kèm Metrics.
    """
    ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else 'Stock' ## Cố gắng lấy ticker từ cột 'ticker', nếu không có thì đặt tên chung là 'Stock'
    logger.info("Forecast generation started | ticker=%s | horizon_days=7", ticker)
    
    trainer = QuantileLightGBM()
    
    # 1. Tính toán Holdout Metrics
    metrics = trainer.evaluate_holdout(df)
    validation_metrics = run_walk_forward_validation(df, target_col=trainer.target_col)
    
    # 2. Chuẩn bị dòng dữ liệu cuối cùng (Today) để làm Input dự đoán Tương lai
    last_row = df.iloc[[-1]]
    X_last = last_row[[col for col in df.columns if col not in ['ticker', 'date', 'target']]]
    
    # 3. Lặp 7 ngày (Direct Multi-step)
    forecasts = []
    logger.info("Quantile forecast training started | ticker=%s | steps=7 | quantiles=5", ticker)
    
    for step in range(1, 8):
        step_result = trainer.train_and_predict_step(df, X_last, step)
        forecasts.append(step_result)
        
    logger.info("Forecast generation completed | ticker=%s | horizon_days=7", ticker)
    
    return {
        "ticker": ticker,
        "current_price": float(df["close"].iloc[-1]) if "close" in df.columns and not df.empty else None,
        "as_of_date": df.index[-1].strftime("%Y-%m-%d") if len(df.index) else None,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "forecasts": forecasts
    }
