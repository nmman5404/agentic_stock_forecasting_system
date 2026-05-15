from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from utils.logger import get_logger

logger = get_logger("WalkForwardValidation")

def _load_config() -> dict:
    """Đọc file config một lần gọn nhẹ."""
    try:
        with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        logger.warning("Could not load model_config.yaml, using defaults.")
        return {}
    
def load_model_params() -> Dict[str, Any]:
    """Trả về cấu hình lightgbm_params hiện tại cho Pipeline."""
    config = _load_config()
    return config.get("lightgbm_params", {})

def run_walk_forward_validation(
    df: pd.DataFrame,
    target_col: str = "close",
    model_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Chạy Walk-forward validation và trả về trực tiếp dictionary Metrics."""

    config = _load_config()
    wf_cfg = config.get("walk_forward_validation", {})
    lgb_params = model_params or config.get("lightgbm_params", {})

    # Đọc cấu hình walk-forward
    horizon = int(wf_cfg.get("horizon", 1))
    step_size = int(wf_cfg.get("step_size", 20))
    val_window = int(wf_cfg.get("validation_window", 20))
    max_windows = wf_cfg.get("max_windows", 8)
    quantiles = wf_cfg.get("quantiles", [0.025, 0.1, 0.5, 0.9, 0.975])
    initial_train_size = int(wf_cfg.get("initial_train_size", 252))

    X, y_return, current_close, actual_price, feature_cols = _prepare_supervised_data(df, target_col, horizon)
    sample_count = len(X)
    
    # Kiểm tra an toàn dữ liệu
    min_required = max(30, val_window + 2)
    if sample_count < min_required or len(feature_cols) == 0:
        logger.warning(f"Insufficient validation data: samples={sample_count}, features={len(feature_cols)}")
        return {"status": "INSUFFICIENT_DATA", "metrics": {}}

    # Tính toán các điểm bắt đầu fold (cửa sổ trượt)
    initial_train_size = min(max(initial_train_size, 30), sample_count - val_window)
    fold_starts = list(range(initial_train_size, sample_count, step_size))
    if max_windows and len(fold_starts) > max_windows:
        fold_starts = fold_starts[-int(max_windows):]

    all_predictions = []

    # Chạy trượt trên các Fold
    for train_end in fold_starts:
        val_start = train_end
        val_end = min(train_end + val_window, sample_count)
        if val_end <= val_start:
            continue

        X_train, y_train = X.iloc[:train_end], y_return.iloc[:train_end]
        X_val = X.iloc[val_start:val_end]

        pred_df = _predict_quantile_prices(
            X_train, y_train, X_val, 
            current_close.iloc[val_start:val_end], 
            actual_price.iloc[val_start:val_end], 
            quantiles, lgb_params
        )
        all_predictions.append(pred_df)

    if not all_predictions:
        return {"status": "INSUFFICIENT_DATA", "metrics": {}}

    # Gộp dự báo và tính Metrics tổng
    combined_preds = pd.concat(all_predictions, axis=0)
    metrics = _calculate_metrics(combined_preds, quantiles)

    logger.info(
        "Walk-forward validation complete: folds=%s, MAE=%.4f, RMSE=%.4f, MAPE=%.4f, directional_accuracy=%.4f",
        len(all_predictions), metrics['mae'], metrics['rmse'], metrics['mape'], metrics['directional_accuracy']
    )

    return {
        "evaluation_method": "walk_forward",
        "status": "COMPLETED",
        "fold_count": len(all_predictions),
        "sample_count": sample_count,
        "feature_count": len(feature_cols),
        "metrics": metrics
    }

def _prepare_supervised_data(df: pd.DataFrame, target_col: str, horizon: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
    ordered = df.copy().sort_index() if isinstance(df.index, pd.DatetimeIndex) else df.copy()
    
    excluded = {"ticker", "date", "target"}
    numeric_features = [col for col in ordered.columns if col not in excluded and pd.api.types.is_numeric_dtype(ordered[col])]

    # Mục tiêu dự báo là % lợi nhuận (Return) thay vì giá tuyệt đối
    target_return = (ordered[target_col].shift(-horizon) - ordered[target_col]) / ordered[target_col]
    
    supervised = ordered[numeric_features].copy()
    supervised["target"] = target_return
    supervised["current_close"] = ordered[target_col]
    supervised["actual_price"] = ordered[target_col].shift(-horizon)
    supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna()

    return supervised[numeric_features], supervised["target"], supervised["current_close"], supervised["actual_price"], numeric_features

def _predict_quantile_prices(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame,
    current_close: pd.Series, actual_price: pd.Series,
    quantiles: Sequence[float], model_params: Dict[str, Any]
) -> pd.DataFrame:
    
    prediction_data = {
        "current_close": current_close.astype(float).to_numpy(),
        "actual_price": actual_price.astype(float).to_numpy()
    }
    raw_quantile_prices = []

    # Train LightGBM cho từng mức Quantile
    for q in quantiles:
        params = {**model_params, "objective": "quantile", "alpha": float(q)}
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Chuyển đổi return dự báo ngược lại thành giá
        predicted_return = model.predict(X_val)
        predicted_price = current_close.to_numpy(dtype=float) * (1.0 + predicted_return)
        raw_quantile_prices.append(predicted_price.astype(float))

    raw_matrix = np.vstack(raw_quantile_prices).T
    sorted_matrix = np.sort(raw_matrix, axis=1) # Sort chống cắt chéo quantile
    crossed_rows = np.any(np.diff(raw_matrix, axis=1) < 0, axis=1)

    for i, q in enumerate(quantiles):
        prediction_data[f"q_{q:g}"] = sorted_matrix[:, i]
        
    prediction_data["quantile_crossed"] = crossed_rows
    return pd.DataFrame(prediction_data, index=X_val.index)

def _calculate_metrics(predictions: pd.DataFrame, quantiles: Sequence[float]) -> Dict[str, float]:
    actual = predictions["actual_price"].to_numpy(dtype=float)
    current = predictions["current_close"].to_numpy(dtype=float)
    median = predictions["q_0.5"].to_numpy(dtype=float)

    error = median - actual
    actual_safe = np.where(np.abs(actual) < 1e-12, np.nan, actual)
    
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(np.square(error)))
    mape = np.mean(np.nan_to_num(np.abs(error) / np.abs(actual_safe)))
    smape = np.mean(np.nan_to_num(2.0 * np.abs(error) / (np.abs(actual_safe) + np.abs(median))))
    
    # Dự đoán đúng hướng lên/xuống
    directional_accuracy = np.mean(np.sign(actual - current) == np.sign(median - current))

    # Độ bao phủ (Coverage)
    interval_80_coverage = np.mean((actual >= predictions["q_0.1"].to_numpy()) & (actual <= predictions["q_0.9"].to_numpy()))
    interval_95_coverage = np.mean((actual >= predictions["q_0.025"].to_numpy()) & (actual <= predictions["q_0.975"].to_numpy()))

    # Pinball Loss cho đánh giá mô hình phân vị (Quantile Regression)
    pinball_values = []
    for q in quantiles:
        q_pred = predictions[f"q_{q:g}"].to_numpy(dtype=float)
        residual = actual - q_pred
        pinball_values.append(np.mean(np.maximum(q * residual, (q - 1.0) * residual)))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape),
        "directional_accuracy": float(directional_accuracy),
        "interval_80_coverage": float(interval_80_coverage),
        "interval_95_coverage": float(interval_95_coverage),
        "pinball_loss": float(np.mean(pinball_values)),
        "prediction_bias": float(np.mean(error)),
        "prediction_bias_pct": float(np.mean(np.nan_to_num(error / actual_safe))),
        "quantile_crossing_rate": float(np.mean(predictions["quantile_crossed"].astype(float)))
    }