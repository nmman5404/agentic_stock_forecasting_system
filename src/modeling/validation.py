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
