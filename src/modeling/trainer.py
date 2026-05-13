import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import yaml
import os
from utils.logger import get_logger

logger = get_logger("ModelTrainer")

def load_config():
    with open("configs/model_config.yaml", "r") as f:
        return yaml.safe_load(f)

class QuantileLightGBM:
    def __init__(self, target_col='close', holdout_days=30):
        self.target_col = target_col
        self.holdout_days = holdout_days
        self.config = load_config()['lightgbm_params']
        self.quantiles = [0.025, 0.1, 0.5, 0.9, 0.975]
        self.features =[]

    def prepare_data(self, df, step=1):
        """
        Thay vì dự đoán Giá, ta dự đoán % Thay đổi (Return) từ ngày T đến ngày T + step.
        """
        df = df.copy()
        self.features = [col for col in df.columns if col not in ['ticker', 'date', 'target']]
        
        # Target = (Close_{T+step} - Close_T) / Close_T
        df['target'] = (df[self.target_col].shift(-step) - df[self.target_col]) / df[self.target_col]
        
        df_train = df.dropna(subset=['target'])
        return df_train[self.features], df_train['target']

    def evaluate_holdout(self, df):
        ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else 'Unknown'
        logger.info("Holdout evaluation started | ticker=%s | holdout_days=%s", ticker, self.holdout_days)
        
        X, y_return = self.prepare_data(df, step=1)
        
        train_size = len(X) - self.holdout_days
        X_train, y_train = X.iloc[:train_size], y_return.iloc[:train_size]
        X_test, y_test = X.iloc[train_size:], y_return.iloc[train_size:]
        
        params = self.config.copy()
        params['objective'] = 'quantile'
        params['alpha'] = 0.5
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Dự đoán Return
        pred_returns = model.predict(X_test)
        
        # KHÔI PHỤC LẠI GIÁ (Reconstruct Price) để tính Metrics cho trực quan
        # Giá hiện tại (Close_T) của tập test
        current_close = df[self.target_col].iloc[train_size : train_size + len(X_test)].values
        
        pred_prices = current_close * (1 + pred_returns)
        actual_prices = current_close * (1 + y_test.values) # Tương đương Close_{T+1}
        
        mae = mean_absolute_error(actual_prices, pred_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
        mape = mean_absolute_percentage_error(actual_prices, pred_prices)
        
        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
        logger.info(
            "Holdout evaluation completed | ticker=%s | mae=%.2f | rmse=%.2f | mape=%.4f",
            ticker,
            mae,
            rmse,
            mape,
        )
        return metrics

    def train_and_predict_step(self, df, X_last, step):
        X, y = self.prepare_data(df, step=step)
        current_close = df[self.target_col].iloc[-1] # Giá ở thời điểm T hiện tại
        
        step_forecast = {"step": step}
        for q in self.quantiles:
            params = self.config.copy()
            params['objective'] = 'quantile'
            params['alpha'] = q
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            
            # Dự đoán Future Return và Khôi phục lại mức Giá
            pred_return = model.predict(X_last)[0]
            pred_price = current_close * (1 + pred_return)
            
            step_forecast[f"q_{q}"] = pred_price
            
        return step_forecast
