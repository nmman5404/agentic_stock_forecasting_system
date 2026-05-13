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