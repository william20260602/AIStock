# features.py
import pandas as pd
import numpy as np

FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume',
                'ma5', 'ma20', 'rsi', 'volume_roc']

def add_features(df, predict_days=5):
    df['return'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['volume_roc'] = df['volume'].pct_change(5)
    df['target'] = df['close'].pct_change(periods=predict_days).shift(-predict_days)
    df.dropna(inplace=True)
    return df