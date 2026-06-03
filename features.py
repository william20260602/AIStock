# features.py
import pandas as pd
import numpy as np

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'ma5', 'ma10', 'ma20',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'volume_roc', 'return_1', 'return_5', 'return_10',
    'high_low_pct', 'close_open_pct'
]

def add_features(df, predict_days=5):
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_middle'] = bb_mid
    df['bb_lower'] = bb_mid - 2 * bb_std

    df['volume_roc'] = df['volume'].pct_change(5)
    df['high_low_pct'] = (df['high'] - df['low']) / df['close']
    df['close_open_pct'] = (df['close'] - df['open']) / df['open']

    df['target'] = df['close'].pct_change(predict_days).shift(-predict_days)
    df.dropna(inplace=True)
    return df
