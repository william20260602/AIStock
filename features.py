# features.py
import pandas as pd
import numpy as np

# 更新特征列，新增 MACD、布林带、KDJ 等
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'ma5', 'ma20', 'ma60',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'k', 'd', 'j',
    'volume_roc', 'return_1', 'return_5'
]

def add_features(df, predict_days=5):
    # 基础收益率
    df['return'] = df['close'].pct_change()
    df['return_1'] = df['close'].pct_change(1)    # 今日涨跌
    df['return_5'] = df['close'].pct_change(5)    # 5日涨跌幅

    # 均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    # RSI (14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 布林带 (20,2)
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_middle'] = bb_mid
    df['bb_lower'] = bb_mid - 2 * bb_std

    # KDJ (9,3,3)
    low_9 = df['low'].rolling(9).min()
    high_9 = df['high'].rolling(9).max()
    rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
    df['k'] = rsv.ewm(com=2).mean()
    df['d'] = df['k'].ewm(com=2).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # 成交量变化率
    df['volume_roc'] = df['volume'].pct_change(5)

    # 预测目标：未来 predict_days 日收益率
    df['target'] = df['close'].pct_change(predict_days).shift(-predict_days)

    df.dropna(inplace=True)
    return df
