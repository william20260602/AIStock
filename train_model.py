# train_model.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
from data_fetcher import fetch_stock_data
from features import add_features, FEATURE_COLS
import config
import warnings
warnings.filterwarnings('ignore')

def train_for_symbol(symbol):
    print(f"\n====== 开始训练股票: {symbol} ======")
    df = fetch_stock_data(symbol)
    if len(df) < config.TIME_STEPS + 100:
        print(f"   [!] 数据不足，跳过 {symbol}")
        return False

    df_feat = add_features(df, predict_days=config.PREDICT_DAYS)
    if len(df_feat) < config.TIME_STEPS:
        print(f"   [!] 特征处理后数据不足，跳过 {symbol}")
        return False

    feature_data = df_feat[FEATURE_COLS].values
    target_data = df_feat['target'].values

    split_idx = int(len(feature_data) * 0.8)
    train_X, test_X = feature_data[:split_idx], feature_data[split_idx:]
    train_y, test_y = target_data[:split_idx], target_data[split_idx:]

    scaler_X = StandardScaler()
    train_X_scaled = scaler_X.fit_transform(train_X)
    test_X_scaled = scaler_X.transform(test_X)

    scaler_y = StandardScaler()
    train_y_scaled = scaler_y.fit_transform(train_y.reshape(-1, 1)).ravel()
    test_y_scaled = scaler_y.transform(test_y.reshape(-1, 1)).ravel()

    print(f"   训练样本数: {len(train_X)}, 测试样本数: {len(test_X)}")

    lgb = LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    xgb = XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model = VotingRegressor(estimators=[('lgb', lgb), ('xgb', xgb)])
    model.fit(train_X_scaled, train_y_scaled)

    preds = model.predict(test_X_scaled)
    preds_real = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    test_y_real = scaler_y.inverse_transform(test_y_scaled.reshape(-1, 1)).ravel()
    mse = np.mean((preds_real - test_y_real) ** 2)
    print(f"   测试集 MSE: {mse:.6f}")

    symbol_dir = os.path.join(config.MODEL_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    joblib.dump(model, os.path.join(symbol_dir, "lgb_model.pkl"))
    joblib.dump(scaler_X, os.path.join(symbol_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(symbol_dir, "scaler_y.pkl"))
    print(f"   [+] 集成模型已保存至 {symbol_dir}")
    return True
