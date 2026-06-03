# train_model.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import lightgbm as lgb
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

    # 不再需要时间序列窗口，直接用当天的特征预测未来收益率
    feature_data = df_feat[FEATURE_COLS].values
    target_data = df_feat['target'].values

    # 划分训练/测试集（按时间顺序，避免未来信息）
    split_idx = int(len(feature_data) * 0.8)
    train_X = feature_data[:split_idx]
    train_y = target_data[:split_idx]
    test_X = feature_data[split_idx:]
    test_y = target_data[split_idx:]

    # 标准化（LightGBM 不需要，但为了和 scaler_y 对齐，我们仍保留 scaler_X 供 daily_alert 使用）
    scaler_X = StandardScaler()
    train_X_scaled = scaler_X.fit_transform(train_X)
    test_X_scaled = scaler_X.transform(test_X)

    # 对目标做归一化，以便还原预测值
    scaler_y = StandardScaler()
    train_y_scaled = scaler_y.fit_transform(train_y.reshape(-1, 1)).ravel()
    test_y_scaled = scaler_y.transform(test_y.reshape(-1, 1)).ravel()

    print(f"   训练样本数: {len(train_X)}, 测试样本数: {len(test_X)}")

    # 创建 LightGBM 数据集
    train_data = lgb.Dataset(train_X_scaled, label=train_y_scaled)
    valid_data = lgb.Dataset(test_X_scaled, label=test_y_scaled, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    # 评估
    preds = model.predict(test_X_scaled)
    preds_real = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    test_y_real = scaler_y.inverse_transform(test_y_scaled.reshape(-1, 1)).ravel()
    mse = np.mean((preds_real - test_y_real) ** 2)
    print(f"   测试集 MSE: {mse:.6f}")

    # 保存模型和 scaler
    symbol_dir = os.path.join(config.MODEL_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    joblib.dump(model, os.path.join(symbol_dir, "lgb_model.pkl"))
    joblib.dump(scaler_X, os.path.join(symbol_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(symbol_dir, "scaler_y.pkl"))
    print(f"   [+] 模型已保存至 {symbol_dir}")
    return True
