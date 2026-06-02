# train_model.py （使用 scikit-learn MLP，无需 TensorFlow）
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import joblib
from data_fetcher import fetch_stock_data
from features import add_features, FEATURE_COLS
import config

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

    # 构建样本：将过去 TIME_STEPS 天的所有特征拉平成一维向量
    X_list, y_list = [], []
    for i in range(config.TIME_STEPS, len(df_feat)):
        past_features = df_feat[FEATURE_COLS].iloc[i-config.TIME_STEPS:i].values.flatten()
        X_list.append(past_features)
        y_list.append(df_feat['target'].iloc[i])
    X = np.array(X_list)
    y = np.array(y_list)

    # 划分训练/测试集
    split_idx = int(len(X) * 0.8)
    train_X, test_X = X[:split_idx], X[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]

    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    train_X_scaled = scaler_X.fit_transform(train_X)
    train_y_scaled = scaler_y.fit_transform(train_y.reshape(-1, 1)).ravel()
    test_X_scaled = scaler_X.transform(test_X)
    test_y_scaled = scaler_y.transform(test_y.reshape(-1, 1)).ravel()

    print(f"   训练样本数: {train_X_scaled.shape[0]}, 测试样本数: {test_X_scaled.shape[0]}")

    # 建立 MLP 神经网络模型
    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='adam', max_iter=200, random_state=42)
    model.fit(train_X_scaled, train_y_scaled)

    # 评估
    train_score = model.score(train_X_scaled, train_y_scaled)
    test_score = model.score(test_X_scaled, test_y_scaled)
    print(f"   训练集 R²: {train_score:.4f}, 测试集 R²: {test_score:.4f}")

    # 保存模型和缩放器
    symbol_dir = os.path.join(config.MODEL_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    joblib.dump(model, os.path.join(symbol_dir, "mlp_model.pkl"))
    joblib.dump(scaler_X, os.path.join(symbol_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(symbol_dir, "scaler_y.pkl"))
    print(f"   [+] 模型已保存至 {symbol_dir}")
    return True