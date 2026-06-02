# daily_alert.py
import os
import json
import numpy as np
import pandas as pd
import joblib
import config
from data_fetcher import fetch_stock_data
from features import add_features, FEATURE_COLS
from utils_push import send_alert

def load_status(symbol):
    os.makedirs(config.STATUS_DIR, exist_ok=True)
    status_file = os.path.join(config.STATUS_DIR, f"{symbol}.json")
    try:
        with open(status_file, 'r') as f:
            return json.load(f)['holding']
    except:
        return False

def save_status(symbol, holding):
    status_file = os.path.join(config.STATUS_DIR, f"{symbol}.json")
    with open(status_file, 'w') as f:
        json.dump({'holding': holding}, f)

def daily_check():
    for symbol in config.STOCK_LIST:
        try:
            end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y%m%d")
            df = fetch_stock_data(symbol, end_date=end_date)
            if len(df) < config.TIME_STEPS + 5:
                print(f"{symbol} 数据不足，跳过")
                continue

            df_feat = add_features(df, predict_days=config.PREDICT_DAYS)
            if len(df_feat) < config.TIME_STEPS:
                continue

            recent = df_feat[FEATURE_COLS].values[-config.TIME_STEPS:].flatten().reshape(1, -1)
            current_price = df_feat['close'].values[-1]

            model_path = os.path.join(config.MODEL_DIR, symbol, "mlp_model.pkl")
            scaler_X_path = os.path.join(config.MODEL_DIR, symbol, "scaler_X.pkl")
            scaler_y_path = os.path.join(config.MODEL_DIR, symbol, "scaler_y.pkl")
            if not os.path.exists(model_path):
                print(f"{symbol} 未找到模型，请先训练")
                continue

            model = joblib.load(model_path)
            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)

            recent_scaled = scaler_X.transform(recent)
            pred_scaled = model.predict(recent_scaled)
            pred_return = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

            if pred_return > config.BUY_THRESHOLD:
                signal = "买入"
            elif pred_return < config.SELL_THRESHOLD:
                signal = "卖出"
            else:
                signal = "持有"

            is_holding = load_status(symbol)
            if signal == "买入" and not is_holding:
                send_alert(symbol, "买入", pred_return, current_price)
                save_status(symbol, True)
                print(f"{symbol} 买入提醒已发送")
            elif signal == "卖出" and is_holding:
                send_alert(symbol, "卖出", pred_return, current_price)
                save_status(symbol, False)
                print(f"{symbol} 卖出提醒已发送")
            else:
                print(f"{symbol} 当前状态：持有={is_holding}，信号={signal}，不推送")

        except Exception as e:
            print(f"{symbol} 处理出错: {e}")

if __name__ == "__main__":
    import datetime, os
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_log.txt")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        daily_check()
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"{now_str} - 执行成功\n")
    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"{now_str} - 出错: {e}\n")