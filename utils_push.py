# utils_push.py
import requests
import pandas as pd
import config

def pushplus(title, content):
    if not config.PUSHPLUS_TOKEN:
        return
    url = "http://www.pushplus.plus/send"
    data = {"token": config.PUSHPLUS_TOKEN, "title": title, "content": content}
    try:
        requests.post(url, json=data, timeout=10)
    except Exception as e:
        print(f"PushPlus 发送失败: {e}")

def dingtalk(msg):
    if not config.DINGTALK_WEBHOOK:
        return
    headers = {'Content-Type': 'application/json'}
    data = {"msgtype": "text", "text": {"content": msg}}
    try:
        requests.post(config.DINGTALK_WEBHOOK, json=data, headers=headers, timeout=10)
    except Exception as e:
        print(f"钉钉发送失败: {e}")

def telegram(msg):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print(f"Telegram 发送失败: {e}")

def send_alert(symbol, signal, pred_return, current_price):
    msg = (f"【{symbol} - {signal}提醒】\n"
           f"预测未来{config.PREDICT_DAYS}日收益率：{pred_return:.2%}\n"
           f"当前价格：{current_price:.2f}\n"
           f"时间：{pd.Timestamp.now()}")
    title = f"{symbol} {signal}"
    pushplus(title, msg)
    dingtalk(msg)
    telegram(msg)