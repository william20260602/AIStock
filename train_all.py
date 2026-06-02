# train_all.py
import os
import time
import config
from train_model import train_for_symbol

if __name__ == "__main__":
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    for i, symbol in enumerate(config.STOCK_LIST):
        train_for_symbol(symbol)
        if i < len(config.STOCK_LIST) - 1:  # 最后一只不用等
            print(f"\n等待 10 秒再训练下一只...")
            time.sleep(10)
    print("\n所有股票训练完成！")