# config.py
# ==================== 请修改下面这些 ====================
# 你要监控的股票代码（上海6开头，深圳0/3开头，不要加SH/SZ）
STOCK_LIST = [
    "605589",  # 圣泉集团
    "002851",  # 麦格米特
    "603276",  # 鸿远电子
]

# PushPlus 的 Token（免费，注册获取后粘贴到这里）
PUSHPLUS_TOKEN = "49673ecf6b96456da2c6b654026567b1"

# 钉钉机器人 Webhook（可选，不用就保留 ""）
DINGTALK_WEBHOOK = ""

# Telegram（可选，不用就保留 ""）
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
# ======================================================

# 预测参数（通常不用改）
TIME_STEPS = 20
PREDICT_DAYS = 5
BUY_THRESHOLD = 0.02
SELL_THRESHOLD = -0.02
MODEL_DIR = "models"
STATUS_DIR = "status"