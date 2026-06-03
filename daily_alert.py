name: AI Stock Alert

on:
  schedule:
    # 北京时间周一至周五 15:30 (UTC 7:30)
    - cron: '30 7 * * 1-5'
  workflow_dispatch:   # 允许手动运行

permissions:
  contents: write

env:
  FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true

jobs:
  run-alert:
    runs-on: ubuntu-latest
    steps:
      - name: 检出代码
        uses: actions/checkout@v4

      - name: 设置 Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: 安装依赖
        run: |
          pip install pandas numpy scikit-learn joblib baostock schedule requests lightgbm

      - name: 从 Secrets 生成配置文件
        run: |
          echo "STOCK_LIST = '${{ secrets.STOCK_LIST }}'.split(',')" > config_temp.py
          echo "PUSHPLUS_TOKEN = '${{ secrets.PUSHPLUS_TOKEN }}'" >> config_temp.py
          echo "DINGTALK_WEBHOOK = ''" >> config_temp.py
          echo "TELEGRAM_BOT_TOKEN = ''" >> config_temp.py
          echo "TELEGRAM_CHAT_ID = ''" >> config_temp.py
          echo "TIME_STEPS = 20" >> config_temp.py
          echo "PREDICT_DAYS = 5" >> config_temp.py
          echo "BUY_THRESHOLD = 0.02" >> config_temp.py
          echo "SELL_THRESHOLD = -0.02" >> config_temp.py
          echo "MODEL_DIR = 'models'" >> config_temp.py
          echo "STATUS_DIR = 'status'" >> config_temp.py
          mv config_temp.py config.py

      - name: 训练模型
        run: python train_all.py

      - name: 运行每日提醒
        run: python daily_alert.py

      - name: 提交并推送持仓状态文件
        if: always()
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add status/
          git diff --quiet && git diff --staged --quiet || (git commit -m "更新持仓状态" && git push)

      - name: 上传运行日志（可选）
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: run-log
          path: run_log.txt
