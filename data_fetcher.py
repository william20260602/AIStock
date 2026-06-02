# data_fetcher.py
import baostock as bs
import pandas as pd
import datetime

def fetch_stock_data(symbol, start_date="2015-01-01", end_date="2030-12-31"):
    # 自动转换代码格式
    if symbol.startswith('6'):
        bs_symbol = f"sh.{symbol}"
    else:
        bs_symbol = f"sz.{symbol}"

    # 如果 end_date 在今天之后，则自动使用今天日期
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    if end_date > today_str:
        end_date = today_str

    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"baostock 登录失败: {lg.error_msg}")

    rs = bs.query_history_k_data_plus(
        bs_symbol,
        "date,open,high,low,close,volume,amount,turn",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"
    )

    if rs.error_code != '0':
        bs.logout()
        raise Exception(f"查询失败: {rs.error_msg}")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.rename(columns={'turn': 'turnover'}, inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df