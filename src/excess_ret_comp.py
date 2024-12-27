##############################################
# 该文件包含计算超额收益的函数。
##############################################

import pandas as pd


def calculate_cumulative_excess_returns(group,methods):
    """
    对每只股票按日期顺序,按原文应为加和而非乘积
    """
    # group 已按日期排序
    if methods == 'sum':
        group['cum_excess_return'] = group['RETX'].rolling(window=36, min_periods=36).sum()
    elif methods == 'prod':
        group['cum_excess_return'] = group['RETX'].rolling(window=36, min_periods=36).apply(lambda x: (1+x).prod()-1)
    return group

def direct_substract(data, ret_col='RETX', market_ret_col='ewretd'):
    """
    计算超额收益：股票收益减去市场收益。

    参数:
    - data: DataFrame，包含股票收益和市场收益的数据。
    - ret_col: str，股票收益列名（默认为 'RETX'）。
    - market_ret_col: str，市场收益列名（默认为 'ewretd'）。

    返回:
    - DataFrame，包含超额收益的新数据。
    """
    # 转换列为数值型，忽略非数值的无效值
    data[ret_col] = pd.to_numeric(data[ret_col], errors='coerce')
    data[market_ret_col] = pd.to_numeric(data[market_ret_col], errors='coerce')

    # 计算超额收益
    data[ret_col] = data[ret_col] - data[market_ret_col]
    return data


