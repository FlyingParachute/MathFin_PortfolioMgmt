##############################################
# 该文件包含计算超额收益的函数。
##############################################

import pandas as pd


import pandas as pd

def calculate_cumulative_excess_returns(group, rolling_window, method):
    if method == 'sum':
        group['cum_excess_return'] = group['RETX'].rolling(
            window=rolling_window,
            min_periods=rolling_window
        ).sum()
    else:
        group['cum_excess_return'] = group['RETX'].rolling(
            window=rolling_window,
            min_periods=rolling_window
        ).apply(lambda x: (1 + x).prod() - 1)
    return group

def direct_substract(df, ret_col='RETX', market_ret_col='ewretd'):
    df[ret_col] = pd.to_numeric(df[ret_col], errors='coerce')
    df[market_ret_col] = pd.to_numeric(df[market_ret_col], errors='coerce')
    df[ret_col] = df[ret_col] - df[market_ret_col]
    return df


