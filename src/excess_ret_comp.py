##############################################
# 该文件包含计算超额收益的函数。
##############################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


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



def calculate_alpha(data, ret_col='RETX', market_ret_col='ewretd', permno_col='PERMNO'):
    """
    计算每只股票的 CAPM alpha，忽略无风险利率 R_f，直接用市场回报率 R_m 替代市场超额收益。

    参数:
    - data: DataFrame，包含股票收益、市场收益以及股票标识符的数据。
    - stock_col: str，股票收益列名（默认为 'RETX'）。
    - market_col: str，市场收益列名（默认为 'ewretd'）。
    - permno_col: str，股票标识符列名（默认为 'PERMNO'）。

    返回:
    - DataFrame，包含 alpha 值的新数据。
    """
   

    # 确保数据为数值型
    data[ret_col] = pd.to_numeric(data[ret_col], errors='coerce')
    data[market_ret_col] = pd.to_numeric(data[market_ret_col], errors='coerce')

    # 创建结果列
    data['alpha'] = np.nan

    # 按每只股票分组计算 alpha
    grouped = data.groupby(permno_col)
    for permno, group in grouped:
        valid_data = group.dropna(subset=[ret_col, market_ret_col])

        if len(valid_data) > 0:
            # 准备回归模型
            model = LinearRegression()
            X = valid_data[[market_ret_col]].values  # 市场回报率
            y = valid_data[ret_col].values  # 股票收益

            # 拟合模型
            model.fit(X, y)
            predictions = model.predict(X)
            alphas = y - predictions  # 计算 alpha

            # 回填 alpha 到原始数据
            data.loc[valid_data.index, 'alpha'] = alphas
    
    data[ret_col] = data['alpha'] 

    return data

