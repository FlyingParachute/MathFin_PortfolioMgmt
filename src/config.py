import pandas as pd

DATA_PATH = './data/monthly/monthly.csv'
METHOD = 'sum'
MIN_MONTHS = 85

# 当为整数 N 时，取前N/后N只股票；当为 0< float <1 时，取对应百分比
TOP_N = 35
BOTTOM_N = 35

# 向后计算累计收益的时间窗口（单位：月）
LOOKBACK_WINDOW = 24

# 回测间隔长度（单位：月）
BACKTEST_INTERVAL = 24

START_DATE = pd.Timestamp('1930-01-01')
END_DATE   = pd.Timestamp('1975-01-01')
