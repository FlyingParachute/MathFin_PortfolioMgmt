import pandas as pd

DATA_PATH = './data/monthly/monthly.csv'
METHOD = 'sum'
MIN_MONTHS = 85


TOP_N = 35
BOTTOM_N = 35

LOOKBACK_WINDOW = 24


BACKTEST_INTERVAL = 24

START_DATE = pd.Timestamp('1930-01-01')
END_DATE   = pd.Timestamp('1975-01-01')
