import numpy as np
import pandas as pd
from config import (
    METHOD, LOOKBACK_WINDOW, BACKTEST_INTERVAL, TOP_N, BOTTOM_N
)
from utils.excess_ret_comp import calculate_cumulative_excess_returns
from utils.helpers import generate_portfolio_starts, select_top_bottom

def backtest(stocks_data, start_date, end_date):
    stocks_data = stocks_data.groupby('PERMNO', group_keys=False).apply(
        calculate_cumulative_excess_returns,
        rolling_window=LOOKBACK_WINDOW,
        method=METHOD
    )

    portfolio_starts = generate_portfolio_starts(start_date, end_date, BACKTEST_INTERVAL)
    all_periods = []

    for start_dt in portfolio_starts:
        formation_date = start_dt - pd.offsets.MonthEnd(1)
        lookback_end = formation_date
        lookback_start = formation_date - pd.DateOffset(months=LOOKBACK_WINDOW - 1)

        lookback_df = stocks_data[
            (stocks_data['date'] >= lookback_start) &
            (stocks_data['date'] <= lookback_end)
        ]
        if lookback_df.empty:
            continue

        sub = lookback_df[lookback_df['date'] <= formation_date]
        sub = sub.groupby('PERMNO', group_keys=False).tail(1)

        winner_ids, loser_ids = select_top_bottom(sub, 'cum_excess_return', TOP_N, BOTTOM_N)

        hold_start = start_dt
        hold_end = start_dt + pd.DateOffset(months=BACKTEST_INTERVAL) - pd.Timedelta(days=1)

        w_hold = stocks_data[
            (stocks_data['PERMNO'].isin(winner_ids)) &
            (stocks_data['date'] >= hold_start) &
            (stocks_data['date'] <= hold_end)
        ]
        l_hold = stocks_data[
            (stocks_data['PERMNO'].isin(loser_ids)) &
            (stocks_data['date'] >= hold_start) &
            (stocks_data['date'] <= hold_end)
        ]

        if w_hold.empty or l_hold.empty:
            all_periods.append(pd.DataFrame({
                'test_period_start': [start_dt],
                'date': [None],
                'avg_u_w': [np.nan],
                'avg_u_l': [np.nan],
            }))
            continue

        w_m = w_hold.groupby('date', as_index=False)['RETX'].mean()
        w_m.rename(columns={'RETX':'avg_u_w'}, inplace=True)

        l_m = l_hold.groupby('date', as_index=False)['RETX'].mean()
        l_m.rename(columns={'RETX':'avg_u_l'}, inplace=True)

        all_months = pd.date_range(hold_start, hold_end, freq='M')
        tmp = pd.DataFrame({'date': all_months})
        tmp = tmp.merge(w_m, on='date', how='left').merge(l_m, on='date', how='left')

        if METHOD == 'sum':
            tmp['CAR_w'] = tmp['avg_u_w'].cumsum(skipna=True)
            tmp['CAR_l'] = tmp['avg_u_l'].cumsum(skipna=True)
        else:
            tmp['CAR_w'] = tmp['avg_u_w'].add(1).cumprod().sub(1)
            tmp['CAR_l'] = tmp['avg_u_l'].add(1).cumprod().sub(1)

        tmp['test_period_start'] = start_dt
        all_periods.append(tmp)

    df_out = pd.concat(all_periods, ignore_index=True)
    df_out.sort_values(['test_period_start','date'], inplace=True)
    return df_out
