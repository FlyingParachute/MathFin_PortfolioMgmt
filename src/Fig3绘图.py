import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import PchipInterpolator
warnings.filterwarnings("ignore")
from excess_ret_comp import calculate_alpha, direct_substract, calculate_cumulative_excess_returns

data_path = './data/monthly/monthly.csv'
monthly_raw_data = pd.read_csv(data_path)
excess_cal_method = 'direct'
method = 'prod'

if excess_cal_method == 'direct':
    monthly_raw_data = direct_substract(monthly_raw_data)
elif excess_cal_method == 'capm':
    monthly_raw_data = calculate_alpha(monthly_raw_data)

cleaned_data = monthly_raw_data.dropna(subset=['RETX'])
cleaned_data = cleaned_data[~cleaned_data['RETX'].isin(['B','C'])]
cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])
stocks_with_enough_data = cleaned_data.groupby('PERMNO').filter(lambda x: len(x) >= 85)
stocks_with_enough_data = stocks_with_enough_data.sort_values(['PERMNO','date']).reset_index(drop=True)
stocks_with_enough_data = (
    stocks_with_enough_data
    .groupby('PERMNO', group_keys=False)
    .apply(calculate_cumulative_excess_returns, methods=method)
)

dates = pd.date_range(start='1933-01-01', end='1978-01-01', freq='AS')

all_periods = []

for start_dt in dates:
    formation_date = start_dt - pd.offsets.MonthEnd(1)
    lookback_end = formation_date
    lookback_start = formation_date - pd.DateOffset(months=60)
    lookback_df = stocks_with_enough_data[
        (stocks_with_enough_data['date'] >= lookback_start) &
        (stocks_with_enough_data['date'] <= lookback_end)
    ].copy()

    if lookback_df.empty:
        continue

    sub = lookback_df[lookback_df['date'] <= formation_date]
    portfolio_data = sub.groupby('PERMNO', group_keys=False).tail(1).copy()
    portfolio_data = portfolio_data.sort_values('cum_excess_return', ascending=False)
    portfolio_data = portfolio_data.dropna(subset=['cum_excess_return'])
    winner_ids = portfolio_data.head(35)['PERMNO']
    loser_ids  = portfolio_data.tail(35)['PERMNO']

    hold_start = start_dt
    hold_end   = start_dt + pd.DateOffset(months=60) - pd.Timedelta(days=1)

    w_hold = stocks_with_enough_data[
        (stocks_with_enough_data['PERMNO'].isin(winner_ids)) &
        (stocks_with_enough_data['date'] >= hold_start) &
        (stocks_with_enough_data['date'] <= hold_end)
    ].copy()
    l_hold = stocks_with_enough_data[
        (stocks_with_enough_data['PERMNO'].isin(loser_ids)) &
        (stocks_with_enough_data['date'] >= hold_start) &
        (stocks_with_enough_data['date'] <= hold_end)
    ].copy()

    if w_hold.empty or l_hold.empty:
        all_periods.append(pd.DataFrame({
            'test_period_start': [start_dt],
            'date': [None],
            'avg_u_w': [np.nan],
            'avg_u_l': [np.nan],
        }))
        continue

    w_monthly = w_hold.groupby('date', as_index=False)['RETX'].mean()
    w_monthly.rename(columns={'RETX':'avg_u_w'}, inplace=True)
    l_monthly = l_hold.groupby('date', as_index=False)['RETX'].mean()
    l_monthly.rename(columns={'RETX':'avg_u_l'}, inplace=True)
    all_months = pd.date_range(hold_start, hold_end, freq='M')
    hold_months_df = pd.DataFrame({'date':all_months})
    merged = (
        hold_months_df
        .merge(w_monthly, on='date', how='left')
        .merge(l_monthly, on='date', how='left')
    )

    if method == 'sum':
        merged['CAR_w'] = merged['avg_u_w'].cumsum(skipna=True)
        merged['CAR_l'] = merged['avg_u_l'].cumsum(skipna=True)
    elif method == 'prod':
        merged['CAR_w'] = merged['avg_u_w'].add(1).cumprod().sub(1)
        merged['CAR_l'] = merged['avg_u_l'].add(1).cumprod().sub(1)

    merged['test_period_start'] = start_dt
    all_periods.append(merged)

all_periods_df = pd.concat(all_periods, ignore_index=True)
all_periods_df.sort_values(['test_period_start','date'], inplace=True)

def add_relative_month(df_):
    df_ = df_.sort_values('date').reset_index(drop=True)
    df_['t'] = np.arange(len(df_)) + 1
    return df_

all_periods_df = all_periods_df.groupby('test_period_start', group_keys=False).apply(add_relative_month)
acar = all_periods_df.groupby('t', as_index=False).agg({
    'CAR_w':'mean',
    'CAR_l':'mean'
})
acar.rename(columns={'CAR_w':'ACAR_w','CAR_l':'ACAR_l'}, inplace=True)
acar['diff'] = acar['ACAR_l'] - acar['ACAR_w']

grouped_n = all_periods_df.groupby(['t','test_period_start'], as_index=False).agg({
    'CAR_w':'last',
    'CAR_l':'last'
})
ttest_list = []
unique_t = sorted(grouped_n['t'].unique())
for t_ in unique_t:
    sub = grouped_n[grouped_n['t'] == t_]
    w_vals = sub['CAR_w'].dropna()
    l_vals = sub['CAR_l'].dropna()
    if len(w_vals) < 2 or len(l_vals) < 2:
        t_stat, p_val = np.nan, np.nan
    else:
        t_stat, p_val = ttest_ind(l_vals, w_vals, nan_policy='omit')
    ttest_list.append({
        't': t_,
        'N_w': len(w_vals),
        'N_l': len(l_vals),
        't_stat': t_stat,
        'p_value': p_val
    })
ttest_df = pd.DataFrame(ttest_list)

t_vals = acar['t'].values
w_vals = acar['ACAR_w'].values
l_vals = acar['ACAR_l'].values
t_w = np.insert(t_vals, 0, 0)
w_w = np.insert(w_vals, 0, 0)
t_l = np.insert(t_vals, 0, 0)
w_l = np.insert(l_vals, 0, 0)
pchip_w = PchipInterpolator(t_w, w_w)
pchip_l = PchipInterpolator(t_l, w_l)
x_smooth = np.linspace(0, 60, 200)
w_smooth = pchip_w(x_smooth)
l_smooth = pchip_l(x_smooth)
plt.figure(figsize=(8,5))
plt.plot(
    x_smooth, w_smooth,
    label='Winner (ACAR)',
    color='green',
    linewidth=2,
    alpha=0.8
)
plt.plot(
    x_smooth, l_smooth,
    label='Loser (ACAR)',
    color='orange',
    linewidth=2,
    alpha=0.8
)
plt.plot(
    t_w, w_w,
    marker='^', markersize=5,
    linestyle='None',
    color='green',
    alpha=0.9
)
plt.plot(
    t_l, w_l,
    marker='^', markersize=5,
    linestyle='None',
    color='orange',
    alpha=0.9
)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xlim(left=0)
plt.xlabel('Month (t=1..60)')
plt.ylabel('Cumulative Excess Return')
plt.title('Winner vs Loser - Average CAR over Five-Year Periods')
plt.legend()
plt.tight_layout()
plt.show()
