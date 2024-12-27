import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp
from scipy.interpolate import PchipInterpolator

def analyze_results(all_periods_df):
    all_periods_df = all_periods_df.groupby('test_period_start', group_keys=False)\
                                   .apply(_add_t)
    acar = all_periods_df.groupby('t', as_index=False).agg({'CAR_w':'mean','CAR_l':'mean'})
    acar.rename(columns={'CAR_w':'ACAR_w','CAR_l':'ACAR_l'}, inplace=True)
    acar['diff'] = acar['ACAR_l'] - acar['ACAR_w']

    grouped_n = all_periods_df.groupby(['t','test_period_start'], as_index=False).agg({
        'CAR_w':'last',
        'CAR_l':'last'
    })
    ttest_list = []
    for t_ in sorted(grouped_n['t'].unique()):
        sub = grouped_n[grouped_n['t'] == t_]
        w_vals = sub['CAR_w'].dropna()
        l_vals = sub['CAR_l'].dropna()
        if len(w_vals) < 2 or len(l_vals) < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = ttest_ind(l_vals, w_vals, nan_policy='omit')
        ttest_list.append({'t': t_, 'N_w': len(w_vals), 'N_l': len(l_vals), 't_stat': t_stat, 'p_value': p_val})
    ttest_df = pd.DataFrame(ttest_list)

    single_ttest_list = []
    for t_ in sorted(all_periods_df['t'].dropna().unique()):
        sub = all_periods_df[all_periods_df['t'] == t_]
        w_vals = sub['avg_u_w'].dropna()
        l_vals = sub['avg_u_l'].dropna()
        if len(w_vals) > 1:
            w_t_stat, w_p_val = ttest_1samp(w_vals, popmean=0, nan_policy='omit')
        else:
            w_t_stat, w_p_val = np.nan, np.nan
        if len(l_vals) > 1:
            l_t_stat, l_p_val = ttest_1samp(l_vals, popmean=0, nan_policy='omit')
        else:
            l_t_stat, l_p_val = np.nan, np.nan
        single_ttest_list.append({
            't': t_,
            'w_t_stat': w_t_stat,
            'w_p_value': w_p_val,
            'l_t_stat': l_t_stat,
            'l_p_value': l_p_val,
        })
    single_ttest_df = pd.DataFrame(single_ttest_list)

    print("====== ACAR (平均累计收益) ======")
    print(acar[['t','ACAR_w','ACAR_l','diff']])
    print("\n====== (ACAR_l - ACAR_w) T检验(简易) ======")
    print(ttest_df)
    print("\n====== 单月平均残差回报的单样本 T检验 ======")
    print(single_ttest_df)

    t_vals = acar['t'].values
    w_vals = acar['ACAR_w'].values
    l_vals = acar['ACAR_l'].values

    t_w = np.insert(t_vals, 0, 0)
    w_w = np.insert(w_vals, 0, 0)
    t_l = np.insert(t_vals, 0, 0)
    w_l = np.insert(l_vals, 0, 0)

    pchip_w = PchipInterpolator(t_w, w_w)
    pchip_l = PchipInterpolator(t_l, w_l)
    x_smooth = np.linspace(0, max(t_vals) if len(t_vals)>0 else 1, 200)
    w_smooth = pchip_w(x_smooth)
    l_smooth = pchip_l(x_smooth)

    plt.figure(figsize=(8,5))
    plt.plot(x_smooth, w_smooth, label='Winner (ACAR)', color='green', linewidth=2, alpha=0.8)
    plt.plot(x_smooth, l_smooth, label='Loser (ACAR)', color='orange', linewidth=2, alpha=0.8)
    plt.plot(t_w, w_w, marker='^', markersize=5, linestyle='None', color='green', alpha=0.9)
    plt.plot(t_l, w_l, marker='^', markersize=5, linestyle='None', color='orange', alpha=0.9)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xlim(left=0)  # 让(0,0)紧贴纵轴
    plt.xlabel('Month (t)')
    plt.ylabel('Cumulative Excess Return')
    plt.title('Winner vs Loser')
    plt.legend()
    plt.tight_layout()
    plt.show()

def _add_t(df_):
    df_ = df_.sort_values('date').reset_index(drop=True)
    df_['t'] = np.arange(len(df_)) + 1
    return df_
