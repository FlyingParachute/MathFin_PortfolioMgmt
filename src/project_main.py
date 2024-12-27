import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
import matplotlib.pyplot as plt
import warnings
from excess_ret_comp import direct_substract, calculate_cumulative_excess_returns
from scipy.interpolate import PchipInterpolator
warnings.filterwarnings("ignore")


# =====================================================================
# ===================== 第 1 节：读取 & 清洗数据 ======================
# =====================================================================

data_path = '.\data\monthly\monthly.csv'
method = 'prod'  # 'sum' or 'prod'
monthly_raw_data = pd.read_csv(data_path)

monthly_raw_data = direct_substract(monthly_raw_data)

# 去除 NaN以及RETX为'B','C'的数据
cleaned_data = monthly_raw_data.dropna(subset=['RETX'])
cleaned_data = cleaned_data[~cleaned_data['RETX'].isin(['B','C'])]

# 转换日期
cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])

# 保证每支股票 >= 85 个月的数据
stocks_with_enough_data = cleaned_data.groupby('PERMNO').filter(lambda x: len(x) >= 85)

# 按 (PERMNO, date) 排序
stocks_with_enough_data = stocks_with_enough_data.sort_values(['PERMNO','date']).reset_index(drop=True)
print(f"符合条件的股票数量: {stocks_with_enough_data['PERMNO'].nunique()}")


# =====================================================================
# = 第 2 节：计算过去36个月的滚动“复利”累积超额收益 (cum_excess_return) =
# =====================================================================



stocks_with_enough_data = (
    stocks_with_enough_data
    .groupby('PERMNO', group_keys=False)
    .apply(calculate_cumulative_excess_returns, methods=method) # 用 'sum' 也可以
)


# =====================================================================
# === 第 3 节：人工指定 16 个三年期起点 (1930/01 ~ 1975/01，每三年一次) ===
# =====================================================================

portfolio_starts = [
    pd.Timestamp('1930-01-01'),
    pd.Timestamp('1933-01-01'),
    pd.Timestamp('1936-01-01'),
    pd.Timestamp('1939-01-01'),
    pd.Timestamp('1942-01-01'),
    pd.Timestamp('1945-01-01'),
    pd.Timestamp('1948-01-01'),
    pd.Timestamp('1951-01-01'),
    pd.Timestamp('1954-01-01'),
    pd.Timestamp('1957-01-01'),
    pd.Timestamp('1960-01-01'),
    pd.Timestamp('1963-01-01'),
    pd.Timestamp('1966-01-01'),
    pd.Timestamp('1969-01-01'),
    pd.Timestamp('1972-01-01'),
    pd.Timestamp('1975-01-01'),
]


# =====================================================================
# = 第 4 节：对每个三年期进行回测：组合形成 + 测试期内逐月收益 =
# =====================================================================

all_periods = []  # 用于保存每个测试期(n=1..16)的逐月数据

for start_dt in portfolio_starts:
    # 4.1 组合形成日 = 起点前1个月末
    formation_date = start_dt - pd.offsets.MonthEnd(1)
    
    # lookback: 过去36个月区间 [formation_date-35M, formation_date]
    lookback_end = formation_date
    lookback_start = formation_date - pd.DateOffset(months=35)
    print(f"\n组合形成日: {formation_date.date()}，lookback区间: {lookback_start.date()} - {lookback_end.date()}")

    # 提取 lookback 区间的数据
    lookback_df = stocks_with_enough_data[
        (stocks_with_enough_data['date'] >= lookback_start) &
        (stocks_with_enough_data['date'] <= lookback_end)
    ].copy()

    if lookback_df.empty:
        # 没数据就跳过
        continue
    
    # 对每只股票取 <= formation_date 最新一条记录
    sub = lookback_df[lookback_df['date'] <= formation_date]
    # 每只股票只保留最后一条(含 cum_excess_return)
    portfolio_data = sub.groupby('PERMNO', group_keys=False).tail(1).copy()
    
    # 排序，选最高35(赢家), 最低35(输家)
    portfolio_data = portfolio_data.sort_values('cum_excess_return', ascending=False)
    portfolio_data = portfolio_data.dropna(subset=['cum_excess_return'])
    print(f"可选股票数量: {portfolio_data['PERMNO'].nunique()}")
    winner_ids = portfolio_data.head(35)['PERMNO']
    loser_ids  = portfolio_data.tail(35)['PERMNO']
    
    # -------------- 日志功能：打印调仓信息 -------------
    print("[调仓日期: {}]".format(formation_date.date()))
    print("  Winner组合股票数:", len(winner_ids))
    print("  Loser组合股票数:", len(loser_ids))
    # 如果还想查看具体股票ID，可直接打印 winner_ids.values 等
    # print("  Winner IDs:", winner_ids.values)
    # print("  Loser IDs:", loser_ids.values)
    # -----------------------------------------------
    
    # 4.2 三年持有期 = [start_dt, start_dt+36个月 - 1天]
    hold_start = start_dt
    hold_end   = start_dt + pd.DateOffset(months=36) - pd.Timedelta(days=1)
    print(f"持有期区间: {hold_start.date()} - {hold_end.date()}")
    
    # 赢家/输家组合 在持有期内的数据
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
    
    # 若两组都无数据，也要存个空表做记录
    if w_hold.empty or l_hold.empty:
        all_periods.append(pd.DataFrame({
            'test_period_start': [start_dt],
            'date': [None],
            'avg_u_w': [np.nan],
            'avg_u_l': [np.nan],
        }))
        continue

    # 4.3 每月组合收益
    
    # 赢家组合：按月份聚合求均值
    w_monthly = w_hold.groupby('date', as_index=False)['RETX'].mean()
    w_monthly.rename(columns={'RETX':'avg_u_w'}, inplace=True)

    # 输家组合：按月份聚合求均值
    l_monthly = l_hold.groupby('date', as_index=False)['RETX'].mean()
    l_monthly.rename(columns={'RETX':'avg_u_l'}, inplace=True)

    # 生成完整的月末区间
    all_months = pd.date_range(hold_start, hold_end, freq='M')
    hold_months_df = pd.DataFrame({'date':all_months})
    
    # 合并赢家/输家
    merged = (
        hold_months_df
        .merge(w_monthly, on='date', how='left')
        .merge(l_monthly, on='date', how='left')
    )
    
    # 计算累计收益 (CAR_w, CAR_l)
    
    if method == 'sum':
        merged['CAR_w'] = merged['avg_u_w'].cumsum(skipna=True)
        merged['CAR_l'] = merged['avg_u_l'].cumsum(skipna=True)
    elif method == 'prod':
        merged['CAR_w'] = merged['avg_u_w'].add(1).cumprod().sub(1)
        merged['CAR_l'] = merged['avg_u_l'].add(1).cumprod().sub(1)

    # 标注测试期起点
    merged['test_period_start'] = start_dt
    
    all_periods.append(merged)


# 合并所有测试期
all_periods_df = pd.concat(all_periods, ignore_index=True)
# 排序
all_periods_df.sort_values(['test_period_start','date'], inplace=True)


# =====================================================================
# = 第 5 节：对齐 t=1..36，计算 ACAR_w(t) & ACAR_l(t) & diff =
# =====================================================================

def add_relative_month(df_):
    """
    为单个测试期添加相对月编号 t=1..36。
    """
    df_ = df_.sort_values('date').reset_index(drop=True)
    df_['t'] = np.arange(len(df_)) + 1
    return df_

all_periods_df = all_periods_df.groupby('test_period_start', group_keys=False).apply(add_relative_month)

# 对同一个 t 跨测试期求均值 => ACAR_w, ACAR_l
acar = all_periods_df.groupby('t', as_index=False).agg({
    'CAR_w':'mean',
    'CAR_l':'mean'
})
acar.rename(columns={'CAR_w':'ACAR_w','CAR_l':'ACAR_l'}, inplace=True)
acar['diff'] = acar['ACAR_l'] - acar['ACAR_w']


# =====================================================================
# === 第 6 节：(ACAR_l - ACAR_w) 的统计检验 (独立样本 T检验) ===
# =====================================================================

# 先拿到：每个 (t, test_period_start) 的 CAR_w, CAR_l
grouped_n = all_periods_df.groupby(['t','test_period_start'], as_index=False).agg({
    'CAR_w':'last',
    'CAR_l':'last'
})
# t=1..36, test_period_start 最多16组

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


# =====================================================================
# == 第 6.1 节：单月平均残差回报的单样本 t 检验（对 0 的显著性） ==
# =====================================================================

single_ttest_list = []
unique_t2 = sorted(all_periods_df['t'].unique())

for t_ in unique_t2:
    # 提取该月的全部记录
    sub = all_periods_df[all_periods_df['t'] == t_]
    
    # 赢家组合本月平均残差
    w_vals = sub['avg_u_w'].dropna()
    # 输家组合本月平均残差
    l_vals = sub['avg_u_l'].dropna()
    
    # 对赢家组合做单样本 T 检验 (H0: 均值=0)
    if len(w_vals) > 1:
        w_t_stat, w_p_val = ttest_1samp(w_vals, popmean=0, nan_policy='omit')
    else:
        w_t_stat, w_p_val = np.nan, np.nan

    # 对输家组合做单样本 T 检验 (H0: 均值=0)
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


# =====================================================================
# == 第 7 节：结果输出与绘图 ==
# =====================================================================

print("====== ACAR (平均累计收益) ======")
print(acar[['t','ACAR_w','ACAR_l','diff']])  # 只展示主要列

print("\n====== (ACAR_l - ACAR_w) T检验(简易) ======")
print(ttest_df)

print("\n====== 单月平均残差回报的单样本 T检验 ======")
print(single_ttest_df)


# ---------- 绘制 Winner vs Loser 的平均累计超额收益曲线 (基于 ACAR) ----------
'''
plt.figure(figsize=(8,5))
plt.plot(acar['t'], acar['ACAR_w'], label='Winner (ACAR)', color='blue')
plt.plot(acar['t'], acar['ACAR_l'], label='Loser (ACAR)', color='red')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Month (t=1..36)')
plt.ylabel('Cumulative Excess Return')
plt.title('Winner vs Loser - Average CAR over 36 months')
plt.legend()
plt.tight_layout()
plt.show()
'''
#============ 1) 数据处理，让曲线从 (0,0) 出发 ============
t_vals = acar['t'].values            # [1, 2, 3, ..., 36]
w_vals = acar['ACAR_w'].values       # winner y值
l_vals = acar['ACAR_l'].values       # loser y值

# 在头部插入 (0,0)，让曲线从原点开始
t_w = np.insert(t_vals, 0, 0)
w_w = np.insert(w_vals, 0, 0)
t_l = np.insert(t_vals, 0, 0)
w_l = np.insert(l_vals, 0, 0)

#============ 2) 使用 PchipInterpolator 构造插值函数 ============
# 它会保证插值后的曲线恰好通过每个给定点
pchip_w = PchipInterpolator(t_w, w_w)
pchip_l = PchipInterpolator(t_l, w_l)

#============ 3) 生成较密集的 x 值以绘制平滑曲线 ============
# 这里以 0~36 为区间，200个点可视需要修改
x_smooth = np.linspace(0, 36, 200)

# 在这些 x_smooth 上分别计算 Winner/Loser 的插值
w_smooth = pchip_w(x_smooth)
l_smooth = pchip_l(x_smooth)

#============ 4) 开始绘图 ============
plt.figure(figsize=(8,5))

# (a) 平滑插值曲线
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

# (b) 在每个整数 t 点上用三角标记，保证肉眼可见曲线“确实穿过这些点”
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

# (c) 参考线、边界等
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xlim(left=0)  # 让(0,0)紧贴纵轴
plt.xlabel('Month (t=1..36)')
plt.ylabel('Cumulative Excess Return')
plt.title('Winner vs Loser - Average CAR over 36 months')
plt.legend()
plt.tight_layout()
plt.show()
