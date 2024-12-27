import pandas as pd

def generate_portfolio_starts(start_date, end_date, interval_months):
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += pd.DateOffset(months=interval_months)
    return dates

def select_top_bottom(df, col, top_n, bottom_n):
    df = df.dropna(subset=[col]).sort_values(col, ascending=False)
    total_count = len(df)

    if isinstance(top_n, float) and 0 < top_n < 1:
        top_count = int(round(total_count * top_n))
    elif isinstance(top_n, int) and top_n > 0:
        top_count = min(top_n, total_count)
    else:
        top_count = 0

    if isinstance(bottom_n, float) and 0 < bottom_n < 1:
        bottom_count = int(round(total_count * bottom_n))
    elif isinstance(bottom_n, int) and bottom_n > 0:
        bottom_count = min(bottom_n, total_count)
    else:
        bottom_count = 0

    w_ids = df.head(top_count)['PERMNO']
    l_ids = df.tail(bottom_count)['PERMNO']
    return w_ids, l_ids
