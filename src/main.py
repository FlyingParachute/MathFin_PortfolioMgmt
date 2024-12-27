from config import START_DATE, END_DATE
from data_loader import load_and_clean_data
from portfolio import backtest
from analysis import analyze_results

def main():
    data = load_and_clean_data()
    all_periods_df = backtest(data, START_DATE, END_DATE)
    analyze_results(all_periods_df)

if __name__ == '__main__':
    main()
