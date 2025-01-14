# Portfolio Management Course Work

This project is the course work for the Portfolio Management course, completed by Xingjian Zhao, Jingtong Xu, Yi Qu.


The dataset is too large to upload on github, please get it through https://drive.google.com/drive/folders/15H3p0d_dlO1QloEIe5VKKIU5yvyoh3F0?usp=drive_link

## config.py

> The global configuration file that centralizes all adjustable parameters in the project.  
> This includes file paths (`DATA_PATH`), the minimum number of months required (`MIN_MONTHS`), the lookback window length (`LOOKBACK_WINDOW`), the backtest interval length (`BACKTEST_INTERVAL`), and the start/end dates (`START_DATE` / `END_DATE`) for the backtest.

---

## main.py

> The main entry point of the project. It orchestrates the entire workflow by sequentially calling the data loader, backtest, and analysis modules. Running this file automatically completes data processing, portfolio backtesting, and final visualization/statistical tests.

---

## project_main.py

> An alternative main file that can also be run directly. Unlike `main.py`, it includes functionality to directly compute excess returns using the CAPM method.

---

## data_loader.py

> Responsible for reading and cleaning raw data from CSV files. This includes type conversions, removing outliers/NaN, and returning a neatly formatted DataFrame for subsequent steps.

---

## portfolio.py

> Implements the core logic for portfolio construction and backtesting. It calculates rolling cumulative excess returns for each stock, selects winners/losers based on configurable criteria, and computes returns during holding periods. Ultimately, it produces a consolidated backtest result table.

---

## analysis.py

> Carries out statistical tests and visualization for the backtest results. It computes Average Cumulative Excess Returns (ACAR) for winners/losers, conducts T-tests (both single-sample and independent-sample), and plots smoothed curves (via PCHIP) to illustrate performance differences.

---

## Fig3Plotting.py

> Responsible for plotting Figure 3 of the paper, which visualizes the backtest results for every January case.

---

## notebook/T_test.ipynb

> A Jupyter notebook dedicated to performing statistical tests, including T-tests, on the backtest results.

---

## utils/helpers.py

> Contains auxiliary utility functions. For example, it can generate a list of portfolio formation dates at specified monthly intervals (`generate_portfolio_starts`) and select the top/bottom N (or percentage) of stocks (`select_top_bottom`).

---

## utils/excess_ret_comp.py

> Defines functions related to excess return calculation. This includes subtracting market return from individual stock returns (`direct_substract`) and applying rolling windows to compute cumulative excess returns (`calculate_cumulative_excess_returns`).
