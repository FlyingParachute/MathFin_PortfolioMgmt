# MathFin_PortfolioMgmt

下面介绍各文件的作用。

---

## config.py

**English:**
> The global configuration file that centralizes all adjustable parameters in the project.  
> This includes file paths (`DATA_PATH`), the minimum number of months required (`MIN_MONTHS`), the lookback window length (`LOOKBACK_WINDOW`), the backtest interval length (`BACKTEST_INTERVAL`), and the start/end dates (`START_DATE` / `END_DATE`) for the backtest.

**中文:**
> 全局配置文件，集中管理本项目中所有可调参数。  
> 包括数据文件路径（`DATA_PATH`）、最少月份限制（`MIN_MONTHS`）、向后计算累计收益的窗口（`LOOKBACK_WINDOW`）、回测区间长度（`BACKTEST_INTERVAL`）、以及回测的起止日期（`START_DATE`、`END_DATE`）等。

---

## main.py

**English:**
> The main entry point of the project. It orchestrates the entire workflow by sequentially calling the data loader, backtest, and analysis modules. Running this file automatically completes data processing, portfolio backtesting, and final visualization/statistical tests.

**中文:**
> 项目的主入口文件。依次调用数据加载、回测和分析模块，串联起整个工作流程。运行该文件即可完成数据预处理、组合回测以及最终的可视化与统计检验。

---

## data_loader.py

**English:**
> Responsible for reading and cleaning raw data from CSV files. This includes type conversions, removing outliers/NaN, and returning a neatly formatted DataFrame for subsequent steps.

**中文:**
> 负责从 CSV 文件中读取并清洗原始数据。包括数据类型转换、去除异常值/空值，最终返回整理好的 DataFrame 供后续使用。

---

## portfolio.py

**English:**
> Implements the core logic for portfolio construction and backtesting. It calculates rolling cumulative excess returns for each stock, selects winners/losers based on configurable criteria, and computes returns during holding periods. Ultimately, it produces a consolidated backtest result table.

**中文:**
> 组合构建与回测的核心逻辑文件。先为每只股票计算滚动累计超额收益，再根据配置选出赢家/输家组合并计算持有期收益。最终返回各个回测区间的合并数据表。

---

## analysis.py

**English:**
> Carries out statistical tests and visualization for the backtest results. It computes Average Cumulative Excess Returns (ACAR) for winners/losers, conducts T-tests (both single-sample and independent-sample), and plots smoothed curves (via PCHIP) to illustrate performance differences.

**中文:**
> 用于对回测结果进行统计分析与可视化。包括计算赢家/输家组合的平均累计超额收益（ACAR），执行单样本与双样本 T 检验，以及通过 PCHIP 插值平滑绘制两组表现曲线。

---

## utils/helpers.py

**English:**
> Contains auxiliary utility functions. For example, it can generate a list of portfolio formation dates at specified monthly intervals (`generate_portfolio_starts`) and select the top/bottom N (or percentage) of stocks (`select_top_bottom`).

**中文:**
> 存放一些通用的辅助函数，如根据指定月份间隔生成组合形成日期列表（`generate_portfolio_starts`），以及根据需要选取前后 N（或前后百分比）的股票（`select_top_bottom`）等。

---

## utils/excess_ret_comp.py

**English:**
> Defines functions related to excess return calculation. This includes subtracting market return from individual stock returns (`direct_substract`) and applying rolling windows to compute cumulative excess returns (`calculate_cumulative_excess_returns`).

**中文:**
> 定义与超额收益计算相关的函数。包括将个股收益减去市场收益（`direct_substract`），以及按滚动窗口计算股票的累计超额收益（`calculate_cumulative_excess_returns`）。  
