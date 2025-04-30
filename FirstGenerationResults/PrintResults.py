# This file contains the terminal statements on executing base_tradingstrategy.py.
"""Loading data from ./data/ohlcv_AAPL.csv...

Preparing data...

Training model...
Fitting 5 folds for each of 81 candidates, totalling 405 fits
Best parameters: {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.8}
Best CV score: 0.0009

Model Evaluation:
MSE: 0.001559
MAE: 0.037138

Generating 1-week forecast...

1-week price forecast:
Day 1: $183.87
Day 2: $174.73
Day 3: $170.41
Day 4: $166.16
Day 5: $162.48"""


# ---------------------------------------------------------------------------------------------------------
# BACKTESTING
"""Data Validation:
First date: 2023-03-15 00:00:00-04:00
Last date: 2023-12-29 00:00:00-05:00
Rows: 201
Recent Close Prices:
Date
2023-12-22 00:00:00-05:00    192.444580
2023-12-26 00:00:00-05:00    191.897858
2023-12-27 00:00:00-05:00    191.997284
2023-12-28 00:00:00-05:00    192.424713
2023-12-29 00:00:00-05:00    191.380966
Name: Close, dtype: float64

Running MA Crossover Strategy (5,20) with debug:

MA Crossover Signals Generated:
Buy signals: 3
Sell signals: 4

Running backtest with trade logging:

Running backtest...
Entered long position at 2023-08-29 00:00:00-04:00: 50 shares @ $182.96
Exited position at 2023-09-12 00:00:00-04:00: 50 shares @ $174.84 (signal)
Entered long position at 2023-10-09 00:00:00-04:00: 50 shares @ $177.87
Exited position at 2023-10-23 00:00:00-04:00: 50 shares @ $171.57 (signal)
Entered long position at 2023-11-06 00:00:00-05:00: 50 shares @ $178.10
Exited position at 2023-12-13 00:00:00-05:00: 50 shares @ $196.58 (take_profit)
Backtest completed successfully

=== Backtest Performance Report ===
Initial Capital: $10,000.00
Final Equity: $10,197.06
Total Return: 1.97%
CAGR: 2.50%
Sharpe Ratio: 0.34
Max Drawdown: -9.76%

Trade Statistics:
Number of Trades: 3
Win Rate: 33.33%
Avg Trade Return: 0.80%
Avg Trade Duration: 21.7 days

Trade Details:
                  entry_date  entry_price                  exit_date  exit_price  return_pct  duration
0  2023-08-29 00:00:00-04:00   182.963130  2023-09-12 00:00:00-04:00  174.842240   -0.044385        14
1  2023-10-09 00:00:00-04:00   177.865371  2023-10-23 00:00:00-04:00  171.569515   -0.035397        14
2  2023-11-06 00:00:00-05:00   178.103860  2023-12-13 00:00:00-05:00  196.581786    0.103748        37"""