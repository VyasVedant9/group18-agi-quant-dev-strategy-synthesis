# Quantitative Trading Strategy Generator and Backtester

This project provides tools for generating and backtesting multiple trading strategies based on OHLCV (Open, High, Low, Close, Volume) data. It is designed to help quantitative traders develop and evaluate robust trading strategies with clear performance metrics.

## Features

- **Multiple Strategy Generation**: Implements several technical analysis-based trading strategies including:
  - Moving Average Crossover
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Volume-Based Breakout
  - MACD (Moving Average Convergence Divergence)

- **Comprehensive Backtesting**: Realistic backtesting with:
  - Transaction costs (commission and slippage)
  - Position sizing
  - Performance metrics calculation
  - Visualization tools

- **Strategy Optimization**: Advanced parameter optimization techniques:
  - Parameter grid search with proper train/test/validation splits
  - Walk-forward optimization to prevent overfitting
  - Performance visualization and comparison across datasets
  - Parallel processing for faster optimization

- **Performance Metrics**: Evaluates strategies based on:
  - Total and annualized returns
  - Maximum drawdown
  - Sharpe, Sortino, and Calmar ratios
  - Win rate and profit factor
  - Trade statistics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/group18-agi-quant-dev-strategy-synthesis.git
cd group18-agi-quant-dev-strategy-synthesis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Fetching Data

To download OHLCV data for a stock:

```bash
python fetch_data.py
```

This will download data for AAPL (default) and save it to a CSV file.

### Generating Strategies

To generate trading strategies based on the downloaded data:

```bash
python strategy_generator.py
```

This will:
1. Load the OHLCV data
2. Generate multiple trading strategies
3. Calculate performance metrics for each strategy
4. Display the results

### Backtesting Strategies

To backtest the generated strategies:

```bash
python backtester.py
```

This will:
1. Load the OHLCV data
2. Generate the strategies
3. Backtest each strategy with realistic assumptions
4. Compare the performance of all strategies
5. Display detailed metrics for the best-performing strategy
6. Generate performance charts

### Optimizing Strategy Parameters

To optimize strategy parameters while avoiding overfitting:

```bash
python run_optimizer.py --strategy ma
```

This will:
1. Load the OHLCV data
2. Split the data into train/test/validation sets
3. Perform grid search to find optimal parameters on training data
4. Evaluate the best parameters on test data
5. Validate on out-of-sample data
6. Generate visualizations and performance metrics

Available optimization options:
- `--strategy ma`: Optimize Moving Average Crossover strategy
- `--strategy rsi`: Optimize RSI strategy
- `--strategy wfo`: Perform Walk-Forward Optimization
- `--strategy all`: Run all optimization methods

For more options:
```bash
python run_optimizer.py --help
```

## Customization

### Modifying Strategies

You can customize the strategy parameters in `strategy_generator.py`. Each strategy function accepts parameters to tune the strategy behavior.

For example, to change the RSI strategy parameters:

```python
# In strategy_generator.py
rsi_strategy = generate_rsi_strategy(period=10, overbought=75, oversold=25)
```

### Adjusting Backtesting Parameters

You can customize backtesting parameters in `backtester.py`:

```python
# In backtester.py
backtester = Backtester(
    df,
    initial_capital=50000.0,  # Starting capital
    commission=0.0005,        # Commission per trade (0.05%)
    slippage=0.0002           # Slippage per trade (0.02%)
)
```

### Configuring Strategy Optimization

You can customize the strategy optimization in your code:

```python
from strategy_optimizer import StrategyOptimizer
from strategy_generator import load_data

# Load data
df = load_data("ohlcv_AAPL.csv")

# Create optimizer with custom data splits
optimizer = StrategyOptimizer(
    df, 
    train_size=0.6,      # 60% of data for training
    test_size=0.2,       # 20% of data for testing
    validation_size=0.2  # 20% of data for validation
)

# Run optimization with custom parameters
results = optimizer.optimize_ma_crossover(
    fast_period_range=(5, 50, 5),   # min, max, step
    slow_period_range=(20, 200, 20),
    metric='sharpe_ratio',
    n_jobs=4  # Use 4 cores for parallel processing
)
```

## Example Output

The backtester provides comprehensive performance metrics:

```
===== Strategy Performance Comparison =====

                  ma_crossover        rsi bollinger_bands volume_breakout       macd
total_return         0.185402  0.103528        0.076291        0.092857   0.142390
annual_return        0.184530  0.103054        0.075943        0.092429   0.141696
max_drawdown        -0.118721 -0.132547       -0.145382       -0.127943  -0.103924
sharpe_ratio         1.342857  0.894376        0.723419        0.812953   1.123876
sortino_ratio        2.102938  1.398753        1.132864        1.273425   1.761392
calmar_ratio         1.554321  0.777496        0.522371        0.722412   1.363474
...
```

The optimizer provides detailed parameter optimization results:

```
===== Optimizing MA Crossover Strategy =====
Data split - Train: 152 rows, Test: 50 rows, Validation: 51 rows
Optimizing MA Crossover with 40 parameter combinations
Evaluated 10/40 parameter combinations
Evaluated 20/40 parameter combinations
Evaluated 30/40 parameter combinations
Evaluated 40/40 parameter combinations

Best MA Crossover Parameters: {'fast_period': 10, 'slow_period': 100}
Train Sharpe Ratio: 1.35
Test Sharpe Ratio: 0.95

===== Strategy Validation Results =====
Strategy Parameters: {'fast_period': 10, 'slow_period': 100}
Validation Sharpe Ratio: 1.21
Validation Total Return: 12.45%
Validation Annual Return: 15.87%
Validation Max Drawdown: -8.92%
Validation Sortino Ratio: 1.89
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- Group 18 Team Members

## Acknowledgments

- Data provided by Yahoo Finance via yfinance library