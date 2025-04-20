# Strategy Optimization System

This module provides tools for optimizing trading strategy parameters while preventing overfitting through proper train/test/validation splits and robust optimization techniques.

## Features

- **Parameter Grid Search**: Systematically explore parameter combinations to find optimal settings
- **Train/Test/Validation Splits**: Proper data separation to prevent overfitting
- **Walk-Forward Optimization**: Time-based cross-validation for robust parameter selection
- **Performance Visualization**: Detailed visualization of optimization results and strategy performance
- **Parallel Processing**: Support for multi-core optimization to speed up parameter search

## Getting Started

### Installation

Make sure you have all the required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

The optimization system builds on the existing backtester and strategy generator modules. To run a basic optimization:

```bash
python run_optimizer.py --data path/to/ohlcv_data.csv --strategy ma --output results
```

### Available Optimization Methods

The system provides several optimization approaches:

1. **Standard Parameter Grid Search** (`--strategy ma` or `--strategy rsi`)
   - Systematically tests combinations of parameters on training data
   - Evaluates best parameters on test data
   - Validates on out-of-sample validation data

2. **Walk-Forward Optimization** (`--strategy wfo`)
   - Uses sliding windows of data to simulate real-world strategy deployment
   - Finds optimal parameters for each time window
   - Tests parameter stability over time

3. **Run All Optimizations** (`--strategy all`)
   - Runs all optimization methods with default settings

### Command Line Options

```
usage: run_optimizer.py [-h] [--data DATA] [--output OUTPUT] 
                        [--strategy {ma,rsi,wfo,all}] [--jobs JOBS]

Optimize trading strategies for OHLCV data.

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Path to OHLCV data CSV file
  --output OUTPUT       Directory to save optimization results
  --strategy {ma,rsi,wfo,all}
                        Strategy to optimize
  --jobs JOBS           Number of parallel jobs for optimization
```

## Implementation Details

### Data Splitting

The optimizer splits data chronologically to prevent look-ahead bias:

- **Training Set**: Used to find optimal parameters (default: 60% of data)
- **Test Set**: Used to evaluate parameters (default: 20% of data)
- **Validation Set**: Used for final validation (default: 20% of data)

### Performance Metrics

The system calculates multiple performance metrics to evaluate strategies:

- **Sharpe Ratio**: Risk-adjusted return
- **Total Return**: Absolute return over the period
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sortino Ratio**: Downside risk-adjusted return
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses

### Custom Strategy Optimization

To optimize a custom strategy, use the `StrategyOptimizer` class directly in your code:

```python
from strategy_optimizer import StrategyOptimizer
from strategy_generator import load_data

# Load data
data = load_data("ohlcv_data.csv")

# Create optimizer
optimizer = StrategyOptimizer(data)

# Define custom strategy function
def my_strategy_func(sg, param1, param2):
    # Implementation of your strategy
    return strategy_signals

# Define parameter grid
param_grid = [
    {'param1': value1, 'param2': value2}
    # More parameter combinations...
]

# Run walk-forward optimization
results = optimizer.walk_forward_optimization(
    strategy_func=my_strategy_func,
    param_grid=param_grid,
    window_size=60,
    step_size=20
)
```

## Example Results

The optimization process generates several visualization files:

1. **Parameter Heatmaps**: Show the relationship between parameters and performance metrics
2. **Equity Curves**: Compare strategy performance across train/test/validation sets
3. **Parameter Stability Plots**: Show how optimal parameters change over time (for walk-forward optimization)
4. **Performance Metrics**: Detailed CSV files with all parameter combinations and their performance

## Avoiding Overfitting

This system implements several techniques to help prevent overfitting:

1. **Proper Data Separation**: Strict separation of training, testing, and validation data
2. **Walk-Forward Testing**: Time-based validation that better represents real trading conditions
3. **Parameter Stability Analysis**: Tools to evaluate if parameters are stable over time
4. **Multiple Performance Metrics**: Considering various aspects of performance beyond just returns

## Common Issues and Solutions

- **Excessive Runtime**: Reduce parameter grid size or increase step sizes
- **Unstable Parameters**: Consider different evaluation metrics or more stable strategies
- **Poor Validation Performance**: May indicate overfitting or regime changes in the data

## Contributing

Contributions to improve the optimization system are welcome. Possible areas for enhancement:

- Additional optimization algorithms (genetic algorithms, Bayesian optimization)
- More strategy types and parameter combinations
- Enhanced visualization tools
- Optimization for additional performance metrics 