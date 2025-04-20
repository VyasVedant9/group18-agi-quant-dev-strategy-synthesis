"""
Prompt: 
I'd like you to implement strategy and parameter optimization to better fit trading strategies to particular assets
Use the results of the backtester.py to guide your strategy optimization and ensure you are using appropriate train/test splits so there is reduced risk of overfit 

Strategy Optimizer for Trading Strategies

This module provides optimization functionality for trading strategies generated
by the strategy_generator module. It helps find optimal parameters for strategies
while avoiding overfitting through proper train/test splits and validation techniques.

Features:
- Parameter grid search and optimization
- Walk-forward testing
- Train/test/validation data splitting
- Hyperparameter optimization
- Cross-validation techniques for robustness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from itertools import product
from functools import partial
import datetime as dt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from sklearn.model_selection import TimeSeriesSplit

from strategy_generator import StrategyGenerator, load_data
from backtester import Backtester


class StrategyOptimizer:
    """A class for optimizing trading strategy parameters."""
    
    def __init__(self, data: pd.DataFrame, train_size: float = 0.6, 
                 test_size: float = 0.2, validation_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the StrategyOptimizer with OHLCV data and data split parameters.
        
        Args:
            data: DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'
            train_size: Proportion of data to use for training (parameter optimization)
            test_size: Proportion of data to use for testing (strategy evaluation)
            validation_size: Proportion of data to use for final validation
            random_state: Random seed for reproducibility
        """
        StrategyGenerator.validate_data(data)
        self.data = data.copy()
        self.train_size = train_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        # Split data into train, test, validation sets
        self._split_data()
        
        # Create strategy generator and backtester instances
        self.strategy_generator = StrategyGenerator(self.data)
        self.train_backtester = Backtester(self.train_data)
        self.test_backtester = Backtester(self.test_data)
        self.validation_backtester = Backtester(self.validation_data)
        
    def _split_data(self) -> None:
        """Split data into train, test, and validation sets chronologically."""
        total_rows = len(self.data)
        train_end = int(total_rows * self.train_size)
        test_end = train_end + int(total_rows * self.test_size)
        
        self.train_data = self.data.iloc[:train_end].copy()
        self.test_data = self.data.iloc[train_end:test_end].copy()
        self.validation_data = self.data.iloc[test_end:].copy()
        
        print(f"Data split - Train: {len(self.train_data)} rows, Test: {len(self.test_data)} rows, "
              f"Validation: {len(self.validation_data)} rows")
        
    def optimize_ma_crossover(self, fast_period_range: Tuple[int, int, int] = (5, 50, 5),
                             slow_period_range: Tuple[int, int, int] = (20, 200, 10),
                             metric: str = 'sharpe_ratio',
                             n_jobs: int = 1) -> Dict:
        """
        Optimize Moving Average Crossover strategy parameters.
        
        Args:
            fast_period_range: Tuple of (min, max, step) for fast period
            slow_period_range: Tuple of (min, max, step) for slow period
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            n_jobs: Number of parallel jobs (1 for sequential)
            
        Returns:
            Dictionary with optimization results
        """
        # Generate parameter grid
        fast_periods = range(fast_period_range[0], fast_period_range[1] + 1, fast_period_range[2])
        slow_periods = range(slow_period_range[0], slow_period_range[1] + 1, slow_period_range[2])
        
        param_grid = []
        for fast_period in fast_periods:
            for slow_period in slow_periods:
                if fast_period < slow_period:  # Ensure fast period is less than slow period
                    param_grid.append({'fast_period': fast_period, 'slow_period': slow_period})
        
        print(f"Optimizing MA Crossover with {len(param_grid)} parameter combinations")
        
        # Create a partial function for parameter evaluation
        evaluate_func = partial(self._evaluate_ma_crossover_params, metric=metric)
        
        # Evaluate parameters
        results = self._parallel_evaluate(evaluate_func, param_grid, n_jobs)
        
        # Sort results by metric value (descending)
        sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
        
        # Get best parameters and evaluate on test set
        best_params = sorted_results[0]['params']
        test_performance = self._evaluate_ma_crossover_test(best_params)
        
        return {
            'best_params': best_params,
            'train_performance': sorted_results[0],
            'test_performance': test_performance,
            'all_results': sorted_results
        }
    
    def _evaluate_ma_crossover_params(self, params: Dict, metric: str) -> Dict:
        """
        Evaluate a set of MA Crossover parameters on the training data.
        
        Args:
            params: Dictionary with parameters
            metric: Metric to optimize
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Generate strategy with given parameters
            sg = StrategyGenerator(self.train_data)
            strategy = sg.generate_ma_crossover_strategy(
                fast_period=params['fast_period'],
                slow_period=params['slow_period']
            )
            
            # Backtest the strategy
            results = self.train_backtester.backtest_strategy(strategy['signals'])
            metrics = self.train_backtester.calculate_metrics(results)
            
            # Add parameters to results for reference
            return {
                'params': params,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_return': metrics['total_return'],
                'annual_return': metrics['annual_return'],
                'max_drawdown': metrics['max_drawdown'],
                'sortino_ratio': metrics['sortino_ratio'],
                'calmar_ratio': metrics['calmar_ratio'],
                'num_trades': metrics['num_trades'],
                'win_rate': metrics['win_rate']
            }
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return {
                'params': params,
                'sharpe_ratio': -np.inf,
                'total_return': -np.inf,
                'annual_return': -np.inf,
                'max_drawdown': -1.0,
                'sortino_ratio': -np.inf,
                'calmar_ratio': -np.inf,
                'num_trades': 0,
                'win_rate': 0
            }
    
    def _evaluate_ma_crossover_test(self, params: Dict) -> Dict:
        """
        Evaluate MA Crossover parameters on the test data.
        
        Args:
            params: Dictionary with parameters
            
        Returns:
            Dictionary with test evaluation results
        """
        # Generate strategy with given parameters on test data
        sg = StrategyGenerator(self.test_data)
        strategy = sg.generate_ma_crossover_strategy(
            fast_period=params['fast_period'],
            slow_period=params['slow_period']
        )
        
        # Backtest the strategy
        results = self.test_backtester.backtest_strategy(strategy['signals'])
        metrics = self.test_backtester.calculate_metrics(results)
        
        return {
            'params': params,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_return': metrics['total_return'],
            'annual_return': metrics['annual_return'],
            'max_drawdown': metrics['max_drawdown'],
            'sortino_ratio': metrics['sortino_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate']
        }
    
    def _parallel_evaluate(self, evaluate_func: Callable, param_grid: List[Dict], n_jobs: int) -> List[Dict]:
        """
        Evaluate parameters in parallel.
        
        Args:
            evaluate_func: Function to evaluate parameters
            param_grid: List of parameter dictionaries
            n_jobs: Number of parallel jobs
            
        Returns:
            List of evaluation results
        """
        results = []
        
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(evaluate_func, params) for params in param_grid]
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    if (i + 1) % 10 == 0:
                        print(f"Evaluated {i + 1}/{len(param_grid)} parameter combinations")
        else:
            for i, params in enumerate(param_grid):
                result = evaluate_func(params)
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"Evaluated {i + 1}/{len(param_grid)} parameter combinations")
        
        return results
    
    def optimize_rsi_strategy(self, period_range: Tuple[int, int, int] = (5, 30, 1),
                             overbought_range: Tuple[int, int, int] = (60, 85, 5),
                             oversold_range: Tuple[int, int, int] = (15, 40, 5),
                             metric: str = 'sharpe_ratio',
                             n_jobs: int = 1) -> Dict:
        """
        Optimize RSI strategy parameters.
        
        Args:
            period_range: Tuple of (min, max, step) for RSI period
            overbought_range: Tuple of (min, max, step) for overbought threshold
            oversold_range: Tuple of (min, max, step) for oversold threshold
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            n_jobs: Number of parallel jobs (1 for sequential)
            
        Returns:
            Dictionary with optimization results
        """
        # Generate parameter grid
        periods = range(period_range[0], period_range[1] + 1, period_range[2])
        overbought_thresholds = range(overbought_range[0], overbought_range[1] + 1, overbought_range[2])
        oversold_thresholds = range(oversold_range[0], oversold_range[1] + 1, oversold_range[2])
        
        param_grid = []
        for period in periods:
            for overbought in overbought_thresholds:
                for oversold in oversold_thresholds:
                    if oversold < overbought:  # Ensure oversold is less than overbought
                        param_grid.append({
                            'period': period,
                            'overbought': overbought,
                            'oversold': oversold
                        })
        
        print(f"Optimizing RSI strategy with {len(param_grid)} parameter combinations")
        
        # Create a partial function for parameter evaluation
        evaluate_func = partial(self._evaluate_rsi_params, metric=metric)
        
        # Evaluate parameters
        results = self._parallel_evaluate(evaluate_func, param_grid, n_jobs)
        
        # Sort results by metric value (descending)
        sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
        
        # Get best parameters and evaluate on test set
        best_params = sorted_results[0]['params']
        test_performance = self._evaluate_rsi_test(best_params)
        
        return {
            'best_params': best_params,
            'train_performance': sorted_results[0],
            'test_performance': test_performance,
            'all_results': sorted_results
        }
    
    def _evaluate_rsi_params(self, params: Dict, metric: str) -> Dict:
        """
        Evaluate a set of RSI parameters on the training data.
        
        Args:
            params: Dictionary with parameters
            metric: Metric to optimize
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Generate strategy with given parameters
            sg = StrategyGenerator(self.train_data)
            strategy = sg.generate_rsi_strategy(
                period=params['period'],
                overbought=params['overbought'],
                oversold=params['oversold']
            )
            
            # Backtest the strategy
            results = self.train_backtester.backtest_strategy(strategy['signals'])
            metrics = self.train_backtester.calculate_metrics(results)
            
            # Add parameters to results for reference
            return {
                'params': params,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_return': metrics['total_return'],
                'annual_return': metrics['annual_return'],
                'max_drawdown': metrics['max_drawdown'],
                'sortino_ratio': metrics['sortino_ratio'],
                'calmar_ratio': metrics['calmar_ratio'],
                'num_trades': metrics['num_trades'],
                'win_rate': metrics['win_rate']
            }
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return {
                'params': params,
                'sharpe_ratio': -np.inf,
                'total_return': -np.inf,
                'annual_return': -np.inf,
                'max_drawdown': -1.0,
                'sortino_ratio': -np.inf,
                'calmar_ratio': -np.inf,
                'num_trades': 0,
                'win_rate': 0
            }
    
    def _evaluate_rsi_test(self, params: Dict) -> Dict:
        """
        Evaluate RSI parameters on the test data.
        
        Args:
            params: Dictionary with parameters
            
        Returns:
            Dictionary with test evaluation results
        """
        # Generate strategy with given parameters on test data
        sg = StrategyGenerator(self.test_data)
        strategy = sg.generate_rsi_strategy(
            period=params['period'],
            overbought=params['overbought'],
            oversold=params['oversold']
        )
        
        # Backtest the strategy
        results = self.test_backtester.backtest_strategy(strategy['signals'])
        metrics = self.test_backtester.calculate_metrics(results)
        
        return {
            'params': params,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_return': metrics['total_return'],
            'annual_return': metrics['annual_return'],
            'max_drawdown': metrics['max_drawdown'],
            'sortino_ratio': metrics['sortino_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate']
        }
    
    def walk_forward_optimization(self, strategy_func: Callable, param_grid: List[Dict],
                                 window_size: int = 60, step_size: int = 20,
                                 metric: str = 'sharpe_ratio') -> Dict:
        """
        Perform walk-forward optimization for a strategy.
        
        Args:
            strategy_func: Function to generate strategy signals
            param_grid: List of parameter dictionaries to test
            window_size: Number of days in each training window
            step_size: Number of days to step forward for each iteration
            metric: Metric to optimize
            
        Returns:
            Dictionary with walk-forward optimization results
        """
        data_length = len(self.data)
        windows = []
        
        # Generate windows
        for start_idx in range(0, data_length - window_size - step_size, step_size):
            train_end = start_idx + window_size
            test_end = train_end + step_size
            
            train_data = self.data.iloc[start_idx:train_end].copy()
            test_data = self.data.iloc[train_end:test_end].copy()
            
            windows.append((train_data, test_data))
        
        print(f"Performing walk-forward optimization with {len(windows)} windows")
        
        # Results storage
        window_results = []
        optimal_parameters = []
        test_performances = []
        combined_signals = pd.DataFrame(index=self.data.index)
        combined_signals['signal'] = 0
        
        # Process each window
        for i, (train_window, test_window) in enumerate(windows):
            print(f"Processing window {i+1}/{len(windows)}")
            
            # Create local strategy generator and backtester
            sg = StrategyGenerator(train_window)
            bt = Backtester(train_window)
            
            # Evaluate each parameter set on training window
            window_param_results = []
            for params in param_grid:
                # Generate and backtest strategy
                strategy = strategy_func(sg, **params)
                backtest_results = bt.backtest_strategy(strategy['signals'])
                metrics = bt.calculate_metrics(backtest_results)
                
                window_param_results.append({
                    'params': params,
                    'metrics': metrics
                })
            
            # Find best parameters for this window
            best_result = max(window_param_results, key=lambda x: x['metrics'][metric])
            optimal_parameters.append(best_result['params'])
            
            # Apply best parameters to test window
            test_sg = StrategyGenerator(test_window)
            test_strategy = strategy_func(test_sg, **best_result['params'])
            test_bt = Backtester(test_window)
            test_results = test_bt.backtest_strategy(test_strategy['signals'])
            test_metrics = test_bt.calculate_metrics(test_results)
            
            test_performances.append({
                'window': i,
                'params': best_result['params'],
                'metrics': test_metrics
            })
            
            # Add signals to combined signals
            window_idx = test_window.index
            if len(window_idx) > 0:
                combined_signals.loc[window_idx, 'signal'] = test_strategy['signals'].loc[window_idx, 'signal']
            
            window_results.append({
                'window': i,
                'train_results': window_param_results,
                'best_params': best_result['params'],
                'test_performance': test_metrics
            })
        
        # Backtest the combined signals
        filtered_signal_df = combined_signals.loc[combined_signals['signal'] != 0].copy()
        if not filtered_signal_df.empty:
            combined_backtest = Backtester(self.data)
            combined_results = combined_backtest.backtest_strategy(combined_signals)
            combined_metrics = combined_backtest.calculate_metrics(combined_results)
        else:
            combined_metrics = {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'num_trades': 0,
                'win_rate': 0
            }
        
        return {
            'window_results': window_results,
            'optimal_parameters': optimal_parameters,
            'test_performances': test_performances,
            'combined_signals': combined_signals,
            'combined_metrics': combined_metrics
        }
    
    def plot_optimization_results(self, results: Dict, param_x: str, param_y: str, 
                                 metric: str = 'sharpe_ratio') -> None:
        """
        Plot optimization results as a heatmap.
        
        Args:
            results: Dictionary with optimization results
            param_x: Parameter name for the x-axis
            param_y: Parameter name for the y-axis
            metric: Metric to visualize
        """
        # Extract parameters and metric values
        all_results = results['all_results']
        
        # Get unique values for each parameter
        x_values = sorted(list(set(r['params'][param_x] for r in all_results)))
        y_values = sorted(list(set(r['params'][param_y] for r in all_results)))
        
        # Create a grid for the heatmap
        grid = np.zeros((len(y_values), len(x_values)))
        
        # Fill in the grid with metric values
        for result in all_results:
            x_idx = x_values.index(result['params'][param_x])
            y_idx = y_values.index(result['params'][param_y])
            grid[y_idx, x_idx] = result[metric]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(grid, cmap='viridis', aspect='auto')
        plt.colorbar(label=metric)
        
        # Set tick labels
        plt.xticks(range(len(x_values)), x_values)
        plt.yticks(range(len(y_values)), y_values)
        
        # Set axis labels and title
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.title(f'Parameter Optimization Heatmap for {metric}')
        
        # Mark best parameters
        best_params = results['best_params']
        best_x_idx = x_values.index(best_params[param_x])
        best_y_idx = y_values.index(best_params[param_y])
        plt.scatter(best_x_idx, best_y_idx, color='red', marker='*', s=200, label='Best Parameters')
        
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    def validate_strategy(self, strategy_func: Callable, params: Dict) -> Dict:
        """
        Validate a strategy with given parameters on the validation data.
        
        Args:
            strategy_func: Function to generate strategy signals
            params: Strategy parameters
            
        Returns:
            Dictionary with validation results
        """
        # Generate strategy on validation data
        sg = StrategyGenerator(self.validation_data)
        strategy = strategy_func(sg, **params)
        
        # Backtest the strategy
        results = self.validation_backtester.backtest_strategy(strategy['signals'])
        metrics = self.validation_backtester.calculate_metrics(results)
        
        # Compare with train and test performance
        print("\n===== Strategy Validation Results =====")
        print(f"Strategy Parameters: {params}")
        print(f"Validation Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Validation Total Return: {metrics['total_return']:.2%}")
        print(f"Validation Annual Return: {metrics['annual_return']:.2%}")
        print(f"Validation Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Validation Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        
        return {
            'params': params,
            'validation_results': results,
            'validation_metrics': metrics
        }
    
    def plot_train_test_validation_performance(self, strategy_func: Callable, params: Dict) -> None:
        """
        Plot performance across train, test, and validation sets.
        
        Args:
            strategy_func: Function to generate strategy signals
            params: Strategy parameters
        """
        # Generate and backtest strategy on each dataset
        datasets = {
            'Train': (self.train_data, self.train_backtester),
            'Test': (self.test_data, self.test_backtester),
            'Validation': (self.validation_data, self.validation_backtester)
        }
        
        results = {}
        metrics = {}
        
        for name, (data, backtester) in datasets.items():
            sg = StrategyGenerator(data)
            strategy = strategy_func(sg, **params)
            backtest_results = backtester.backtest_strategy(strategy['signals'])
            results[name] = backtest_results
            metrics[name] = backtester.calculate_metrics(backtest_results)
        
        # Plot equity curves
        plt.figure(figsize=(12, 8))
        
        for name, result_df in results.items():
            # Normalize to starting at 100
            norm_equity = result_df['portfolio_value'] / result_df['portfolio_value'].iloc[0] * 100
            plt.plot(result_df.index, norm_equity, label=f"{name} (SR: {metrics[name]['sharpe_ratio']:.2f})")
        
        plt.title(f"Strategy Performance Comparison (Params: {params})")
        plt.xlabel("Date")
        plt.ylabel("Equity (Normalized to 100)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Print metrics comparison
        print("\n===== Performance Metrics Comparison =====")
        metrics_df = pd.DataFrame({
            'Train': metrics['Train'],
            'Test': metrics['Test'],
            'Validation': metrics['Validation']
        })
        print(metrics_df.T)


def main():
    """Main function to demonstrate strategy optimization functionality."""
    # Load OHLCV data
    data_path = 'data/ohlcv_AAPL.csv'
    try:
        # Load data
        df = load_data(data_path)
        print(f"Loaded data from {data_path}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        
        # Create optimizer
        optimizer = StrategyOptimizer(df, train_size=0.6, test_size=0.2, validation_size=0.2)
        
        # Optimize MA Crossover strategy
        print("\n===== Optimizing MA Crossover Strategy =====")
        ma_results = optimizer.optimize_ma_crossover(
            fast_period_range=(5, 50, 5),
            slow_period_range=(20, 200, 20),
            metric='sharpe_ratio',
            n_jobs=1  # Use more jobs for parallel processing
        )
        
        # Plot optimization results
        optimizer.plot_optimization_results(
            ma_results, 'fast_period', 'slow_period', 'sharpe_ratio'
        )
        
        # Get best parameters
        best_ma_params = ma_results['best_params']
        print(f"\nBest MA Crossover Parameters: {best_ma_params}")
        print(f"Train Sharpe Ratio: {ma_results['train_performance']['sharpe_ratio']:.2f}")
        print(f"Test Sharpe Ratio: {ma_results['test_performance']['sharpe_ratio']:.2f}")
        
        # Validate strategy
        ma_validation = optimizer.validate_strategy(
            lambda sg, **params: sg.generate_ma_crossover_strategy(**params),
            best_ma_params
        )
        
        # Plot performance across all datasets
        optimizer.plot_train_test_validation_performance(
            lambda sg, **params: sg.generate_ma_crossover_strategy(**params),
            best_ma_params
        )
        
        # Optimize RSI strategy
        print("\n===== Optimizing RSI Strategy =====")
        rsi_results = optimizer.optimize_rsi_strategy(
            period_range=(5, 30, 5),
            overbought_range=(60, 85, 5),
            oversold_range=(15, 40, 5),
            metric='sharpe_ratio',
            n_jobs=1
        )
        
        # Get best parameters
        best_rsi_params = rsi_results['best_params']
        print(f"\nBest RSI Parameters: {best_rsi_params}")
        print(f"Train Sharpe Ratio: {rsi_results['train_performance']['sharpe_ratio']:.2f}")
        print(f"Test Sharpe Ratio: {rsi_results['test_performance']['sharpe_ratio']:.2f}")
        
        # Validate strategy
        rsi_validation = optimizer.validate_strategy(
            lambda sg, **params: sg.generate_rsi_strategy(**params),
            best_rsi_params
        )
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 