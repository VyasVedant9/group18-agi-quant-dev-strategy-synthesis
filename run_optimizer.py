#!/usr/bin/env python3
"""
Prompt: 
I'd like you to implement strategy and parameter optimization to better fit trading strategies to particular assets
Use the results of the backtester.py to guide your strategy optimization and ensure you are using appropriate train/test splits so there is reduced risk of overfit 

Strategy Optimization Runner

This script demonstrates how to use the strategy_optimizer module to find optimal
strategy parameters for various trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, List, Tuple

from strategy_generator import StrategyGenerator, load_data
from backtester import Backtester
from strategy_optimizer import StrategyOptimizer


def optimize_ma_crossover(data_path: str, output_dir: str = "results", n_jobs: int = 1) -> None:
    """
    Optimize Moving Average Crossover strategy parameters.
    
    Args:
        data_path: Path to OHLCV data CSV
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs for optimization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Create optimizer with 60/20/20 train/test/validation split
    optimizer = StrategyOptimizer(df, train_size=0.6, test_size=0.2, validation_size=0.2)
    
    # Run optimization with fine-grained parameters for MA Crossover
    print("\n===== Running MA Crossover Optimization =====")
    ma_results = optimizer.optimize_ma_crossover(
        fast_period_range=(5, 50, 5),   # Fast MA periods from 5 to 50 by steps of 5
        slow_period_range=(20, 200, 20), # Slow MA periods from 20 to 200 by steps of 20
        metric='sharpe_ratio',          # Optimize for Sharpe ratio
        n_jobs=n_jobs                   # Parallel processing
    )
    
    # Display results
    best_params = ma_results['best_params']
    print(f"\nBest MA Crossover Parameters: {best_params}")
    print(f"Train Performance:")
    for key, value in ma_results['train_performance'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTest Performance:")
    for key, value in ma_results['test_performance'].items():
        if isinstance(value, float) and key != 'params':
            print(f"  {key}: {value:.4f}")
    
    # Plot optimization results
    plt.figure(figsize=(10, 8))
    optimizer.plot_optimization_results(
        ma_results, 'fast_period', 'slow_period', 'sharpe_ratio'
    )
    plt.savefig(os.path.join(output_dir, "ma_crossover_optimization.png"))
    
    # Validate strategy on out-of-sample data
    validation_results = optimizer.validate_strategy(
        lambda sg, **params: sg.generate_ma_crossover_strategy(**params),
        best_params
    )
    
    # Plot performance across all datasets
    plt.figure(figsize=(12, 8))
    optimizer.plot_train_test_validation_performance(
        lambda sg, **params: sg.generate_ma_crossover_strategy(**params),
        best_params
    )
    plt.savefig(os.path.join(output_dir, "ma_crossover_performance.png"))
    
    # Save results to CSV
    results_df = pd.DataFrame(ma_results['all_results'])
    results_df.to_csv(os.path.join(output_dir, "ma_crossover_results.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}")


def optimize_rsi_strategy(data_path: str, output_dir: str = "results", n_jobs: int = 1) -> None:
    """
    Optimize RSI strategy parameters.
    
    Args:
        data_path: Path to OHLCV data CSV
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs for optimization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Create optimizer
    optimizer = StrategyOptimizer(df, train_size=0.6, test_size=0.2, validation_size=0.2)
    
    # Run optimization
    print("\n===== Running RSI Strategy Optimization =====")
    rsi_results = optimizer.optimize_rsi_strategy(
        period_range=(5, 30, 5),        # RSI periods from 5 to 30 by steps of 5
        overbought_range=(60, 85, 5),   # Overbought thresholds from 60 to 85 by steps of 5
        oversold_range=(15, 40, 5),     # Oversold thresholds from 15 to 40 by steps of 5
        metric='sharpe_ratio',
        n_jobs=n_jobs
    )
    
    # Display results
    best_params = rsi_results['best_params']
    print(f"\nBest RSI Parameters: {best_params}")
    print(f"Train Performance:")
    for key, value in rsi_results['train_performance'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTest Performance:")
    for key, value in rsi_results['test_performance'].items():
        if isinstance(value, float) and key != 'params':
            print(f"  {key}: {value:.4f}")
    
    # Validate strategy
    validation_results = optimizer.validate_strategy(
        lambda sg, **params: sg.generate_rsi_strategy(**params),
        best_params
    )
    
    # Plot performance
    plt.figure(figsize=(12, 8))
    optimizer.plot_train_test_validation_performance(
        lambda sg, **params: sg.generate_rsi_strategy(**params),
        best_params
    )
    plt.savefig(os.path.join(output_dir, "rsi_strategy_performance.png"))
    
    # Save results
    results_df = pd.DataFrame(rsi_results['all_results'])
    results_df.to_csv(os.path.join(output_dir, "rsi_strategy_results.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}")


def run_walk_forward_optimization(data_path: str, output_dir: str = "results") -> None:
    """
    Run walk-forward optimization for a strategy.
    
    Args:
        data_path: Path to OHLCV data CSV
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Create optimizer
    optimizer = StrategyOptimizer(df, train_size=0.8, test_size=0.2, validation_size=0.0)
    
    # Define strategy function and parameter grid
    def strategy_func(sg, fast_period, slow_period):
        return sg.generate_ma_crossover_strategy(
            fast_period=fast_period,
            slow_period=slow_period
        )
    
    # Create parameter grid (smaller grid for walk-forward to be faster)
    param_grid = []
    for fast in [5, 10, 15, 20]:
        for slow in [50, 100, 150]:
            if fast < slow:
                param_grid.append({
                    'fast_period': fast,
                    'slow_period': slow
                })
    
    # Run walk-forward optimization
    print("\n===== Running Walk-Forward Optimization =====")
    wfo_results = optimizer.walk_forward_optimization(
        strategy_func=strategy_func,
        param_grid=param_grid,
        window_size=60,  # 60-day training window
        step_size=20,    # 20-day testing window
        metric='sharpe_ratio'
    )
    
    # Display combined results
    print("\nWalk-Forward Optimization Results:")
    print(f"Combined Metrics:")
    for key, value in wfo_results['combined_metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Plot parameter stability
    plt.figure(figsize=(12, 6))
    
    # Extract optimal parameters for each window
    windows = range(len(wfo_results['optimal_parameters']))
    fast_periods = [p['fast_period'] for p in wfo_results['optimal_parameters']]
    slow_periods = [p['slow_period'] for p in wfo_results['optimal_parameters']]
    
    plt.subplot(2, 1, 1)
    plt.plot(windows, fast_periods, 'o-', label='Fast Period')
    plt.title('Optimal Fast Period by Window')
    plt.xlabel('Window')
    plt.ylabel('Fast Period')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(windows, slow_periods, 'o-', label='Slow Period')
    plt.title('Optimal Slow Period by Window')
    plt.xlabel('Window')
    plt.ylabel('Slow Period')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "walk_forward_parameters.png"))
    
    # Plot test performance by window
    plt.figure(figsize=(12, 6))
    
    # Extract performance metrics for each window
    returns = [p['metrics']['total_return'] for p in wfo_results['test_performances']]
    sharpes = [p['metrics']['sharpe_ratio'] for p in wfo_results['test_performances']]
    
    plt.subplot(2, 1, 1)
    plt.bar(windows, returns)
    plt.title('Total Return by Window')
    plt.xlabel('Window')
    plt.ylabel('Total Return')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.bar(windows, sharpes)
    plt.title('Sharpe Ratio by Window')
    plt.xlabel('Window')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "walk_forward_performance.png"))
    
    print(f"\nResults saved to {output_dir}")


def run_all_optimizations(data_path: str, output_dir: str = "results", n_jobs: int = 1) -> None:
    """
    Run all optimization methods.
    
    Args:
        data_path: Path to OHLCV data CSV
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs for optimization
    """
    # Create separate directories for each strategy
    ma_dir = os.path.join(output_dir, "ma_crossover")
    rsi_dir = os.path.join(output_dir, "rsi")
    wfo_dir = os.path.join(output_dir, "walk_forward")
    
    os.makedirs(ma_dir, exist_ok=True)
    os.makedirs(rsi_dir, exist_ok=True)
    os.makedirs(wfo_dir, exist_ok=True)
    
    # Run all optimizations
    optimize_ma_crossover(data_path, ma_dir, n_jobs)
    optimize_rsi_strategy(data_path, rsi_dir, n_jobs)
    run_walk_forward_optimization(data_path, wfo_dir)


def main():
    """Main function to parse arguments and run optimizations."""
    parser = argparse.ArgumentParser(description='Optimize trading strategies for OHLCV data.')
    parser.add_argument('--data', type=str, default='ohlcv_AAPL.csv',
                        help='Path to OHLCV data CSV file')
    parser.add_argument('--output', type=str, default='optimization_results',
                        help='Directory to save optimization results')
    parser.add_argument('--strategy', type=str, choices=['ma', 'rsi', 'wfo', 'all'],
                        default='all', help='Strategy to optimize')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of parallel jobs for optimization')
    
    args = parser.parse_args()
    
    # Run selected optimization
    if args.strategy == 'ma':
        optimize_ma_crossover(args.data, args.output, args.jobs)
    elif args.strategy == 'rsi':
        optimize_rsi_strategy(args.data, args.output, args.jobs)
    elif args.strategy == 'wfo':
        run_walk_forward_optimization(args.data, args.output)
    else:  # 'all'
        run_all_optimizations(args.data, args.output, args.jobs)


if __name__ == "__main__":
    main() 