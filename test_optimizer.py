#!/usr/bin/env python3
"""
Prompt: 
I'd like you to implement strategy and parameter optimization to better fit trading strategies to particular assets
Use the results of the backtester.py to guide your strategy optimization and ensure you are using appropriate train/test splits so there is reduced risk of overfit 

Quick test script for the strategy optimizer
"""

import os
import matplotlib.pyplot as plt
from strategy_generator import load_data
from strategy_optimizer import StrategyOptimizer

def main():
    """Test the strategy optimizer with a small parameter grid."""
    # Ensure we're testing in the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Data path
    data_path = 'data/ohlcv_AAPL.csv'
    
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Create optimizer with a smaller train/test split for faster testing
    optimizer = StrategyOptimizer(df, train_size=0.7, test_size=0.3, validation_size=0.0)
    
    # Run a quick optimization with a smaller parameter grid
    print("\n===== Quick MA Crossover Optimization Test =====")
    ma_results = optimizer.optimize_ma_crossover(
        fast_period_range=(10, 30, 10),  # Only 3 values: 10, 20, 30
        slow_period_range=(50, 150, 50), # Only 3 values: 50, 100, 150
        metric='sharpe_ratio'
    )
    
    # Display results
    best_params = ma_results['best_params']
    print(f"\nBest MA Crossover Parameters: {best_params}")
    print(f"Train Sharpe Ratio: {ma_results['train_performance']['sharpe_ratio']:.2f}")
    print(f"Test Sharpe Ratio: {ma_results['test_performance']['sharpe_ratio']:.2f}")
    
    # Plot optimization results
    optimizer.plot_optimization_results(
        ma_results, 'fast_period', 'slow_period', 'sharpe_ratio'
    )
    
    # Plot performance
    print("\nPlotting performance...")
    strategy_func = lambda sg, **params: sg.generate_ma_crossover_strategy(**params)
    
    # Create test data only visualization (since we skipped validation)
    train_sg = StrategyOptimizer(optimizer.train_data)
    train_sg.test_size = 0.3
    train_sg.validation_size = 0.0
    train_sg._split_data()
    
    # Show performance on train/test split of the training data
    train_sg.plot_train_test_validation_performance(
        strategy_func, best_params
    )
    plt.title(f"Strategy Performance with Best Parameters: {best_params}")
    plt.show()
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 