"""
Message to Claude 3.7 Sonnet 
Given the OHLCV data for a stock, use the properties of the equity to develop multiple trading strategies with clear success metrics. 
- It is important to ensure that strategies generated have robust performance and are specific to the given trading data.
- make sure the output strategies are easily backtestable

File description: 
Backtester for OHLCV Data Trading Strategies

This module provides backtesting functionality for trading strategies generated
by the strategy_generator module. It simulates trading based on strategy signals
and calculates performance metrics.

Features:
- Backtesting with realistic trading assumptions (slippage, commission)
- Visualization of equity curves and trade performancef
- Generation of performance metrics and statistics
- Support for comparing multiple strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import datetime as dt
from strategy_generator import StrategyGenerator, load_data


class Backtester:
    """A class for backtesting trading strategies."""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0,
                 commission: float = 0.001, slippage: float = 0.001):
        """
        Initialize the Backtester with OHLCV data and trading parameters.
        
        Args:
            data: DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'
            initial_capital: Initial capital for backtesting
            commission: Commission cost per trade (as a fraction of trade value)
            slippage: Slippage cost per trade (as a fraction of trade value)
        """
        StrategyGenerator.validate_data(data)
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def backtest_strategy(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest a trading strategy based on signals.
        
        Args:
            signals: DataFrame with 'signal' column containing trading signals
                    (1 for buy, -1 for sell, 0 for hold)
            
        Returns:
            DataFrame with backtest results including portfolio value,
            returns, drawdowns, and trade statistics
        """
        # Prepare the results dataframe
        results = signals.copy()
        
        # Generate positions (1 = long, 0 = cash, -1 = short)
        # Convert signal (trade entry signals) to position (held positions)
        if 'position' not in results.columns:
            # If position not provided, calculate from signal
            results['position'] = results['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
            
        # Calculate entry/exit prices (assuming next-day execution at open)
        results['entry_price'] = np.nan
        results['exit_price'] = np.nan
        
        # Find entry points
        entries = results[results['position'].diff() != 0].copy()
        for i, entry in enumerate(entries.index):
            if i < len(entries) - 1:
                exit_idx = entries.index[i + 1]
                results.loc[entry, 'entry_price'] = results.loc[entry, 'Open']
                results.loc[exit_idx, 'exit_price'] = results.loc[exit_idx, 'Open']
        
        # Calculate shares based on available capital
        results['portfolio_value'] = self.initial_capital
        results['shares'] = 0.0
        results['cash'] = self.initial_capital
        
        # Track last trade price for calculating unrealized P&L
        last_trade_price = 0
        
        # Simulate trading
        for i in range(1, len(results)):
            yesterday = results.index[i-1]
            today = results.index[i]
            
            position_change = results.loc[today, 'position'] - results.loc[yesterday, 'position']
            
            # If position changed, execute trade
            if position_change != 0:
                # Get execution price (Open of the day)
                execution_price = results.loc[today, 'Open']
                
                # Apply slippage
                if position_change > 0:  # Buying
                    execution_price *= (1 + self.slippage)
                elif position_change < 0:  # Selling
                    execution_price *= (1 - self.slippage)
                
                # Calculate transaction cost
                transaction_value = abs(position_change) * results.loc[yesterday, 'portfolio_value']
                transaction_cost = transaction_value * self.commission
                
                # Update cash (reduce by transaction cost and trade value if buying)
                if position_change > 0:  # Buying
                    shares_bought = (results.loc[yesterday, 'portfolio_value'] * abs(position_change)) / execution_price
                    results.loc[today, 'shares'] = shares_bought
                    results.loc[today, 'cash'] = results.loc[yesterday, 'portfolio_value'] - \
                                                (shares_bought * execution_price) - transaction_cost
                    last_trade_price = execution_price
                elif position_change < 0:  # Selling
                    shares_sold = results.loc[yesterday, 'shares']
                    results.loc[today, 'shares'] = 0
                    results.loc[today, 'cash'] = results.loc[yesterday, 'cash'] + \
                                                (shares_sold * execution_price) - transaction_cost
            else:
                # No position change, carry forward shares
                results.loc[today, 'shares'] = results.loc[yesterday, 'shares']
                results.loc[today, 'cash'] = results.loc[yesterday, 'cash']
            
            # Calculate portfolio value
            if results.loc[today, 'shares'] > 0:
                equity = results.loc[today, 'shares'] * results.loc[today, 'Close']
                results.loc[today, 'portfolio_value'] = equity + results.loc[today, 'cash']
            else:
                results.loc[today, 'portfolio_value'] = results.loc[today, 'cash']
        
        # Calculate returns and metrics
        results['daily_returns'] = results['portfolio_value'].pct_change()
        results['cumulative_returns'] = (1 + results['daily_returns']).cumprod() - 1
        
        # Calculate drawdown
        results['high_water_mark'] = results['portfolio_value'].cummax()
        results['drawdown'] = (results['portfolio_value'] / results['high_water_mark']) - 1
        
        return results
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        # Filter out NaN values
        returns = results['daily_returns'].dropna()
        
        # Calculate total return
        total_return = (results['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        # Calculate annualized return
        days = (results.index[-1] - results.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Calculate max drawdown
        max_drawdown = results['drawdown'].min()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 else np.nan
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Calculate number of trades
        trades = results[results['position'].diff() != 0]
        num_trades = len(trades) // 2  # Each round trip is considered one trade
        
        # Calculate win rate
        if num_trades > 0:
            entry_prices = results['entry_price'].dropna()
            exit_prices = results['exit_price'].dropna()
            
            if len(entry_prices) > 0 and len(exit_prices) > 0:
                # Calculate trade P&L
                trade_pnl = []
                for i in range(min(len(entry_prices), len(exit_prices))):
                    if i % 2 == 0:  # Every even index is a trade entry, odd is exit
                        entry = entry_prices.iloc[i]
                        if i + 1 < len(exit_prices):
                            exit = exit_prices.iloc[i + 1]
                            pnl = (exit - entry) / entry
                            trade_pnl.append(pnl)
                
                win_rate = len([p for p in trade_pnl if p > 0]) / len(trade_pnl) if len(trade_pnl) > 0 else 0
                avg_win = np.mean([p for p in trade_pnl if p > 0]) if len([p for p in trade_pnl if p > 0]) > 0 else 0
                avg_loss = np.mean([p for p in trade_pnl if p <= 0]) if len([p for p in trade_pnl if p <= 0]) > 0 else 0
                profit_factor = abs(sum([p for p in trade_pnl if p > 0]) / sum([p for p in trade_pnl if p <= 0])) if sum([p for p in trade_pnl if p <= 0]) != 0 else np.inf
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def plot_performance(self, results: pd.DataFrame, title: str = "Strategy Performance") -> None:
        """
        Plot backtest performance charts.
        
        Args:
            results: DataFrame with backtest results
            title: Title for the plot
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot equity curve
        axs[0].plot(results.index, results['portfolio_value'])
        axs[0].set_title(f"{title} - Equity Curve")
        axs[0].set_ylabel("Portfolio Value ($)")
        axs[0].grid(True)
        
        # Plot drawdown
        axs[1].fill_between(results.index, 0, results['drawdown'] * 100, color='red', alpha=0.3)
        axs[1].set_title("Drawdown (%)")
        axs[1].set_ylabel("Drawdown %")
        axs[1].grid(True)
        
        # Plot position changes
        axs[2].plot(results.index, results['position'])
        axs[2].set_title("Position")
        axs[2].set_ylabel("Position")
        axs[2].set_ylim([-1.1, 1.1])
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_strategies(self, strategies_dict: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Compare multiple strategies.
        
        Args:
            strategies_dict: Dictionary of strategy signals with names as keys
            
        Returns:
            Tuple of (DataFrame with comparison metrics, Dictionary of backtest results)
        """
        results = {}
        metrics = {}
        
        for name, signals in strategies_dict.items():
            backtest_results = self.backtest_strategy(signals)
            metrics[name] = self.calculate_metrics(backtest_results)
            results[name] = backtest_results
        
        # Convert metrics to DataFrame for easy comparison
        metrics_df = pd.DataFrame(metrics)
        
        return metrics_df, results
    
    def plot_comparison(self, results_dict: Dict) -> None:
        """
        Plot comparison of equity curves for multiple strategies.
        
        Args:
            results_dict: Dictionary of backtest results with strategy names as keys
        """
        plt.figure(figsize=(12, 8))
        
        # Plot baseline (buy and hold)
        buy_hold_returns = (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0]) - 1
        buy_hold_equity = self.initial_capital * (1 + buy_hold_returns * np.arange(len(self.data)) / len(self.data))
        plt.plot(self.data.index, buy_hold_equity, label='Buy & Hold', linestyle='--', alpha=0.7)
        
        # Plot strategy equity curves
        for name, results in results_dict.items():
            plt.plot(results.index, results['portfolio_value'], label=name)
        
        plt.title("Strategy Comparison - Equity Curves")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    """Main function to demonstrate backtesting functionality."""
    # Load OHLCV data
    data_path = 'data/ohlcv_AAPL.csv'
    try:
        # Load data
        df = load_data(data_path)
        print(f"Loaded data from {data_path}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        
        # Generate strategies
        strategy_gen = StrategyGenerator(df)
        strategies = strategy_gen.generate_all_strategies()
        
        # Create backtester
        backtester = Backtester(df)
        
        # Extract signals from strategies
        signals_dict = {
            name: strat['signals'] for name, strat in strategies.items()
        }
        
        # Compare strategies
        metrics_df, results_dict = backtester.compare_strategies(signals_dict)
        
        # Print metrics
        print("\n===== Strategy Performance Comparison =====\n")
        print(metrics_df.T)
        
        # Plot comparison
        backtester.plot_comparison(results_dict)
        
        # Detailed analysis of best strategy
        best_strategy = metrics_df.loc['annual_return'].idxmax()
        print(f"\n===== Detailed Analysis of Best Strategy: {best_strategy} =====\n")
        best_results = results_dict[best_strategy]
        best_metrics = backtester.calculate_metrics(best_results)
        
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                if metric.endswith('_rate'):
                    print(f"{metric}: {value:.2%}")
                else:
                    print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        # Plot performance of best strategy
        backtester.plot_performance(best_results, title=f"Best Strategy: {best_strategy}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
