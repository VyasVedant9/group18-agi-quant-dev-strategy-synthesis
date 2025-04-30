import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns
from scipy.stats import norm


# PROMPT
"""Build a robust backtesting framework for a trading strategy using OHLCV data from a CSV file (e.g., AAPL data with Date, Open, High, Low, Close, Volume).

Requirements:

The strategy function must take historical data and generate buy/sell signals based on defined logic (e.g., moving average crossover, RSI levels).

The backtester should simulate trades using these signals:

Track cash, equity, open positions.

Assume fixed position sizing (e.g., 100 shares per trade) or allow configurable sizing.

Include parameters for slippage and transaction fees.

Handle long-only or long/short positions.

Output:

A trade log (entry/exit date, price, size, P&L per trade).

Portfolio equity curve over time.

Key performance metrics: total return, CAGR, Sharpe ratio, max drawdown, number of trades, win rate, average trade duration.

Ensure that only past data is used when making trading decisions (no lookahead bias).

Optionally visualize trades on price chart with entry/exit points.

Keep the design modular so different strategies can be plugged in easily."""
class Backtester:
    def __init__(self, data_path='./data/ohlcv_AAPL.csv', initial_capital=10000, position_size=0.1, 
                 commission=0, slippage=0, stop_loss_pct=0.05, take_profit_pct=0.10):
        """
        Initialize with percentage-based position sizing
        """
        self.initial_capital = initial_capital
        self.position_size = position_size  # Now as percentage (0.1 = 10%)
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        # Load data with absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_data_path = os.path.join(script_dir, data_path)
        self.data = self._load_data(full_data_path)
        self._prepare_data()
        self.reset_backtest()
    def _check_stop_loss(self, current_date, current_price):
        """Check if stop loss was hit for current position"""
        if self.current_position == 0 or not self.stop_loss_pct:
            return False
            
        entry_price = self.trades[-1]['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        
        if pnl_pct <= -self.stop_loss_pct:
            self._exit_position(current_date, current_price, reason='stop_loss')
            return True
        return False
    def _check_take_profit(self, current_date, current_price):
        """Check if take profit was hit"""
        if self.current_position == 0 or not self.take_profit_pct:
            return False
            
        entry_price = self.trades[-1]['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        
        if pnl_pct >= self.take_profit_pct:
            self._exit_position(current_date, current_price, reason='take_profit')
            return True
        return False
    def _load_data(self, data_path):
        """Load data with validation"""
        try:
            df = pd.read_csv(data_path, parse_dates=['Date'])
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns. Needed: {required_cols}")
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            raise ValueError(f"Error loading {data_path}: {str(e)}")

    def moving_average_crossover_strategy(self, short_window=5, long_window=20):
        """Enhanced crossover strategy"""
        self.data['ma_short'] = self.data['Close'].rolling(short_window).mean()
        self.data['ma_long'] = self.data['Close'].rolling(long_window).mean()
        
        # Generate signals only on crosses
        self.signals['signal'] = 0
        cross_above = (self.data['ma_short'] > self.data['ma_long']) & \
                     (self.data['ma_short'].shift() <= self.data['ma_long'].shift())
        cross_below = (self.data['ma_short'] < self.data['ma_long']) & \
                     (self.data['ma_short'].shift() >= self.data['ma_long'].shift())
        
        self.signals.loc[cross_above, 'signal'] = 1
        self.signals.loc[cross_below, 'signal'] = -1
        
        print(f"\nSignals generated - Buy: {sum(self.signals['signal'] == 1)}, Sell: {sum(self.signals['signal'] == -1)}")

    def _enter_position(self, date, price):
        """Percentage-based position sizing"""
        if self.current_position != 0:
            return
            
        # Calculate shares based on percentage of capital
        max_shares = int((self.current_cash * self.position_size) / price)
        if max_shares < 1:
            print(f"Not enough capital for even 1 share at {price}")
            return
            
        cost = max_shares * price * (1 + self.slippage) + self.commission
        if cost > self.current_cash:
            max_shares = int(self.current_cash / (price * (1 + self.slippage)))
            if max_shares < 1:
                return
                
        self.current_position = max_shares
        self.current_cash -= cost
        print(f"Bought {max_shares} shares at {price:.2f} on {date.date()}")
    
    def _prepare_data(self):
        """Enhanced data preparation with more indicators"""
        # Validate data exists
        if self.data.empty:
            raise ValueError("No data available after loading")
            
        # Calculate basic metrics
        self.data['returns'] = self.data['Close'].pct_change()
        
        # Enhanced moving averages
        self.data['ma5'] = self.data['Close'].rolling(window=5).mean()
        self.data['ma20'] = self.data['Close'].rolling(window=20).mean()
        self.data['ma50'] = self.data['Close'].rolling(window=50).mean()
        
        # Enhanced RSI calculation
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # Additional indicators
        self.data['atr'] = self._calculate_atr()
        self.data['macd'], self.data['signal_line'] = self._calculate_macd()
        
        # Clean data
        self.data.dropna(inplace=True)
        
        if self.data.empty:
            raise ValueError("No data available after cleaning")
    
    def _calculate_atr(self, window=14):
        """Calculate Average True Range"""
        high_low = self.data['High'] - self.data['Low']
        high_close = (self.data['High'] - self.data['Close'].shift()).abs()
        low_close = (self.data['Low'] - self.data['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def _calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def reset_backtest(self):
        """Reset all backtesting variables"""
        self.current_cash = self.initial_capital
        self.current_position = 0
        self.current_equity = [self.initial_capital]
        self.trades = []
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['signal'] = 0
    
    def moving_average_crossover_strategy(self, short_window=5, long_window=20):
        """Enhanced MA crossover strategy with signal validation"""
        # Validate windows
        if short_window >= long_window:
            raise ValueError("Short window must be smaller than long window")
            
        # Calculate moving averages
        self.data['ma_short'] = self.data['Close'].rolling(window=short_window).mean()
        self.data['ma_long'] = self.data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0
        
        # Buy when short MA crosses above long MA
        signals.loc[(self.data['ma_short'] > self.data['ma_long']) & 
                  (self.data['ma_short'].shift() <= self.data['ma_long'].shift()), 'signal'] = 1
        
        # Sell when short MA crosses below long MA
        signals.loc[(self.data['ma_short'] < self.data['ma_long']) & 
                  (self.data['ma_short'].shift() >= self.data['ma_long'].shift()), 'signal'] = -1
        
        # Debug output
        print(f"\nMA Crossover Signals Generated:")
        print(f"Buy signals: {sum(signals['signal'] == 1)}")
        print(f"Sell signals: {sum(signals['signal'] == -1)}")
        
        self.signals['signal'] = signals['signal']
    
    def rsi_strategy(self, oversold=30, overbought=70):
        """RSI threshold strategy with validation"""
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0
        
        # Buy when RSI crosses above oversold
        signals.loc[(self.data['rsi'] > oversold) & 
                  (self.data['rsi'].shift() <= oversold), 'signal'] = 1
        
        # Sell when RSI crosses below overbought
        signals.loc[(self.data['rsi'] < overbought) & 
                  (self.data['rsi'].shift() >= overbought), 'signal'] = -1
        
        print(f"\nRSI Signals Generated:")
        print(f"Buy signals: {sum(signals['signal'] == 1)}")
        print(f"Sell signals: {sum(signals['signal'] == -1)}")
        
        self.signals['signal'] = signals['signal']
    
    def run_backtest(self):
        """Enhanced backtest execution with validation"""
        if not hasattr(self, 'signals'):
            raise ValueError("No strategy signals generated. Run a strategy first.")
            
        print("\nRunning backtest...")
        
        for i in range(1, len(self.data)):
            current_date = self.data.index[i]
            prev_date = self.data.index[i-1]
            
            # Exit conditions for existing position
            if self.current_position != 0:
                current_trade = self.trades[-1]
                current_price = self.data.loc[current_date, 'Close']
                pnl_pct = (current_price - current_trade['entry_price']) / current_trade['entry_price']
                
                # Stop loss check
                if self.stop_loss_pct and pnl_pct <= -self.stop_loss_pct:
                    self._exit_position(current_date, current_price, reason='stop_loss')
                    continue
                    
                # Take profit check
                if self.take_profit_pct and pnl_pct >= self.take_profit_pct:
                    self._exit_position(current_date, current_price, reason='take_profit')
                    continue
            
            # Signal-based trading
            current_signal = self.signals.loc[current_date, 'signal']
            prev_signal = self.signals.loc[prev_date, 'signal']
            
            # New buy signal
            if current_signal == 1 and prev_signal != 1:
                self._enter_position(current_date, self.data.loc[current_date, 'Close'])
            
            # New sell signal (only if we have a position)
            elif current_signal == -1 and prev_signal != -1 and self.current_position > 0:
                self._exit_position(current_date, self.data.loc[current_date, 'Close'])
            
            # Update equity curve
            self._update_equity(current_date)
        
        print("Backtest completed successfully")
    
    def _enter_position(self, date, price):
        """Enter a new long position"""
        if self.current_position != 0:
            return  # Can't enter if we already have a position
        
        # Apply slippage
        entry_price = price * (1 + self.slippage)
        
        # Calculate position size
        if isinstance(self.position_size, int):  # Fixed shares
            shares = self.position_size
        else:  # Fixed dollar amount
            shares = int(self.position_size / entry_price)
        
        # Calculate cost
        cost = shares * entry_price + self.commission
        
        if cost > self.current_cash:
            print(f"Insufficient cash to enter position at {date}")
            return
        
        # Update position
        self.current_position = shares
        self.current_cash -= cost
        
        # Record trade
        trade = {
            'entry_date': date,
            'entry_price': entry_price,
            'shares': shares,
            'exit_date': None,
            'exit_price': None,
            'pnl': None,
            'return_pct': None,
            'duration': None,
            'exit_reason': None
        }
        self.trades.append(trade)
        print(f"Entered long position at {date}: {shares} shares @ ${entry_price:.2f}")
    
    def _exit_position(self, date, price, reason='signal'):
        """Exit current position"""
        if self.current_position == 0:
            return
        
        # Apply slippage
        exit_price = price * (1 - self.slippage)
        
        # Calculate proceeds
        proceeds = self.current_position * exit_price - self.commission
        entry_price = self.trades[-1]['entry_price']
        
        # Calculate P&L
        pnl = proceeds - (self.current_position * entry_price)
        return_pct = (exit_price - entry_price) / entry_price
        duration = (date - self.trades[-1]['entry_date']).days
        
        # Update cash and position
        self.current_cash += proceeds
        self.current_position = 0
        
        # Update trade record
        self.trades[-1].update({
            'exit_date': date,
            'exit_price': exit_price,
            'pnl': pnl,
            'return_pct': return_pct,
            'duration': duration,
            'exit_reason': reason
        })
        print(f"Exited position at {date}: {self.trades[-1]['shares']} shares @ ${exit_price:.2f} ({reason})")
    
    def _update_equity(self, date):
        """Update equity curve"""
        position_value = self.current_position * self.data.loc[date, 'Close']
        total_equity = self.current_cash + position_value
        self.current_equity.append(total_equity)
    
    def get_performance_metrics(self):
        """Calculate and return performance metrics with error handling"""
        metrics = {
            'total_return_pct': 0,
            'cagr_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'num_trades': 0,
            'win_rate_pct': 0,
            'avg_trade_return_pct': 0,
            'avg_trade_duration_days': 0,
            'final_equity': self.current_equity[-1] if hasattr(self, 'current_equity') else self.initial_capital
        }

        if not hasattr(self, 'trades') or not self.trades:
            return metrics
            
        # Prepare trade dataframe
        trades_df = pd.DataFrame(self.trades)
        trades_df = trades_df[trades_df['exit_date'].notna()]  # Only completed trades
        
        if trades_df.empty:
            return metrics
            
        try:
            # Calculate metrics
            total_return = (self.current_equity[-1] / self.initial_capital - 1) * 100
            days = (self.data.index[-1] - self.data.index[0]).days
            cagr = ((self.current_equity[-1] / self.initial_capital) ** (365/days) - 1) * 100 if days > 0 else 0
            
            # Sharpe ratio
            equity_series = pd.Series(self.current_equity, index=self.data.index[:len(self.current_equity)])
            daily_returns = equity_series.pct_change().dropna()
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 1 else 0
            
            # Drawdown
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Trade metrics
            num_trades = len(trades_df)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
            avg_trade_return = trades_df['return_pct'].mean() * 100 if num_trades > 0 else 0
            avg_trade_duration = trades_df['duration'].mean() if num_trades > 0 else 0
            
            metrics.update({
                'total_return_pct': total_return,
                'cagr_pct': cagr,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'num_trades': num_trades,
                'win_rate_pct': win_rate,
                'avg_trade_return_pct': avg_trade_return,
                'avg_trade_duration_days': avg_trade_duration,
                'final_equity': self.current_equity[-1]
            })
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
        
        return metrics
    
    def plot_equity_curve(self):
        """Plot the equity curve"""
        plt.figure(figsize=(12, 6))
        equity_series = pd.Series(self.current_equity, index=self.data.index[:len(self.current_equity)])
        equity_series.plot(label='Equity Curve')
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_trades(self):
        """Plot price chart with trade entries and exits"""
        plt.figure(figsize=(14, 7))
        
        # Plot price
        self.data['Close'].plot(label='Price', alpha=0.5)
        
        # Plot moving averages
        if 'ma_short' in self.data.columns:
            self.data[['ma_short', 'ma_long']].plot(ax=plt.gca(), alpha=0.7)
        
        # Plot trades
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            plt.scatter(trades_df['entry_date'], 
                       trades_df['entry_price'], 
                       color='green', marker='^', alpha=1, label='Buy')
            
            plt.scatter(trades_df['exit_date'], 
                       trades_df['exit_price'], 
                       color='red', marker='v', alpha=1, label='Sell')
        
        plt.title('Trade Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def generate_report(self):
        """Generate report with error handling"""
        try:
            metrics = self.get_performance_metrics()
            
            print("\n=== Backtest Performance Report ===")
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"Final Equity: ${metrics.get('final_equity', self.initial_capital):,.2f}")
            print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
            print(f"CAGR: {metrics.get('cagr_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"\nTrade Statistics:")
            print(f"Number of Trades: {metrics.get('num_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
            print(f"Avg Trade Return: {metrics.get('avg_trade_return_pct', 0):.2f}%")
            print(f"Avg Trade Duration: {metrics.get('avg_trade_duration_days', 0):.1f} days")
            
            if metrics.get('num_trades', 0) > 0:
                self.plot_equity_curve()
                self.plot_trades()
            else:
                print("\nNo trades were executed during this backtest period.")
                print("Possible reasons:")
                print("- No signals were generated by your strategy")
                print("- Prices never hit your entry/exit criteria")
                print("- Insufficient capital for position sizing")
                
        except Exception as e:
            print(f"\nError generating report: {str(e)}")
        
        return metrics


if __name__ == "__main__":
    import os
    try:
        # Initialize with enhanced parameters for AAPL
        backtester = Backtester(
            data_path='./data/ohlcv_AAPL.csv',
            initial_capital=10000,
            position_size=50,  # Reduced shares for better testing
            commission=1.00,   # Small commission
            slippage=0.001,    # Minor slippage
            stop_loss_pct=0.05,  # 5% stop loss
            take_profit_pct=0.10  # 10% take profit
        )
        results_dir = "./FirstGenerationResults"
        # Print data validation
        print("\nData Validation:")
        print(f"First date: {backtester.data.index[0]}")
        print(f"Last date: {backtester.data.index[-1]}")
        print(f"Rows: {len(backtester.data)}")
        print(f"Recent Close Prices:\n{backtester.data['Close'].tail()}")
        
        # Run strategy with debug output
        print("\nRunning MA Crossover Strategy (5,20) with debug:")
        backtester.moving_average_crossover_strategy(short_window=5, long_window=20)
        
        # Visualize indicators before backtest
        plt.figure(figsize=(14, 6))
        plt.plot(backtester.data['Close'], label='Price', alpha=0.5)
        plt.plot(backtester.data['ma5'], label='5-day MA', alpha=0.75)
        plt.plot(backtester.data['ma20'], label='20-day MA', alpha=0.75)
        plt.title('AAPL Price and Moving Averages')
        plt.legend()
        # plt.show()
        plt.savefig(f"{results_dir}/PriceAndMovingAverages.png")
        plt.show()
        # plt.close()
        
        # Execute backtest with progress tracking
        print("\nRunning backtest with trade logging:")
        backtester.run_backtest()
        
        # Generate detailed report
        report = backtester.generate_report()
        
        # Enhanced visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Price and signals
        backtester.data['Close'].plot(ax=ax1, label='Price')
        ax1.plot(backtester.data['ma5'], label='5-day MA', alpha=0.75)
        ax1.plot(backtester.data['ma20'], label='20-day MA', alpha=0.75)
        
        trades_df = pd.DataFrame(backtester.trades)
        if not trades_df.empty:
            ax1.scatter(trades_df['entry_date'], 
                       trades_df['entry_price'], 
                       color='green', marker='^', label='Buy')
            ax1.scatter(trades_df['exit_date'], 
                       trades_df['exit_price'], 
                       color='red', marker='v', label='Sell')
        
        ax1.set_title('AAPL Trading Strategy Performance')
        ax1.legend()
        
        # Equity curve
        equity = pd.Series(backtester.current_equity, index=backtester.data.index[:len(backtester.current_equity)])
        equity.plot(ax=ax2, label='Equity', color='purple')
        ax2.set_title('Portfolio Equity')
        ax2.axhline(y=backtester.initial_capital, color='gray', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/StrategyPerformanceAndEquity.png")
        plt.show()
        
        
        # Trade analysis
        if not trades_df.empty:
            print("\nTrade Details:")
            print(trades_df[['entry_date', 'entry_price', 'exit_date', 
                           'exit_price', 'return_pct', 'duration']].tail())
            
            # Plot returns distribution
            plt.figure(figsize=(10, 5))
            trades_df['return_pct'].hist(bins=20)
            plt.title('Distribution of Trade Returns')
            plt.xlabel('Return (%)')
            plt.ylabel('Frequency')
            plt.savefig(f"{results_dir}/TradeReturnsDistribution.png")
            plt.show()
            trades_df.to_csv(f"{results_dir}/TradeDetails.csv", index=False)

            with open(f"{results_dir}/PerformanceMetrics.txt", "w") as f:
                f.write("=== Backtest Performance Report ===\n")
                f.write(f"Initial Capital: ${backtester.initial_capital:,.2f}\n")
                f.write(f"Final Equity: ${backtester.current_equity[-1]:,.2f}\n")
                f.write(f"Total Return: {report.get('total_return_pct', 0):.2f}%\n")
                f.write(f"CAGR: {report.get('cagr_pct', 0):.2f}%\n")
                f.write(f"Sharpe Ratio: {report.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Max Drawdown: {report.get('max_drawdown_pct', 0):.2f}%\n")
                f.write("\nTrade Statistics:\n")
                f.write(f"Number of Trades: {report.get('num_trades', 0)}\n")
                f.write(f"Win Rate: {report.get('win_rate_pct', 0):.2f}%\n")
                f.write(f"Avg Trade Return: {report.get('avg_trade_return_pct', 0):.2f}%\n")
                f.write(f"Avg Trade Duration: {report.get('avg_trade_duration_days', 0):.1f} days\n")
        
        print("\nAll results saved to FirstGenerationResults folder:")
        print(f"- PriceAndMovingAverages.png")
        print(f"- StrategyPerformanceAndEquity.png")
        if not trades_df.empty:
            print(f"- TradeReturnsDistribution.png")
            print(f"- TradeDetails.csv")
            print(f"- PerformanceMetrics.txt")
    except Exception as e:
        print(f"\nCritical Error: {str(e)}")
        print("\nTroubleshooting Checklist:")
        print("1. Verify ohlcv_AAPL.csv exists in ./data/ folder")
        print("2. Check CSV contains columns: Date,Open,High,Low,Close,Volume")
        print("3. Ensure data covers at least 3 months with no gaps")
        print("4. Validate moving averages are calculated (should see MA values in debug output)")
        print("5. Check signal generation count matches your expectations")