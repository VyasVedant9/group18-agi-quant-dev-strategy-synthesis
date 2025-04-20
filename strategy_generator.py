"""
Message to Claude 3.7 Sonnet 
Given the OHLCV data for a stock, use the properties of the equity to develop multiple trading strategies with clear success metrics. 
- It is important to ensure that strategies generated have robust performance and are specific to the given trading data.
- make sure the output strategies are easily backtestable

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


class StrategyGenerator:
    """A class that generates multiple trading strategies based on OHLCV data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the StrategyGenerator with OHLCV data.
        
        Args:
            data: DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'
        """
        self.validate_data(data)
        self.data = data.copy()
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['position'] = 0  # 1 for long, -1 for short, 0 for neutral
        
    @staticmethod
    def validate_data(data: pd.DataFrame) -> None:
        """
        Validate that the data contains the required columns.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Add common technical indicators to the data.
        
        Returns:
            DataFrame with added technical indicators
        """
        df = self.data.copy()
        
        # Simple Moving Averages
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI (14-period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (20-day, 2 standard deviations)
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        
        # Return dataframe with indicators
        return df
    
    def generate_ma_crossover_strategy(self, fast_period: int = 20, slow_period: int = 50) -> Dict:
        """
        Generate a Moving Average Crossover strategy.
        
        Args:
            fast_period: Period of fast moving average
            slow_period: Period of slow moving average
            
        Returns:
            Dictionary with strategy signals and parameters
        """
        df = self.data.copy()
        df[f'SMA{fast_period}'] = df['Close'].rolling(window=fast_period).mean()
        df[f'SMA{slow_period}'] = df['Close'].rolling(window=slow_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df[f'SMA{fast_period}'] > df[f'SMA{slow_period}'], 1, 0)
        df['position'] = df['signal'].diff()
        
        # Parameters for backtesting
        params = {
            'strategy_name': 'MA Crossover',
            'fast_period': fast_period,
            'slow_period': slow_period,
            'parameters': f"Fast MA: {fast_period}, Slow MA: {slow_period}",
            'description': f"Buy when {fast_period}-day MA crosses above {slow_period}-day MA, sell when it crosses below."
        }
        
        return {'signals': df, 'params': params}
    
    def generate_rsi_strategy(self, period: int = 14, overbought: int = 70, oversold: int = 30) -> Dict:
        """
        Generate an RSI-based strategy.
        
        Args:
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold
            
        Returns:
            Dictionary with strategy signals and parameters
        """
        df = self.data.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df['RSI'] < oversold, 1, 0)  # Buy when RSI < oversold
        df['signal'] = np.where(df['RSI'] > overbought, -1, df['signal'])  # Sell when RSI > overbought
        
        # Convert to position changes
        df['position'] = df['signal'].diff()
        
        # Parameters for backtesting
        params = {
            'strategy_name': 'RSI Strategy',
            'period': period,
            'overbought': overbought,
            'oversold': oversold,
            'parameters': f"Period: {period}, Overbought: {overbought}, Oversold: {oversold}",
            'description': f"Buy when RSI falls below {oversold}, sell when it rises above {overbought}."
        }
        
        return {'signals': df, 'params': params}
    
    def generate_bollinger_bands_strategy(self, period: int = 20, std_dev: float = 2.0) -> Dict:
        """
        Generate a Bollinger Bands strategy.
        
        Args:
            period: Bollinger Bands period
            std_dev: Number of standard deviations
            
        Returns:
            Dictionary with strategy signals and parameters
        """
        df = self.data.copy()
        
        # Calculate Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=period).mean()
        df['BB_std'] = df['Close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + std_dev * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - std_dev * df['BB_std']
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df['Close'] < df['BB_lower'], 1, 0)  # Buy when price below lower band
        df['signal'] = np.where(df['Close'] > df['BB_upper'], -1, df['signal'])  # Sell when price above upper band
        
        # Convert to position changes
        df['position'] = df['signal'].diff()
        
        # Parameters for backtesting
        params = {
            'strategy_name': 'Bollinger Bands Strategy',
            'period': period,
            'std_dev': std_dev,
            'parameters': f"Period: {period}, Standard Deviations: {std_dev}",
            'description': f"Buy when price touches lower band, sell when it touches upper band."
        }
        
        return {'signals': df, 'params': params}
    
    def generate_volume_breakout_strategy(self, volume_threshold: float = 1.5, price_period: int = 20) -> Dict:
        """
        Generate a Volume Breakout strategy.
        
        Args:
            volume_threshold: Volume increase threshold (e.g., 1.5 = 50% above average)
            price_period: Period for price high/low calculation
            
        Returns:
            Dictionary with strategy signals and parameters
        """
        df = self.data.copy()
        
        # Calculate volume and price indicators
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['Price_High'] = df['Close'].rolling(window=price_period).max()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
        
        # Generate signals
        df['signal'] = 0
        # Buy when volume spikes and price breaks out above recent high
        breakout_condition = (df['Volume_Ratio'] > volume_threshold) & (df['Close'] > df['Price_High'].shift(1))
        df.loc[breakout_condition, 'signal'] = 1
        
        # Sell after holding for a specified period (e.g., 5 days)
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:
                exit_idx = min(i + 5, len(df) - 1)
                df['signal'].iloc[exit_idx] = -1
        
        # Convert to position changes
        df['position'] = df['signal'].diff()
        
        # Parameters for backtesting
        params = {
            'strategy_name': 'Volume Breakout Strategy',
            'volume_threshold': volume_threshold,
            'price_period': price_period,
            'parameters': f"Volume Threshold: {volume_threshold}x average, Price Period: {price_period} days",
            'description': f"Buy when volume exceeds {volume_threshold}x average and price breaks above {price_period}-day high."
        }
        
        return {'signals': df, 'params': params}
    
    def generate_macd_strategy(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict:
        """
        Generate a MACD strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Dictionary with strategy signals and parameters
        """
        df = self.data.copy()
        
        # Calculate MACD
        df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Generate signals
        df['signal'] = 0
        # Buy when MACD crosses above signal line
        df['signal'] = np.where((df['MACD'] > df['MACD_signal']) & 
                                (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 1, 0)
        # Sell when MACD crosses below signal line
        df['signal'] = np.where((df['MACD'] < df['MACD_signal']) & 
                                (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), -1, df['signal'])
        
        # Convert to position changes
        df['position'] = df['signal']
        
        # Parameters for backtesting
        params = {
            'strategy_name': 'MACD Strategy',
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'parameters': f"Fast EMA: {fast_period}, Slow EMA: {slow_period}, Signal: {signal_period}",
            'description': f"Buy when MACD crosses above signal line, sell when it crosses below."
        }
        
        return {'signals': df, 'params': params}
    
    def evaluate_strategy(self, strategy_signals: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for a strategy.
        
        Args:
            strategy_signals: DataFrame with position signals
            
        Returns:
            Dictionary of performance metrics
        """
        df = strategy_signals.copy()
        
        # Calculate daily returns
        df['daily_return'] = df['Close'].pct_change()
        
        # Calculate strategy returns (position taken at close, realized next day)
        df['strategy_return'] = df['daily_return'].shift(-1) * df['signal']
        
        # Remove NaN values
        df = df.dropna(subset=['strategy_return'])
        
        # Calculate metrics
        total_return = (1 + df['strategy_return']).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        
        # Calculate drawdown
        cumulative_return = (1 + df['strategy_return']).cumprod()
        running_max = cumulative_return.cummax()
        drawdown = (cumulative_return / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * df['strategy_return'].mean() / df['strategy_return'].std()
        
        # Calculate hit ratio (% of profitable trades)
        trades = df[df['position'] != 0]
        if len(trades) > 0:
            hit_ratio = len(trades[trades['strategy_return'] > 0]) / len(trades)
        else:
            hit_ratio = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'hit_ratio': hit_ratio,
            'trade_count': len(trades)
        }
    
    def generate_all_strategies(self) -> Dict:
        """
        Generate all available strategies and evaluate their performance.
        
        Returns:
            Dictionary containing all strategies and their performance metrics
        """
        # Generate strategies
        ma_strategy = self.generate_ma_crossover_strategy()
        rsi_strategy = self.generate_rsi_strategy()
        bb_strategy = self.generate_bollinger_bands_strategy()
        volume_strategy = self.generate_volume_breakout_strategy()
        macd_strategy = self.generate_macd_strategy()
        
        # Evaluate strategies
        strategies = {
            'ma_crossover': {
                'signals': ma_strategy['signals'],
                'params': ma_strategy['params'],
                'metrics': self.evaluate_strategy(ma_strategy['signals'])
            },
            'rsi': {
                'signals': rsi_strategy['signals'],
                'params': rsi_strategy['params'],
                'metrics': self.evaluate_strategy(rsi_strategy['signals'])
            },
            'bollinger_bands': {
                'signals': bb_strategy['signals'],
                'params': bb_strategy['params'],
                'metrics': self.evaluate_strategy(bb_strategy['signals'])
            },
            'volume_breakout': {
                'signals': volume_strategy['signals'],
                'params': volume_strategy['params'],
                'metrics': self.evaluate_strategy(volume_strategy['signals'])
            },
            'macd': {
                'signals': macd_strategy['signals'],
                'params': macd_strategy['params'],
                'metrics': self.evaluate_strategy(macd_strategy['signals'])
            }
        }
        
        return strategies
    
    def print_strategy_metrics(self, strategies: Dict) -> None:
        """
        Print the performance metrics for all strategies.
        
        Args:
            strategies: Dictionary of strategies and their metrics
        """
        print("\n===== Strategy Performance Metrics =====\n")
        for name, strategy in strategies.items():
            metrics = strategy['metrics']
            params = strategy['params']
            
            print(f"Strategy: {params['strategy_name']}")
            print(f"Parameters: {params['parameters']}")
            print(f"Description: {params['description']}")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Annual Return: {metrics['annual_return']:.2%}")
            print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Hit Ratio: {metrics['hit_ratio']:.2%}")
            print(f"Number of Trades: {metrics['trade_count']}")
            print("-" * 50)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with OHLCV data
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df


def main():
    """Main function to run the strategy generator."""
    # Load OHLCV data
    data_path = 'data/ohlcv_AAPL.csv'
    try:
        df = load_data(data_path)
        print(f"Loaded data from {data_path}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Number of data points: {len(df)}")
        
        # Create strategy generator
        strategy_gen = StrategyGenerator(df)
        
        # Generate and evaluate all strategies
        strategies = strategy_gen.generate_all_strategies()
        
        # Print strategy metrics
        strategy_gen.print_strategy_metrics(strategies)
        
        print("\nStrategies generated successfully. Use these signals with a backtesting framework.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
