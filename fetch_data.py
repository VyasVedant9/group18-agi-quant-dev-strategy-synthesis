import yfinance as yf
import pandas as pd
import os
from time import sleep
from datetime import datetime

def fetch_ohlcv(ticker_symbol, start_date, end_date, max_retries=3):
    """
    Fetch OHLCV data for a given stock symbol with retry logic.
    
    Args:
        ticker_symbol (str): Stock symbol (e.g., "AAPL")
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data
        
    Raises:
        ValueError: If ticker is invalid or data is empty
        RuntimeError: If Yahoo Finance fails after max retries
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # Validate ticker exists by checking basic info
    try:
        info = ticker.info
        if not info:
            raise ValueError(f"Ticker symbol '{ticker_symbol}' not found on Yahoo Finance")
    except Exception as e:
        raise ValueError(f"Error validating ticker '{ticker_symbol}': {str(e)}")
    
    retries = 0
    while retries < max_retries:
        try:
            print(f"Fetching data for {ticker_symbol} (attempt {retries + 1}/{max_retries})...")
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data available for {ticker_symbol} between {start_date} and {end_date}")
                
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return df
        
        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"Attempt failed, retrying in {retries} seconds...")
                sleep(retries)
            else:
                raise RuntimeError(f"Failed to fetch data after {max_retries} attempts. Last error: {str(e)}")

def save_to_csv(df, ticker_symbol, output_dir="./data"):
    """
    Save OHLCV data to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data
        ticker_symbol (str): Stock symbol (e.g., "AAPL")
        output_dir (str): Output directory path
        
    Returns:
        str: Path to the saved CSV file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"ohlcv_{ticker_symbol}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")
    
    return filepath

def main():
    # Example configuration - you could move these to config files or environment variables
    ticker_symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    print(f"Fetching OHLCV data for {ticker_symbol} from {start_date} to {end_date}")
    
    try:
        # Fetch data
        df = fetch_ohlcv(ticker_symbol, start_date, end_date)
        
        # Print preview
        print("\nData preview:")
        print(df.head())
        print(f"\nTotal rows fetched: {len(df)}")
        
        # Save to CSV
        save_to_csv(df, ticker_symbol)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()