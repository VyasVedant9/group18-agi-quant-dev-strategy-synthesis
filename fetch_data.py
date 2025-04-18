"""

Model Used: ChatGPT 40 
Prompt: Write a Python script that downloads daily OHLCV (Open, High, Low, Close, Volume) data for a given stock symbol (e.g., "AAPL") using the yfinance library over a specific date range. The script should:

Use yf.Ticker().history() instead of yf.download()

Include retry logic (up to 3 times) in case of failure

Validate that the resulting DataFrame contains the required columns: ['Open', 'High', 'Low', 'Close', 'Volume']

Save the data to a CSV file (e.g., ./data/ohlcv_AAPL.csv)

Print a preview of the data

Raise informative errors if the ticker is invalid, data is empty, or Yahoo Finance fails

Include helpful print statements to indicate progress

Avoid interactive input or hardcoded credentials

Follow good coding practices with well-structured functions for fetching and saving data.

"""


import yfinance as yf
import pandas as pd
import os
import time

# Configuration
TICKER = "AAPL"
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"
OUTPUT_DIR = "./data"
OUTPUT_FILE = f"ohlcv_{TICKER}.csv"
REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
MAX_RETRIES = 3


def fetch_ohlcv_data(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Fetches OHLCV data for a given ticker and date range using yfinance."""
    print(f"Starting download for {ticker} from {start} to {end}...")
    attempt = 0
    while attempt < retries:
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start, end=end, interval="1d")

            if df.empty:
                raise ValueError("No data returned from Yahoo Finance.")

            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing expected columns: {missing_cols}")

            print("Data download successful.")
            return df[REQUIRED_COLUMNS]  # Return only required columns

        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                print("Retrying...")
                time.sleep(2)  # Wait before retrying
            else:
                raise RuntimeError(f"Failed to fetch data for {ticker} after {retries} attempts.") from e


def save_to_csv(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")


def main():
    df = fetch_ohlcv_data(TICKER, START_DATE, END_DATE, MAX_RETRIES)
    print("Preview of downloaded data:")
    print(df.head())
    save_to_csv(df, os.path.join(OUTPUT_DIR, OUTPUT_FILE))


if __name__ == "__main__":
    main()