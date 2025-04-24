# You are given a CSV file containing historical daily stock data for a single ticker. The columns in the CSV are: Date, Open, High, Low, Close, and Volume. 
# Write Python code that builds a trading strategy or model to forecast the stock’s closing price 5 trading days (1 week) into the future. 
# You have complete freedom in choosing the approach—whether it’s a statistical method, machine learning model, or any other technique—as long as it 
# uses only the input CSV data and produces a robust market forecast aimed at maximizing predictive accuracy. You are encouraged to engineer any features you 
# find useful and should split the data into training and test sets using a fixed random seed. The model must take historical data as input and output a 
# 1-week-ahead forecast. You don’t need to implement a backtester yet, but ensure the forecast output is compatible with one.




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb # Requires installation: pip install lightgbm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time # To time execution

# --- Configuration ---
CSV_FILE_PATH = 'ohlcv_AAPL.csv'
FORECAST_HORIZON = 5 # Days ahead to forecast
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_SEED = 42 # For model reproducibility

# --- Function Definitions ---

def load_and_preprocess_data(file_path):
    """Loads data, handles dates, sorts, and fills initial NaNs."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    print("Preprocessing data...")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    initial_nulls = df.isnull().sum().sum()
    if initial_nulls > 0:
        print(f"Warning: Found {initial_nulls} missing values. Filling with forward fill.")
        df.fillna(method='ffill', inplace=True)
        # Handle any remaining NaNs at the beginning if ffill wasn't enough
        df.dropna(inplace=True)

    print(f"Data shape after initial load & preprocess: {df.shape}")
    return df

def engineer_features(df, forecast_out=5):
    """Engineers features and target variable, drops NaNs."""
    print("Engineering features...")
    df_feat = df.copy()

    # Target variable: Close price 'forecast_out' days ahead
    df_feat['Target'] = df_feat['Close'].shift(-forecast_out)

    # Simple Lag Features
    for lag in range(1, 6): # Lags from 1 to 5 days ago
        df_feat[f'Close_Lag_{lag}'] = df_feat['Close'].shift(lag)
        df_feat[f'Volume_Lag_{lag}'] = df_feat['Volume'].shift(lag)

    # Moving Averages
    df_feat['MA_5'] = df_feat['Close'].rolling(window=5).mean()
    df_feat['MA_20'] = df_feat['Close'].rolling(window=20).mean()

    # Volatility (Rolling Standard Deviation)
    df_feat['Volatility_20'] = df_feat['Close'].rolling(window=20).std()

    # Relative Strength Index (RSI) - common window is 14
    delta = df_feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feat['RSI_14'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['MACD'] = exp1 - exp2
    df_feat['MACD_Signal'] = df_feat['MACD'].ewm(span=9, adjust=False).mean()

    # Drop rows with NaNs created by feature engineering and target shift
    initial_rows = len(df_feat)
    df_feat.dropna(inplace=True)
    rows_dropped = initial_rows - len(df_feat)
    print(f"Dropped {rows_dropped} rows due to NaN values from feature engineering.")
    print(f"Data shape after feature engineering: {df_feat.shape}")

    return df_feat

def plot_forecast(index, y_true, y_pred, model_name, filename):
    """Generates and saves a plot comparing actual vs predicted values."""
    plt.figure(figsize=(14, 7))
    plt.plot(index, y_true, label='Actual 5-Day Ahead Close', color='blue', alpha=0.7)
    plt.plot(index, y_pred, label=f'Predicted 5-Day Ahead Close ({model_name})', color='red', linestyle='--')
    plt.title(f'AAPL 5-Day Ahead Close Price Forecast vs Actual ({model_name})')
    plt.xlabel('Date (Prediction Made)')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close() # Close the plot to free memory
    print(f"Plot saved as {filename}")


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # 1. Load and Preprocess
    df_raw = load_and_preprocess_data(CSV_FILE_PATH)
    if df_raw is None:
        exit() # Exit if data loading failed

    # 2. Feature Engineering
    df_processed = engineer_features(df_raw, forecast_out=FORECAST_HORIZON)
    if df_processed.empty:
        print("Error: No data left after feature engineering and NaN removal.")
        exit()

    # 3. Define Features (X) and Target (y)
    features = [col for col in df_processed.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    X = df_processed[features]
    y = df_processed['Target']
    print(f"\nUsing {len(features)} features: {features}")

    # 4. Split Data (Chronological Split)
    split_index = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print(f"\nData Split:")
    print(f"Training set size: {X_train.shape[0]} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
    print(f"Test set size: {X_test.shape[0]} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")

    # --- Model Training and Prediction Loop ---
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1, max_depth=10, min_samples_split=10),
        "Linear Regression": LinearRegression(n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, n_estimators=100, learning_rate=0.1, num_leaves=31),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1) # SVR requires scaling
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training and Evaluating {name} ---")
        model_start_time = time.time()

        # Handle scaling for SVR
        if name == "SVR":
            print("Scaling features for SVR...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # Train SVR on scaled data
            model.fit(X_train_scaled, y_train)
            # Predict on scaled data
            y_pred = model.predict(X_test_scaled)
        else:
            # Train other models on original data
            model.fit(X_train, y_train)
            # Predict on original data
            y_pred = model.predict(X_test)

        model_train_time = time.time() - model_start_time

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = rmse
        print(f"Model training time: {model_train_time:.2f} seconds")
        print(f"RMSE on Test Set: {rmse:.4f}")

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Actual_Future_Close': y_test,
            'Predicted_Future_Close': y_pred
        }, index=y_test.index)
        # Add actual close on prediction day for context
        forecast_df['Actual_Close_Today'] = df_processed.loc[y_test.index, 'Close']

        print(f"\n{name} Forecast Output (first 5 rows):")
        print(forecast_df.head())

        # Plot results
        plot_filename = f'figs/aapl_forecast_vs_actual_{name.lower().replace(" ", "_")}.png'
        plot_forecast(y_test.index, y_test, y_pred, name, plot_filename)

    # --- Summary ---
    print("\n--- Model Comparison (RMSE on Test Set) ---")
    for name, rmse in sorted(results.items(), key=lambda item: item[1]):
        print(f"{name}: {rmse:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds.")
    print("\nScript finished.")