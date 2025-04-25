import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from ta import add_all_ta_features
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def load_data_from_csv(filepath):
    """Load data from the previously saved CSV file"""
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    return df

def calculate_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    df = df.copy()
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    
    # Moving averages
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['ma21'] = df['Close'].rolling(window=21).mean()
    
    # Bollinger Bands
    df['std20'] = df['Close'].rolling(20).std()
    df['upper_band'] = df['ma21'] + (df['std20'] * 2)
    df['lower_band'] = df['ma21'] - (df['std20'] * 2)
    
    # Momentum indicators
    df['momentum'] = df['Close'] - df['Close'].shift(4)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(21)
    
    # Volume features
    df['volume_ma'] = df['Volume'].rolling(window=5).mean()
    df['volume_spike'] = (df['Volume'] / df['volume_ma'] - 1)
    
    # Using ta library for additional indicators
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    
    # Drop columns that are not useful or cause leakage
    df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True, errors='ignore')
    
    return df

def create_target_variable(df, horizon=5):
    """Create target variable - future returns"""
    df['target'] = (df['Close'].shift(-horizon) / df['Close'] - 1)
    return df

def clean_data(df):
    """Clean the data by removing NaN values and infinite values"""
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns with too many NaN values
    df = df.dropna(axis=1, thresh=len(df)*0.7)
    
    # Drop remaining rows with NaN values
    df = df.dropna()
    
    return df

def prepare_features(df):
    """Prepare features for modeling"""
    # Drop the original price/volume columns to avoid leakage
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'target']
    features = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return features

def train_xgboost(X_train, y_train):
    """Train XGBoost model with cross-validation"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\nModel Evaluation:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Plot feature importance
    xgb.plot_importance(model, max_num_features=20)
    plt.show()
    
    return predictions

def generate_forecast(model, latest_features, latest_price, horizon=5):
    """Generate future forecasts"""
    forecast_returns = []
    current_features = latest_features.copy()
    
    for _ in range(horizon):
        pred_return = model.predict(current_features)[0]
        forecast_returns.append(pred_return)
        
        # Update the features with the predicted return
        # Note: In a real implementation, you'd need to properly update all technical indicators
        # This is a simplified approach for demonstration
        current_features = current_features * (1 + pred_return)
    
    # Convert returns to price predictions
    forecast_prices = [latest_price]
    for ret in forecast_returns:
        forecast_prices.append(forecast_prices[-1] * (1 + ret))
    
    return forecast_prices[1:]  # Skip the first element (current price)

def main():
    # Load data from the CSV file we saved earlier
    csv_filepath = "./data/ohlcv_AAPL.csv"
    print(f"Loading data from {csv_filepath}...")
    df = load_data_from_csv(csv_filepath)
    
    # Prepare data
    print("\nPreparing data...")
    df = calculate_technical_indicators(df)
    df = create_target_variable(df, horizon=5)
    df = clean_data(df)  # This will remove NaN values
    
    # Split data (time-series aware split)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Prepare features and target
    X_train = prepare_features(train_df)
    y_train = train_df['target']
    X_test = prepare_features(test_df)
    y_test = test_df['target']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining model...")
    model = train_xgboost(X_train_scaled, y_train)
    
    # Evaluate model
    predictions = evaluate_model(model, X_test_scaled, y_test)
    
    # Generate forecast
    print("\nGenerating 1-week forecast...")
    
    # Get the latest data point for forecasting
    latest_data = df.iloc[-1:]
    latest_features = prepare_features(latest_data)
    latest_features_scaled = scaler.transform(latest_features)
    latest_price = latest_data['Close'].iloc[-1]
    
    forecast_prices = generate_forecast(model, latest_features_scaled, latest_price)
    
    print("\n1-week price forecast:")
    for i, price in enumerate(forecast_prices, 1):
        print(f"Day {i}: ${price:.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Close'], label='Actual Prices')
    
    # Create dates for forecast period
    last_date = test_df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5)
    
    plt.plot(forecast_dates, forecast_prices, 'ro-', label='Forecasted Prices')
    plt.title('AAPL Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()