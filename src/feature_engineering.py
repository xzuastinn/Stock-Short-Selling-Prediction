import os
import pandas as pd
from data_processing import load_and_clean_data


def calculate_sma(df, window=20):
    """Calculates Simple Moving Average (SMA)"""
    df[f'SMA_{window}'] = df.groupby("Company")['Close'].transform(lambda x: x.rolling(window=window).mean())
    return df

def calculate_ema(df, window=20):
    """Calculates Exponential Moving Average (EMA)"""
    df[f'EMA_{window}'] = df.groupby("Company")['Close'].transform(lambda x: x.ewm(span=window, adjust=False).mean())
    return df

def calculate_rsi(df, window=14):
    """Calculates Relative Strength Index (RSI)"""
    delta = df.groupby("Company")['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20):
    """Calculates Bollinger Bands (Upper and Lower)"""
    rolling_mean = df.groupby("Company")['Close'].transform(lambda x: x.rolling(window=window).mean())
    rolling_std = df.groupby("Company")['Close'].transform(lambda x: x.rolling(window=window).std())
    df[f'Bollinger_Upper_{window}'] = rolling_mean + (rolling_std * 2)
    df[f'Bollinger_Lower_{window}'] = rolling_mean - (rolling_std * 2)
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculates Moving Average Convergence Divergence (MACD)"""
    short_ema = df.groupby("Company")['Close'].transform(lambda x: x.ewm(span=short_window, adjust=False).mean())
    long_ema = df.groupby("Company")['Close'].transform(lambda x: x.ewm(span=long_window, adjust=False).mean())
    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df.groupby("Company")['MACD'].transform(lambda x: x.ewm(span=signal_window, adjust=False).mean())
    return df

def calculate_volatility(df, window=10):
    """Calculates Rolling Volatility"""
    df[f'Volatility_{window}'] = df.groupby("Company")['Close'].transform(lambda x: x.pct_change().rolling(window=window).std())
    return df

def engineer_features(df):
    """Applies all feature engineering functions to dataset."""
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)
    df = calculate_volatility(df)
    return df

def main():
    """Main function to load, process, and save the dataset with new features."""
    file_path = "../stock_details_5_years.csv"  # Adjust path if needed
    output_path = "../data/stock_with_features.csv"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = load_and_clean_data(file_path)  # Use cleaned data from preprocessing
    df = engineer_features(df)
    df.to_csv(output_path, index=False)
    print("Feature engineering complete! Data saved to", output_path)

if __name__ == "__main__":
    main()
