import os
import pandas as pd
import numpy as np
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

def calculate_rate_of_change(df, window=10):
    """Calculates Rate of Change (ROC)"""
    df[f'ROC_{window}'] = df.groupby("Company")["Close"].transform(lambda x: x.pct_change(periods=window) * 100)
    return df

def calculate_atr(df, window=14):
    """Calculates Average True Range (ATR)"""
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df[f'ATR_{window}'] = df.groupby("Company")["TR"].transform(lambda x: x.rolling(window=window).mean())
    df.drop(columns=['High_Low', 'High_Close', 'Low_Close', 'TR'], inplace=True)
    return df

def calculate_obv(df):
    """Calculates On-Balance Volume (OBV)"""
    obv = []
    for company, group in df.groupby("Company"):
        group = group.sort_values("Date")
        obv_series = [0]
        for i in range(1, len(group)):
            if group["Close"].iloc[i] > group["Close"].iloc[i - 1]:
                obv_series.append(obv_series[-1] + group["Volume"].iloc[i])
            elif group["Close"].iloc[i] < group["Close"].iloc[i - 1]:
                obv_series.append(obv_series[-1] - group["Volume"].iloc[i])
            else:
                obv_series.append(obv_series[-1])
        df.loc[group.index, "OBV"] = obv_series
    return df

def add_lag_features(df, lags=[1, 3, 5]):
    for lag in lags:
        df[f'Lag_Close_Change_{lag}'] = df.groupby("Company")["Close"].pct_change(periods=lag) * 100
    return df

def calculate_drawdown(df):
    """Calculates rolling drawdown from the peak for each stock"""
    df['Rolling_Max'] = df.groupby("Company")['Close'].transform(lambda x: x.cummax())
    df['Drawdown_%'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max'] * 100
    df.drop(columns=['Rolling_Max'], inplace=True)
    return df

def calculate_volume_spike(df, window=10):
    df[f'Volume_Spike_{window}'] = df.groupby("Company")["Volume"].transform(
        lambda x: x / x.rolling(window).mean()
    )
    return df

def calculate_price_ma_ratio(df, window=50):
    sma = df.groupby("Company")["Close"].transform(lambda x: x.rolling(window).mean())
    df[f'Price_to_SMA_{window}'] = df["Close"] / sma
    return df

def calculate_volatility_ratio(df, short_window=5, long_window=20):
    short_vol = df.groupby("Company")['Close'].transform(lambda x: x.pct_change().rolling(short_window).std())
    long_vol = df.groupby("Company")['Close'].transform(lambda x: x.pct_change().rolling(long_window).std())
    df['Volatility_Ratio'] = short_vol / long_vol
    return df

def engineer_features(df):
    """Applies all feature engineering functions to dataset."""
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)
    df = calculate_volatility(df)
    df = calculate_obv(df)               
    df = add_lag_features(df, lags=[1, 3, 5])
    df = calculate_drawdown(df)
    df = calculate_volume_spike(df)
    df = calculate_price_ma_ratio(df)
    df = calculate_volatility_ratio(df)
    return df


def main():
    """Main function to load, process, and save the dataset with new features."""
    file_path = os.path.join("data", "stock_details_5_years.csv") 
    output_path = os.path.join("data", "stock_with_features.csv")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = load_and_clean_data(file_path)  # Use cleaned data from preprocessing
    df = engineer_features(df)
    df.to_csv(output_path, index=False)
    print("Feature engineering complete! Data saved to", output_path)

if __name__ == "__main__":
    main()
