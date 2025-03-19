import pandas as pd
import numpy as np

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads stock data, converts Date column to datetime, and sorts it.

    Parameters:
    file_path (str): Path to the stock data CSV file.

    Returns:
    pd.DataFrame: Cleaned stock data.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Convert Date column to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
    else:
        raise KeyError("'Date' column is missing in the dataset.")

    # Sort by Company & Date
    df = df.sort_values(by=["Company", "Date"])

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Warning: Missing values detected!")
        print(missing_values[missing_values > 0])
        df = df.dropna()  # Drop missing values

    return df

def main():
    """ Main function to load and display cleaned stock data. """
    file_path = "data/stock_details_5_years.csv"


    try:
        stock_data = load_and_clean_data(file_path)
        print("Data loaded and cleaned successfully!")
        print(stock_data.head())  # Display sample data
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
