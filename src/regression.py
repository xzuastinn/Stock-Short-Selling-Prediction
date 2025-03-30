import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def prepare_regression_data(df):
    """Prepares data for regression models."""
    
    # Define future percentage drop over the next 10 trading days
    df["Future_Drop%"] = ((df["Close"] - df["Close"].shift(-10)) / df["Close"]) * 100

    # Create Short_Label to match classification logic (drop ≥5%)
    df["Short_Label"] = (df["Future_Drop%"] >= 5).astype(int)

    # Keep only rows where we expect a drop (Short_Label = 1)
    df = df[df["Short_Label"] == 1].dropna()

    # Select features
    feature_cols = [col for col in df.columns if col not in ["Date", "Company", "Close", "Short_Label", "Future_Drop%"]]
    X = df[feature_cols].values
    y = df["Future_Drop%"].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (time-based)
    split_index = int(len(df) * 0.5)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test, scaler



def train_dnn_regressor(X_train, y_train, epochs=100, batch_size=32):
    """
    Trains a PyTorch DNN regressor.
    """
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(loader):.4f}")

    return model


def train_regression_models(X_train, y_train):
    """
    Trains regression models.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    dnn = train_dnn_regressor(X_train, y_train)

    return {
        "LinearRegression": lr,
        "RandomForest": rf,
        "PyTorchDNN": dnn
    }


def evaluate_regression(models, X_test, y_test):
    """
    Evaluates regression models.
    """
    results = {}
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    for name, model in models.items():
        if name == "PyTorchDNN":
            model.eval()
            with torch.no_grad():
                preds = model(X_test_torch).numpy().flatten()
        else:
            preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        }

        print(f"\n{name} Regression Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")

    return results


def main():
    print("Loading data...")
    df = pd.read_csv("data/stock_with_features.csv")

    print("Preparing regression dataset...")
    X_train, X_test, y_train, y_test, _ = prepare_regression_data(df)

    print("Training regression models...")
    models = train_regression_models(X_train, y_train)

    print("Evaluating regression models...")
    results = evaluate_regression(models, X_test, y_test)

    print("\nRegression task complete!")


if __name__ == "__main__":
    main()