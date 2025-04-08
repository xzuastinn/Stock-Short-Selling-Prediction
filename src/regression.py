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
    """Prepares training data for regression models using actual historical drops in first half."""
    df["Future_Drop%"] = ((df["Close"] - df["Close"].shift(-3)) / df["Close"]) * 100
    df["Short_Label"] = (df["Future_Drop%"] >= 5).astype(int)
    df = df.dropna()

    # Keep only first half of data
    split_index = int(len(df) * 0.5)
    df_train = df.iloc[:split_index]

    # Filter only real drops uncomment to change
    #df_train = df_train[df_train["Short_Label"] == 1]

    feature_cols = [col for col in df_train.columns if col not in ["Date", "Company", "Close", "Short_Label", "Future_Drop%", 
                                                                   "Drawdown_%", "Volume_Spike_10", "Price_to_SMA_50", 
                                                                   "Volatility_Ratio"]]
    X = df_train[feature_cols].values
    y = df_train["Future_Drop%"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df, scaler

def prepare_test_data(df, predicted_drops, scaler):
    """Prepares test data based on classifier-predicted drops."""
    df["Future_Drop%"] = ((df["Close"] - df["Close"].shift(-3)) / df["Close"]) * 100
    df["Short_Label"] = (df["Future_Drop%"] >= 5).astype(int)
    df = df.dropna()

    df_test = df.merge(predicted_drops[["Date", "Company"]], on=["Date", "Company"], how="inner")

    feature_cols = [col for col in df_test.columns if col not in ["Date", "Company", "Close", 
                                                                  "Short_Label", "Future_Drop%",
                                                                    "Drawdown_%", "Volume_Spike_10", 
                                                                    "Price_to_SMA_50", "Volatility_Ratio"]]
    X_test = scaler.transform(df_test[feature_cols])
    y_test = df_test["Future_Drop%"].values

    return X_test, y_test

def train_dnn_regressor(X_train, y_train, epochs=100, batch_size=64):
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
    nn.Linear(X_train.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
    ) #CONTROL Z THIS AWAY

    criterion = nn.HuberLoss(delta=2.0) #changed delta from 1 hoping to reduce senstivity to noise..
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

        # Bin predictions
        bins = [0, 1, 3, 5, 7, 9, float("inf")]
        labels = ["<1%", "1–3%", "3–5%", "5–7%", "7–9%", "≥9%"]
        bin_indices = np.digitize(preds, bins)

        print("\nAverage actual drop (y_test) for each predicted bin:")
        for i, label in enumerate(labels, 1):
            actual_drops = y_test[bin_indices == i]
            actual_drops = actual_drops[actual_drops < 0]  # Only include actual drops CAN COMMENT OUT FOR ALL
            if len(actual_drops) > 0:
                avg_drop = -np.mean(actual_drops)  # Convert to positive for reporting
                print(f"  {label}: {avg_drop:.2f} (n = {len(actual_drops)})")
            else:
                print(f"  {label}: No actual drops in this bin")

    return results

def main():
    predicted_drops = pd.read_csv("data/predicted_drops.csv")

    print("Loading data...")
    df = pd.read_csv("data/stock_with_features.csv")

    print("Preparing training dataset...")
    X_train, y_train, df_full, scaler = prepare_regression_data(df)

    print("Training regression models...")
    models = train_regression_models(X_train, y_train)

    print("Preparing test dataset from classifier predictions...")
    X_test, y_test = prepare_test_data(df, predicted_drops, scaler)

    print("Evaluating regression models...")
    results = evaluate_regression(models, X_test, y_test)

    print("\nRegression task complete!")

if __name__ == "__main__":
    main()
