import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ---------------------------
# PyTorch imports
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cpu")


def prepare_regression_data(df):
    """
    Prepares data for regression models
    """
    df["Future_Close"] = df["Close"].shift(-5)
    df["Future_Drop%"] = ((df["Close"] - df["Future_Close"]) / df["Close"]) * 100
    df = df.dropna()

    feature_cols = [col for col in df.columns if col not in ["Date", "Company", "Close", "Future_Close", "Future_Drop%"]]
    X = df[feature_cols].values
    y = df["Future_Drop%"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # time-based split 
    split_index = int(len(df) * 0.5)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test, scaler



def train_regression_dnn(X_train, y_train, epochs=1000, batch_size=32):
    """
    Trains a PyTorch DNN for regression.
    Returns the trained model.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return model


def evaluate_regression_dnn(model, X_test, y_test):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRegression Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return {"MAE": mae, "MSE": mse, "R2": r2}



def save_regression_model(model, path="../models/best_regression_model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Best regression model saved to {path}")


def main():
    print("Loading data...")
    df = pd.read_csv("data/stock_with_features.csv")

    print("Preparing data for regression...")
    X_train, X_test, y_train, y_test, scaler = prepare_regression_data(df)

    print("Training regression...")
    model = train_regression_dnn(X_train, y_train, epochs=1000, batch_size=32)

    print("Evaluating regression...")
    results = evaluate_regression_dnn(model, X_test, y_test)

    # print("Saving regression model...")
    # save_regression_model(model)

    print("Regression complete.")

if __name__ == "__main__":
    main()