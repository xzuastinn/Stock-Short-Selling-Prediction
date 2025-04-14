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
    df["Future_Drop%"] = ((df["Close"] - df["Close"].shift(-1)) / df["Close"]) * 100
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
    df["Future_Drop%"] = ((df["Close"] - df["Close"].shift(-1)) / df["Close"]) * 100
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

def train_dnn_regressor(X_train, y_train, X_test=None, y_test=None, learning_rate=0.005, epochs=100, batch_size=64):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    if X_test is not None and y_test is not None:
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    criterion = nn.HuberLoss(delta=2.0)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)

    # Track metrics
    train_losses = []
    train_r2s = []
    test_r2s = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        y_train_preds = []

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            y_train_preds.extend(outputs.detach().numpy())

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        # Compute R² on training set
        y_train_preds_np = np.array(y_train_preds).flatten()
        train_r2 = r2_score(y_train, y_train_preds_np)
        train_r2s.append(train_r2)

        # Compute R² on test set (optional)
        if X_test is not None:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t).numpy().flatten()
                test_r2 = r2_score(y_test, test_preds)
                test_r2s.append(test_r2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train R²: {train_r2:.4f}", end="")
            if X_test is not None:
                print(f" | Test R²: {test_r2:.4f}")
            else:
                print()

    return model, train_losses, train_r2s, test_r2s

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

def main(horizons=[], learning_rates=[], batch_sizes=[]):
    if not horizons or not learning_rates or not batch_sizes:
        raise ValueError("You must provide non-empty lists for horizons, learning_rates, and batch_sizes.")

    df = pd.read_csv("data/stock_with_features.csv")
    results_by_model = {}
    dnn_training_stats = {}

    for h in horizons:
        print(f"\n=== Running regression for horizon={h} ===")
        predicted_drops = pd.read_csv(f"data/predicted_drops_h{h}.csv")

        print("Preparing training dataset...")
        X_train, y_train, df_full, scaler = prepare_regression_data(df)

        print("Preparing test dataset from classifier predictions...")
        X_test, y_test = prepare_test_data(df, predicted_drops, scaler)

        # ----------------------------
        # Grid search DNN
        # ----------------------------
        best_dnn = None
        best_dnn_results = None
        best_mae = float("inf")
        best_config = None
        best_train_losses = []

        for lr in learning_rates:
            for bs in batch_sizes:
                print(f"\nTraining DNN with lr={lr}, batch_size={bs}")
                dnn, train_losses, train_r2s, test_r2s = train_dnn_regressor(X_train, y_train, X_test, y_test, 
                                                                             epochs=100, batch_size=bs, learning_rate=lr)
                results = evaluate_regression({"PyTorchDNN": dnn}, X_test, y_test)["PyTorchDNN"]

                if results["MAE"] < best_mae:
                    best_mae = results["MAE"]
                    best_dnn = dnn
                    best_dnn_results = results
                    best_config = {"lr": lr, "batch_size": bs}
                    best_train_losses = train_losses
                    best_train_r2s = train_r2s
                    best_test_r2s = test_r2s

        print(f"\nBest DNN config for horizon={h}: {best_config} with MAE={best_mae:.4f}")
        torch.save(best_dnn.state_dict(), f"../models/best_dnn_regressor_h{h}.pt")

        # Save DNN training stats
        dnn_training_stats[h] = {
            "train_losses": best_train_losses,
            "train_r2s": best_train_r2s,
            "test_r2s": best_test_r2s,
            "config": best_config
        }

        # ----------------------------
        # Train and evaluate baseline models
        # ----------------------------
        print("\nTraining baseline models...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        all_models = {
            "LinearRegression": lr_model,
            "RandomForest": rf_model,
            "PyTorchDNN": best_dnn
        }

        print("\nEvaluating all models for horizon", h)
        evaluation = evaluate_regression(all_models, X_test, y_test)
        results_by_model[h] = evaluation

    print("\nAll horizons complete.")
    return {
        "results_by_model": results_by_model,
        "dnn_training_stats": dnn_training_stats
    }

if __name__ == "__main__":
    main()

