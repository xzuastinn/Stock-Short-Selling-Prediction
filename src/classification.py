import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ---------------------------
# PyTorch imports
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def train_pytorch_dnn(X_train, y_train, epochs=10, batch_size=32):
    """
    Trains a simple PyTorch DNN for binary classification.
    Returns the trained model.
    """
    # Convert NumPy arrays to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    # Create a PyTorch Dataset & DataLoader
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define a simple feedforward network
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # (Optional) Print epoch loss:
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def prepare_classification_data(df):
    """Prepares data for classification models."""
    
    # Define classification target: Will the stock drop by ≥5% in 5 days?
    df["Short_Label"] = (df["Close"].shift(-5) < df["Close"] * 0.95).astype(int)
    df = df.dropna()  # Drop NaN values caused by shifting
    
    # Select features
    feature_cols = [col for col in df.columns if col not in ["Date", "Company", "Close", "Short_Label"]]
    X = df[feature_cols].values
    y = df["Short_Label"].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-based Train-Test Split (first half train, second half test)
    split_index = int(len(df) * 0.5)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test, scaler


def train_classification_models(X_train, y_train):
    """
    Trains:
      - LogisticRegression
      - SVM
      - PyTorch DNN
    Returns a dictionary of model_name -> model_object
    """
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Support Vector Machine
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)

    # PyTorch DNN
    pytorch_dnn = train_pytorch_dnn(X_train, y_train, epochs=10, batch_size=32)

    # Return all three in a dict
    return {
        "LogisticRegression": log_reg,
        "SVM": svm_classifier,
        "PyTorchDNN": pytorch_dnn
    }


def evaluate_classification(models, X_test, y_test):
    """
    Evaluates all models, prints metrics, returns a dict of model_name -> metric dict.
    """
    results = {}
    
    # Convert test data to torch tensor for PyTorch predictions
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    for name, model in models.items():
        if name == "PyTorchDNN":
            # PyTorch prediction
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_torch).numpy()  # shape: (n_samples, 1)
            y_pred = (outputs > 0.5).astype(int).flatten()
        else:
            # scikit-learn prediction
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        }
        
        print(f"\n{name} Classification Metrics:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")

    return results


def save_best_model(models, results):
    """Saves best model (highest F1) using joblib to ../models/ folder."""
    best_model_name = max(results, key=lambda x: results[x]["F1-Score"])
    best_model = models[best_model_name]

    # NOTE: You cannot directly joblib.dump a PyTorch model unless
    # you convert it to a state_dict or handle it differently.
    # For scikit-learn, joblib works fine. For PyTorch, see below.

    if best_model_name == "PyTorchDNN":
        # For PyTorch, we typically save model.state_dict() instead:
        import os
        os.makedirs("../models", exist_ok=True)
        torch.save(best_model.state_dict(), "../models/best_pytorch_model.pt")
        print("Best Classification Model is PyTorchDNN, saved state_dict to ../models/best_pytorch_model.pt")
    else:
        # scikit-learn model
        joblib.dump(best_model, "../models/best_classification_model.pkl")
        print(f"Best Classification Model: {best_model_name} saved to ../models/best_classification_model.pkl")


def main():
    print("Loading data...")
    # Load feature-engineered CSV (no cleaning needed if already done)
    df = pd.read_csv("data/stock_with_features.csv")

    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_classification_data(df)
    
    print("Training classification models...")
    models = train_classification_models(X_train, y_train)
    
    print("Evaluating classification models...")
    results = evaluate_classification(models, X_test, y_test)
    
    print("Saving best model...")
    save_best_model(models, results)
    print("Classification training complete!")


if __name__ == "__main__":
    main()
