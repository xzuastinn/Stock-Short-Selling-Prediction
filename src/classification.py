import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader

#LR is best at .001 before .ipynb 
def train_dnn(X_train, y_train, X_test=None, y_test=None, learning_rate=0.001, epochs=100, batch_size=64):
    """
    Trains a PyTorch DNN for binary classification with class imbalance handling.
    Returns the trained model, training loss, and (optional) accuracy histories.
    """
    # Convert NumPy arrays to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    if X_test is not None and y_test is not None:
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    # Compute class imbalance weight
    num_negatives = np.sum(y_train == 0)
    num_positives = np.sum(y_train == 1)
    pos_weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Neural network
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss(weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # Track metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()

        train_acc = correct / total
        avg_loss = epoch_loss / len(dataloader)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        # Evaluate on test set if provided
        if X_test is not None:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_preds = (test_outputs > 0.5).float()
                test_correct = (test_preds == y_test_t).sum().item()
                test_acc = test_correct / len(y_test_t)
                test_accuracies.append(test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}", end="")
            if X_test is not None:
                print(f" | Test Acc: {test_acc:.4f}")
            else:
                print()

    return model, train_losses, train_accuracies, test_accuracies




def prepare_classification_data(df, horizon):
    """Prepares data for classification models."""
    
    # Define classification target: Will the stock drop by ≥5% in 5 days?
    df["Short_Label"] = (df["Close"].shift(-horizon) < df["Close"] * 0.95).astype(int)
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


def train_classification_models(X_train, y_train, learning_rate, batch_size):
    """
    Trains:
      - LogisticRegression
      - SVM
      - RandomForest
      - PyTorch DNN
    Returns a dictionary of model_name -> model_object
    """
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Support Vector Machine
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # PyTorch DNN
    pytorch_dnn = train_dnn(X_train, y_train, epochs=100, batch_size=batch_size, learning_rate=learning_rate)

    return {
        "LogisticRegression": log_reg,
        "SVM": svm_classifier,
        "RandomForest": rf_classifier,
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
            threshold = 0.4 ## CHANGED FROM .6 4/5/26 
            y_pred = (outputs > threshold).astype(int).flatten()
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
    best_model_name = max(results, key=lambda x: results[x]["Precision"])
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


def main(horizon=[], learning_rate=[], batch_size=[]):
    if not horizon or not learning_rate or not batch_size:
        raise ValueError("You must provide lists for horizon, learning_rate, and batch_size.")

    df = pd.read_csv("data/stock_with_features.csv")
    results_by_model = {}
    dnn_training_stats = {}

    for h in horizon:
        print(f"\n=== Running classification for horizon={h} ===")
        X_train, X_test, y_train, y_test, scaler = prepare_classification_data(df.copy(), h)

        # ----------------------
        # Find Best DNN for this horizon
        # ----------------------
        best_dnn = None
        best_dnn_results = None
        best_precision = -1
        best_config = None
        dnn_train_losses = dnn_train_accs = dnn_test_accs = None

        for lr in learning_rate:
            for bs in batch_size:
                print(f"\nTraining DNN with lr={lr}, batch_size={bs}...")
                model, train_losses, train_accs, test_accs = train_dnn(
                    X_train, y_train, X_test, y_test,
                    learning_rate=lr, batch_size=bs
                )

                metrics = evaluate_classification({"PyTorchDNN": model}, X_test, y_test)["PyTorchDNN"]
                if metrics["Precision"] > best_precision:
                    best_dnn = model
                    best_dnn_results = metrics
                    best_precision = metrics["Precision"]
                    best_config = {"lr": lr, "batch_size": bs}
                    dnn_train_losses = train_losses
                    dnn_train_accs = train_accs
                    dnn_test_accs = test_accs

        dnn_training_stats[h] = {
            "train_losses": dnn_train_losses,
            "train_accuracies": dnn_train_accs,
            "test_accuracies": dnn_test_accs,
            "config": best_config
            }

        print(f"\nBest DNN config for horizon={h}: {best_config} with precision={best_precision:.4f}")

        # ----------------------
        # Train non-DNN models ONCE
        # ----------------------
        print("\nTraining other models for comparison...")
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)

        svm_classifier = SVC()
        svm_classifier.fit(X_train, y_train)

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        models = {
            "LogisticRegression": log_reg,
            "SVM": svm_classifier,
            "RandomForest": rf_classifier,
            "PyTorchDNN": best_dnn
        }

        # ----------------------
        # Evaluate and Save
        # ----------------------
        results = evaluate_classification(models, X_test, y_test)
        save_best_model(models, results)

        # Save predicted drops
        test_df = df.iloc[int(len(df) * 0.5):].reset_index(drop=True)
        feature_cols = [col for col in test_df.columns if col not in ["Date", "Company", "Close", "Short_Label"]]
        X_test_scaled = scaler.transform(test_df[feature_cols])
        nan_mask = ~np.isnan(X_test_scaled).any(axis=1)
        X_test_scaled = X_test_scaled[nan_mask]
        test_df = test_df[nan_mask].reset_index(drop=True)

        best_model_name = max(results, key=lambda x: results[x]["Precision"])
        best_model = models[best_model_name]

        if best_model_name == "PyTorchDNN":
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            with torch.no_grad():
                probs = best_model(X_test_tensor)
                preds = (probs > 0.6).int().numpy().flatten()
        else:
            preds = best_model.predict(X_test_scaled)

        predicted_drops = test_df[preds == 1]
        print(f"\nBest model: {best_model_name}")
        print(f"Predicted drops found: {len(predicted_drops)}")
        predicted_drops.to_csv(f"data/predicted_drops_h{h}.csv", index=False)

        # ----------------------
        # Store results for notebook plotting
        # ----------------------
        for model_name, metrics in results.items():
            if model_name not in results_by_model:
                results_by_model[model_name] = {}
            results_by_model[model_name][h] = metrics

    return {
    "results_by_model": results_by_model,
    "dnn_training_stats": dnn_training_stats  
}


if __name__ == "__main__":
    main()