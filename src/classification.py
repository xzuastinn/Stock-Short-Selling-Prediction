import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from data_processing import load_and_clean_data

def prepare_classification_data(df):
    """Prepares data for classification models."""
    
    # Define classification target: Will the stock drop by ≥5% in 5 days?
    df["Short_Label"] = (df["Close"].shift(-5) < df["Close"] * 0.95).astype(int)
    df = df.dropna()  # Drop NaN values caused by shifting
    
    # Select features (excluding raw price columns to prevent leakage)
    feature_cols = [col for col in df.columns if col not in ["Date", "Company", "Close", "Short_Label"]]
    X = df[feature_cols]
    y = df["Short_Label"]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # **Time-based Train-Test Split** (First half for training, second half for testing)
    split_index = int(len(df) * 0.5)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test, scaler

def train_classification_models(X_train, y_train):
    """Trains Logistic Regression, SVM, and DNN classification models."""
    
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    
    # Support Vector Machine (SVM)
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    
    # Deep Neural Network (DNN)
    dnn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dnn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    return {"LogisticRegression": log_reg, "SVM": svm_classifier, "DNN": dnn}

def evaluate_classification(models, X_test, y_test):
    """Evaluates classification models and prints performance metrics."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test) if name != "DNN" else (model.predict(X_test) > 0.5).astype(int)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        }
        print(f"\n{name} Classification Metrics:")
        for metric, value in results[name].items():
            print(f"{metric}: {value:.4f}")
    return results

def save_best_model(models, results):
    """Saves the best classification model based on highest F1-score."""
    best_model_name = max(results, key=lambda x: results[x]["F1-Score"])
    joblib.dump(models[best_model_name], "../models/best_classification_model.pkl")
    print(f"Best Classification Model: {best_model_name} saved!")

def main():
    """Main function to train and evaluate classification models."""
    print("Loading data...")
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