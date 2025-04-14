# Stock Drop Prediction and Regression Pipeline

This project implements a two-stage machine learning pipeline to predict short-term stock price drops and estimate how much a drop will be using both classification and regression models.

---

## Overview

The program is divided into two main components:

### Classification (`classification.py`)

- Predicts whether a stock will drop by ≥5% in the next X trading days (user-defined horizon)
- Trains multiple models: Logistic Regression, SVM, Random Forest, and PyTorch DNN.
- Automatically selects and saves the best model based on **precision**.
- Outputs horizon-specific predictions to files like data/predicted_drops_h1.csv, data/predicted_drops_h3.csv, etc.

### Regression (`regression.py`)

- Estimates the percent drop for each horizon using predictions from classification.
- Trains and compares: Linear Regression, Random Forest, and PyTorch DNN. Selects best model based on R² per horizon.
- Evaluates performance using MAE, MSE, and R².
- Provides bin-wise summaries of actual vs. predicted drops.

---

## Folder Structure

```
short_selling_project/
│
├── data/
│   ├── stock_with_features.csv         # Input dataset (must be pre-engineered)
│   └── predicted_drops_hX.csv          # Output from classification (X = horizon)
│
├── models/
│   └── best_classification_model.pkl   # Saved model (or best_pytorch_model.pt for DNN)
│
├── src/
│   ├── classification.py               # Trains classifiers and outputs predictions
│   └── regression.py                   # Trains regressors on predicted drops
│
│
└── pipeline_runner.ipynb               # Jupyter notebook to run and visualize the full pipeline
|
└── README.md
```

---

## How It Works

### Classification

- Uses historical features (e.g., momentum, volatility).
- Defines short-term drops using a 5% threshold over user-defined time horizons.
- Handles class imbalance using a weighted loss function (for DNN).
- Evaluates using Accuracy, Precision, Recall, F1-Score.
- Selects and saves the best model by precision per horizon. Also performs hyperparameter tuning (learning rate, batch size) for DNN.
- Makes predictions on the second half of the dataset.

### Regression

- Uses horizon-specific prediction files (e.g., `predicted_drops_h1.csv`) as input.
- Trains on historical actual drops from the first half of the dataset.
- Evaluates using MAE, MSE, and R².
- Bins predictions and prints summaries comparing predicted vs. actual outcomes.
- Performs hyperparameter tuning for DNNs (learning rate, batch size).
- Compares DNN performance to baseline models (Linear, RF) and selects the best per horizon using R².

---

## How to Run

Make sure you have **Python 3.10+** and the required packages installed. Use Jupyter notebook to run and visualize the full pipeline. 

## Dependencies
pip install -r requirements.txt


### 1. Prepare the Data

Place your feature-engineered dataset in:

```
data/stock_with_features.csv
```

This dataset must include:
- Technical indicators
- Company names
- Dates
- Close prices

### 2. Run Classification

```bash
python src/classification.py
```

This will:
- Train and evaluate classification models
- Save the best model
- Output predicted drops to `data/predicted_drops.csv`

### 3. Run Regression

```bash
python src/regression.py
```

This will:
- Train regression models to estimate drop magnitude
- Output evaluation metrics and bin-wise summaries

---

## Dependencies

Install the required libraries:

```bash
pip install pandas numpy scikit-learn torch
```

If using a virtual environment or `pyenv`, activate it first.

---

## Example Outputs

- **Classification Precision**: ~0.68
- **Regression MAE**: as low as ~4.8 (varies by horizon)
- **Bin Summary**: Displays how predicted drops align with actual drop distributions
- **Best R² observed**: 0.4364
- **Best Model per Horizon**: PyTorchDNN (H=3,5), based on R²

---

## Notes

- Classification drop threshold is set to **5% over the specified horizon** (can be adjusted in `classification.py`).
- Regression models are trained on the first half of the dataset and tested on classifier-identified predicted drops from the second half.
- **Huber loss** is used in regression to reduce sensitivity to outliers.
- **Batch normalization + RMSprop** improve DNN training stability.
- DNNs handle class imbalance internally using weighted loss.

---
