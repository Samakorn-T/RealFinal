#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:54:38 2024

@author: padprow
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from scipy.stats import uniform

# **Step 1: Load and Prepare Data**
data_folder = 'Prepare_datavisual'
data_file = 'Filtered_Fingerprints_Data.xlsx'
data_path = os.path.join(data_folder, data_file)
data = pd.read_excel(data_path)

# Define Input (Fingerprint bits) and Output (Targets)
X = data[[col for col in data.columns if col.startswith("FP_")]]  # Input
y = data[["Aij", "Bij", "Alpha"]]  # Output

# Standardize the input data (important for SVR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Step 2: Define a function for Hyperparameter Tuning with RandomizedSearchCV**
def tune_svr_random(X, y, target):
    print(f"Tuning SVR model for target: {target}")
    
    # Define the parameter distribution for SVR
    param_dist = {
        'kernel': ['rbf'],  # Kernel types
        'C': uniform(0.1, 100),  # Regularization parameter (uniform distribution)
        'epsilon': uniform(0.01, 0.2)  # Epsilon for tolerance
    }
    
    # Perform RandomizedSearchCV with Cross-Validation
    svr = SVR()
    random_search = RandomizedSearchCV(estimator=svr, param_distributions=param_dist, 
                                       scoring='neg_mean_squared_error', 
                                       n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X, y)
    
    print(f"Best parameters for {target}: {random_search.best_params_}")
    return random_search.best_estimator_

# **Step 3: Cross-Validation**
def evaluate_model_cv(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mse_scores = cross_val_score(model, X, y, scoring=make_scorer(mean_squared_error), cv=kf)
    r2_scores = cross_val_score(model, X, y, scoring=make_scorer(r2_score), cv=kf)
    return np.mean(mse_scores), np.mean(r2_scores)

# **Step 4: Tune Models, Train, and Evaluate**
targets = ["Aij", "Bij", "Alpha"]
models = {}
cv_results = {}

for target in tqdm(targets, desc="Tuning and Evaluating Models"):
    best_svr = tune_svr_random(X_scaled, y[target], target)
    models[target] = best_svr  # Save the tuned model
    
    # Cross-Validate the model
    mse_cv, r2_cv = evaluate_model_cv(best_svr, X_scaled, y[target], cv=5)
    print(f"Target: {target}, Cross-Validated MSE: {mse_cv:.4f}, Cross-Validated R^2: {r2_cv:.4f}")
    cv_results[target] = {"MSE": mse_cv, "R2": r2_cv}

# **Step 5: Create Separate Folders**
model_folder = "Tuned_SVR_Models"
plot_folder = "Tuned_SVR_Plots"

# Create directories if they don't exist
os.makedirs(model_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

# Save each model in the 'Tuned_SVR_Models' folder
for target, model in models.items():
    filename = os.path.join(model_folder, f"tuned_svr_model_{target}.pkl")
    joblib.dump(model, filename)
    print(f"Model for {target} saved to {filename}")

# **Step 6: Visualize Results**
for target in tqdm(targets, desc="Plotting Results"):
    y_pred = models[target].predict(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y[target], y_pred, alpha=0.7)
    plt.plot([y[target].min(), y[target].max()], 
             [y[target].min(), y[target].max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Tuned SVR Prediction for {target}")
    
    # Save plot to 'Tuned_SVR_Plots' folder
    plot_path = os.path.join(plot_folder, f"Tuned_SVR_Prediction_{target}.png")
    plt.savefig(plot_path)
    print(f"Plot for {target} saved to {plot_path}")
    
    plt.show()