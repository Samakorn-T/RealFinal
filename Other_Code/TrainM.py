import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm  # สำหรับ progress bar

# **Step 1: Load and Prepare Data**
data_folder = 'Prepare_datavisual'
data_file = 'Expanded_Fingerprints_Data.xlsx'
data_path = os.path.join(data_folder, data_file)
data = pd.read_excel(data_path)

# Define Input (Fingerprint bits) and Output (Targets)
X = data[[col for col in data.columns if col.startswith("FP_")]]  # Input
y = data[["Aij", "Bij", "Alpha"]]  # Output

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input data (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Step 2: Define a function for Hyperparameter Tuning**
def tune_svr(X_train, y_train, target):
    print(f"Tuning SVR model for target: {target}")
    
    # Define the parameter grid for SVR
    param_grid = {
        'kernel': ['rbf'],  # Kernel types
        'C': [0.1, 1, 10, 100],       # Regularization parameter
        'epsilon': [0.01, 0.1, 0.2]   # Epsilon for tolerance
    }
    
    # Perform GridSearchCV
    svr = SVR()
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, 
                               scoring='neg_mean_squared_error', 
                               cv=5, n_jobs=-1, verbose=2)  # เพิ่ม verbose
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {target}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# **Step 3: Tune Models and Train**
targets = ["Aij", "Bij", "Alpha"]
models = {}
results = {}

# Progress bar for tuning and training
for target in tqdm(targets, desc="Tuning and Training Models"):
    best_svr = tune_svr(X_train_scaled, y_train[target], target)
    models[target] = best_svr  # Save the tuned model
    
    # Predict on test set
    y_pred = best_svr.predict(X_test_scaled)
    
    # Evaluate model
    mse = mean_squared_error(y_test[target], y_pred)
    r2 = r2_score(y_test[target], y_pred)
    
    print(f"Target: {target}, MSE: {mse:.4f}, R^2: {r2:.4f}")
    results[target] = {"MSE": mse, "R2": r2}

# **Step 4: Create Separate Folders**
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

# **Step 5: Visualize and Save Results**
for target in tqdm(targets, desc="Plotting Results"):
    y_pred = models[target].predict(X_test_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test[target], y_pred, alpha=0.7)
    plt.plot([y_test[target].min(), y_test[target].max()], 
             [y_test[target].min(), y_test[target].max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Tuned SVR Prediction for {target}")
    
    # Save plot to 'Tuned_SVR_Plots' folder
    plot_path = os.path.join(plot_folder, f"Tuned_SVR_Prediction_{target}.png")
    plt.savefig(plot_path)
    print(f"Plot for {target} saved to {plot_path}")
    
    plt.show()