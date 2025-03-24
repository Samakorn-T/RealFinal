import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib
import logging

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# **Step 2: Load and Prepare Data**
data_folder = 'Prepare_datavisual'
data_file = 'Expanded_Fingerprints_Data.csv'
data_path = os.path.join(data_folder, data_file)
data = pd.read_csv(data_path)

# Define Input (Fingerprint bits) and Output (Targets)
X = data[[col for col in data.columns if col.startswith("FP_")]]  # Input
y = data[["Bij"]]  # Output
y_scaled = y / 303.15  # Scaled Output for second model

# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Step 3: Define Hyperparameter Space**
param_grid = {
    'n_estimators': Integer(50, 500),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 20),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'gamma': Real(0, 1),
    'reg_alpha': Real(0, 1),
    'reg_lambda': Real(0.1, 10, prior='log-uniform')
}

# **Step 4: Train XGBoost using Cross-Validation**
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
logger.info("Training XGBoost with Cross-Validation...")

def train_xgboost(X, y, label):
    bayes_search_cv = BayesSearchCV(
        estimator=XGBRegressor(random_state=42, n_jobs=-1),
        search_spaces=param_grid,
        scoring='r2',
        n_iter=50,
        cv=kfold,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    bayes_search_cv.fit(X, y.values.ravel())
    best_model = bayes_search_cv.best_estimator_
    logger.info(f"Best parameters for {label}: {bayes_search_cv.best_params_}")
    
    # Evaluate on full dataset
    y_pred = best_model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    logger.info(f"{label} Results - R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    return best_model

# Train both models
model_normal = train_xgboost(X_scaled, y, "Normal Bij")
model_scaled = train_xgboost(X_scaled, y_scaled, "Scaled Bij (Bij/303.15)")

# **Step 5: Save Models**
output_folder = "T5_Folder"
os.makedirs(output_folder, exist_ok=True)

normal_model_path = os.path.join(output_folder, "XGBoost_Normal.pkl")
scaled_model_path = os.path.join(output_folder, "XGBoost_Scaled.pkl")
joblib.dump(model_normal, normal_model_path)
joblib.dump(model_scaled, scaled_model_path)
logger.info(f"Saved Normal model to {normal_model_path}")
logger.info(f"Saved Scaled model to {scaled_model_path}")