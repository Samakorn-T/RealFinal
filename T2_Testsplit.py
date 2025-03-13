import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
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

# **Step 4: Train XGBoost using K-Fold Cross-Validation**
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
logger.info("Training XGBoost with Cross-Validation...")

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

bayes_search_cv.fit(X_scaled, y.values.ravel())
best_model_cv = bayes_search_cv.best_estimator_
logger.info(f"Best parameters from CV: {bayes_search_cv.best_params_}")

# Evaluate on full dataset
y_pred_cv = best_model_cv.predict(X_scaled)
mae_cv = mean_absolute_error(y, y_pred_cv)
mse_cv = mean_squared_error(y, y_pred_cv)
r2_cv = r2_score(y, y_pred_cv)
logger.info(f"Cross-Validation Results - R²: {r2_cv:.4f}, MAE: {mae_cv:.4f}, MSE: {mse_cv:.4f}")

# **Step 5: Train XGBoost with 80/20 Split**
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
logger.info("Training XGBoost with 80/20 Train-Test Split...")

bayes_search_split = BayesSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1),
    search_spaces=param_grid,
    scoring='r2',
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

bayes_search_split.fit(X_train, y_train.values.ravel())
best_model_split = bayes_search_split.best_estimator_
logger.info(f"Best parameters from 80/20 split: {bayes_search_split.best_params_}")

# Evaluate on test data
y_pred_test = best_model_split.predict(X_test)
mae_split = mean_absolute_error(y_test, y_pred_test)
mse_split = mean_squared_error(y_test, y_pred_test)
r2_split = r2_score(y_test, y_pred_test)
logger.info(f"80/20 Split Results - R²: {r2_split:.4f}, MAE: {mae_split:.4f}, MSE: {mse_split:.4f}")

# **Step 6: Save Models and Metrics**
output_folder = "T2_Folder"
os.makedirs(output_folder, exist_ok=True)

cv_model_path = os.path.join(output_folder, "XGBoost_CV.pkl")
split_model_path = os.path.join(output_folder, "XGBoost_Split.pkl")
joblib.dump(best_model_cv, cv_model_path)
joblib.dump(best_model_split, split_model_path)
logger.info(f"Saved CV model to {cv_model_path}")
logger.info(f"Saved Split model to {split_model_path}")
