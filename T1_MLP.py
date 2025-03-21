import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import BayesianOptimization
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load Data
data_folder = 'Prepare_datavisual'
data_file = 'Expanded_Fingerprints_Data.csv'
data_path = os.path.join(data_folder, data_file)
data = pd.read_csv(data_path)

X = data[[col for col in data.columns if col.startswith("FP_")]]  # Input
y = data[["Bij"]]  # Output

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define Model for Bayesian Optimization
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_1', 500, 800, step=50), activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(hp.Int('units_2', 1500, 1800, step=50), activation='swish'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.4, step=0.05)))
    model.add(Dense(hp.Int('units_3', 800, 1000, step=50), activation='selu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.4, step=0.05)))
    model.add(Dense(hp.Int('units_4', 1200, 1400, step=50), activation='selu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_3', 0.1, 0.3, step=0.05)))
    model.add(Dense(hp.Int('units_5', 1500, 1700, step=50), activation='swish'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_4', 0.2, 0.4, step=0.05)))
    model.add(Dense(hp.Int('units_6', 500, 700, step=50), activation='selu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_5', 0.4, 0.6, step=0.05)))
    model.add(Dense(hp.Int('units_7', 1000, 1300, step=50), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_6', 0.0, 0.2, step=0.05)))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', 0.0005, 0.005, step=0.0005)), loss='mse')
    return model

# Bayesian Optimization
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=50,
    executions_per_trial=1,
    directory='T1_Folder',
    project_name='MLP_Bij_Tuning'
)

tuner.search(X_scaled, y, epochs=50, validation_split=0.2, verbose=1)

# Get Best Model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_scaled, y, epochs=100, validation_split=0.2, verbose=1)

# Save Best Model
model_folder = "T1_Folder/Models"
os.makedirs(model_folder, exist_ok=True)
model_path = os.path.join(model_folder, "MLP_Bij_Best_Model.h5")
best_model.save(model_path)
logger.info(f"Best MLP model saved to {model_path}")

# Evaluate Model
y_pred = best_model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
logger.info(f"Best Model Metrics - R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

# Save Metrics
metrics_folder = "T1_Folder/Metrics"
os.makedirs(metrics_folder, exist_ok=True)
metrics_path = os.path.join(metrics_folder, "MLP_Bij_Metrics.csv")
pd.DataFrame([{"MAE": mae, "MSE": mse, "R2": r2}]).to_csv(metrics_path, index=False)
logger.info(f"Metrics saved to {metrics_path}")
