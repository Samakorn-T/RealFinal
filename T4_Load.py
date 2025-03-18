import pandas as pd
import os
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# **Step 2: Create Output Folder**
load_folder = "T4_Folder"
os.makedirs(load_folder, exist_ok=True)
plot_folder = os.path.join(load_folder, "Plots")
os.makedirs(plot_folder, exist_ok=True)

# **Step 3: Load Data**
data_folder = "Prepare_datavisual"
data_file = "Expanded_Fingerprints_Data_Cleaned.csv"
data_path = os.path.join(data_folder, data_file)
data = pd.read_csv(data_path)

# Extract required columns
name_i = data["Name i"]
name_j = data["Name j"]
smiles_i = data["SMILES i"]
smiles_j = data["SMILES j"]
y_actual = data["Bij"]

# **Step 4: Load Preprocessing Tools and Models**
xgb_folder = "T4_Folder"
model_predictions = {"Actual": y_actual}

model_names = [
    f"best_xgboost_model_Expanded_Fingerprints_Data_Cleaned.pkl",
    f"best_xgboost_model_Expanded_Fingerprints_Data.pkl"
]

for model_name in model_names:
    dataset_name = model_name.replace("best_xgboost_model_", "").replace(".pkl", "")
    selected_features_path = os.path.join(xgb_folder, f"selected_features_{dataset_name}.txt")
    scaler_path = os.path.join(xgb_folder, f"scaler_{dataset_name}.pkl")
    model_path = os.path.join(xgb_folder, model_name)
    
    if os.path.exists(selected_features_path):
        with open(selected_features_path, "r") as f:
            selected_features = f.read().splitlines()
    else:
        logger.warning(f"Feature selection file {selected_features_path} not found! Using all features.")
        selected_features = [col for col in data.columns if col.startswith("FP_")]
    
    X_selected = data[selected_features]
    
    scaler = joblib.load(scaler_path)
    best_xgb = joblib.load(model_path)
    logger.info(f"Loaded model and scaler for {dataset_name}.")
    
    X_scaled = scaler.transform(X_selected)
    y_pred = best_xgb.predict(X_scaled)
    model_predictions[f"Model_{dataset_name}"] = y_pred
    
    # Compute Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    # Generate and save the Actual vs Predicted plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, alpha=0.7, label="Predicted vs Actual", color="blue")
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Bij", fontsize=12)
    plt.ylabel("Predicted Bij", fontsize=12)
    plt.title(f"XGBoost Prediction ({dataset_name})", fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    
    # Add metrics to plot
    plt.text(0.95, 0.05, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment="bottom",
             horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    
    plot_path = os.path.join(plot_folder, f"XGBoost_Prediction_{dataset_name}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved plot for {dataset_name} at {plot_path}")

# **Step 5: Save Predictions to CSV**
predictions_df = pd.DataFrame(model_predictions)
predictions_df.insert(0, "Name i", name_i)
predictions_df.insert(1, "Name j", name_j)
predictions_df.insert(2, "SMILES i", smiles_i)
predictions_df.insert(3, "SMILES j", smiles_j)

csv_path = os.path.join(load_folder, "XGBoost_Predictions_TwoModels.csv")
predictions_df.to_csv(csv_path, index=False)
logger.info(f"Predictions saved to {csv_path}")
