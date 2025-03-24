import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **Step 1: Load Data**
data_folder = 'Prepare_datavisual'
data_file = 'Expanded_Fingerprints_Data.csv'
data_path = os.path.join(data_folder, data_file)
data = pd.read_csv(data_path)

# Extract input and output
X = data[[col for col in data.columns if col.startswith("FP_")]]  # Input features
y = data[["Bij"]]  # Target variable

# Metadata columns
metadata_cols = ["Name i", "Name j", "SMILES i", "SMILES j"]
metadata_df = data[metadata_cols].copy()
metadata_df["Actual_Bij"] = y  # Add actual values

# **Step 2: Load Models**
model_folder = "T5_Folder"
normal_model_path = os.path.join(model_folder, "XGBoost_Normal.pkl")
scaled_model_path = os.path.join(model_folder, "XGBoost_Scaled.pkl")

normal_model = joblib.load(normal_model_path)
scaled_model = joblib.load(scaled_model_path)

# Standardize input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Step 3: Define Evaluation Function**
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
    return {"Model": model_name, "R²": r2, "MAE": mae, "MSE": mse}

# **Step 4: Get Predictions for Both Models**
metadata_df["XGBoost_Normal_Predict"] = normal_model.predict(X_scaled)
metadata_df["XGBoost_Scaled_Predict"] = scaled_model.predict(X_scaled) * 303.15  # Convert back to normal scale

normal_metrics = evaluate_model(y, metadata_df["XGBoost_Normal_Predict"], "XGBoost_Normal")
scaled_metrics = evaluate_model(y, metadata_df["XGBoost_Scaled_Predict"], "XGBoost_Scaled")

# **Step 5: Save All Predictions in One CSV**
output_csv_path = os.path.join(model_folder, "Predictions_All.csv")
metadata_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved: {output_csv_path}")

# **Step 6: Plot Results for Both Models**
def plot_results(y_true, y_pred, model_name, color):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color=color, label=f"Predicted ({model_name})")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Bij")
    plt.ylabel("Predicted Bij")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.text(0.95, 0.95, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment="top",
             horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.legend()
    plt.show()

plot_results(y, metadata_df["XGBoost_Normal_Predict"], "XGBoost_Normal", "blue")
plot_results(y, metadata_df["XGBoost_Scaled_Predict"], "XGBoost_Scaled", "blue")
