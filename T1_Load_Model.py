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

# Extract necessary columns
X = data[[col for col in data.columns if col.startswith("FP_")]]  # Input features
y = data[["Bij"]]  # Target variable

# Metadata columns to keep
metadata_cols = ["Name i", "Name j", "SMILES i", "SMILES j"]
metadata_df = data[metadata_cols].copy()
metadata_df["Actual_Bij"] = y  # Add actual values

# **Step 2: Load Saved Models and Scaler**
model_folder = "T1_Folder/Models"
plot_folder = "T1_Folder/Plots"
output_csv_path = "T1_Folder/Predictions.csv"

os.makedirs(plot_folder, exist_ok=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Step 3: Generate Actual vs. Predicted Plots and Save Predictions**
predictions_df = metadata_df.copy()  # Start with metadata and actual values

for target in y.columns:
    for model_name in ["OLS", "DecisionTree", "SVR", "XGBoost"]:
        model_path = os.path.join(model_folder, f"{model_name}_{target}.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_scaled)
            
            # Add predictions to DataFrame
            predictions_df[f"{model_name}_Predict"] = y_pred
            
            # Calculate metrics
            mae = mean_absolute_error(y[target], y_pred)
            mse = mean_squared_error(y[target], y_pred)
            r2 = r2_score(y[target], y_pred)

            # **Step 9: Plot Actual vs Predicted**
            plt.figure(figsize=(8, 6))
            plt.scatter(y[target], y_pred, alpha=0.7, label="Predicted vs Actual", color="blue")
            plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], 'r--', label="Perfect Fit")
            plt.xlabel(f"Actual {target}", fontsize=12)
            plt.ylabel(f"Predicted {target}", fontsize=12)
            plt.title(f"{model_name} Prediction on Full Dataset ({target})", fontsize=14)
            plt.legend(loc="upper left", fontsize=10)

            # Add metrics as a text box
            plt.text(0.95, 0.95, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}",
                     transform=plt.gca().transAxes,
                     fontsize=10,
                     verticalalignment="top",
                     horizontalalignment="right",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

            # Save plot
            plot_path = os.path.join(plot_folder, f"Actual_vs_Predicted_{target}_{model_name}.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"Plot saved: {plot_path}")  # Debugging print statement

# **Step 4: Save Predictions to CSV**
predictions_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved in {output_csv_path}")