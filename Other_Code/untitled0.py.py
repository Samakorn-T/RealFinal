import pandas as pd
import os
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, DataStructs
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fsolve
from pubchempy import get_compounds

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.warning')

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Define Paths
model_folder = "XGB_Folder"

def smiles_to_fingerprint(smiles, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=2, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.tolist()
    return None

def get_compound_name(smiles):
    compounds = get_compounds(smiles, "smiles")
    return compounds[0].iupac_name if compounds else smiles

def load_model_and_scaler(target_name):
    model_path = os.path.join(model_folder, f"best_xgboost_model_{target_name}.pkl")
    scaler_path = os.path.join(model_folder, f"scaler_{target_name}.pkl")
    feature_path = os.path.join(model_folder, f"selected_features_{target_name}.txt")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.error(f"Missing model or scaler for {target_name}.")
        return None, None, None

    selected_features = None
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            selected_features = f.read().splitlines()
    else:
        logger.warning(f"Feature selection file {feature_path} not found!")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler, selected_features

def predict_values(smiles_i, smiles_j):
    fp_i, fp_j = smiles_to_fingerprint(smiles_i), smiles_to_fingerprint(smiles_j)
    if fp_i is None or fp_j is None:
        logger.error("Invalid SMILES input. Could not generate fingerprint.")
        return None
    
    feature_names = [f"FP_i_{i}" for i in range(2048)] + [f"FP_j_{i}" for i in range(2048)]
    X_input = pd.DataFrame([fp_i + fp_j], columns=feature_names)

    def get_model_predictions(X_input):
        results = {}
        for target in ["Aij", "Bij", "Alpha"]:
            model, scaler, selected_features = load_model_and_scaler(target)
            if not model or not scaler or not selected_features:
                continue
            X_selected = X_input[selected_features]
            X_scaled = scaler.transform(X_selected)
            results[target] = model.predict(X_scaled)[0]
        return results

    original_results = get_model_predictions(X_input)
    swapped_results = get_model_predictions(pd.DataFrame([fp_j + fp_i], columns=feature_names))
    
    final_results = {
        "Aij": original_results.get("Aij"), "Bij": original_results.get("Bij"), "Alpha": original_results.get("Alpha"),
        "Aji": swapped_results.get("Aij"), "Bji": swapped_results.get("Bij")
    }
    logger.info(f"Predictions: {final_results}")
    return final_results

def antoine_eq(A, B, C, T):
    return 10 ** (A - B / (T + C))

def gamma_nrtl(x1, x2, Aij, Aji, Bij, Bji, alpha, T):
    R = 8.314
    tau12 = (Aij + Bij / (T + 273.15)) / R
    tau21 = (Aji + Bji / (T + 273.15)) / R
    G12, G21 = np.exp(-alpha * tau12), np.exp(-alpha * tau21)
    gamma1 = np.exp(x2**2 * (tau21 * G21 / (x1 + x2 * G21))**2)
    gamma2 = np.exp(x1**2 * (tau12 * G12 / (x2 + x1 * G12))**2)
    return gamma1, gamma2

def vle_equation(T, x1, x2, Aij, Aji, Bij, Bji, alpha, A_i, B_i, C_i, A_j, B_j, C_j):
    gamma1, gamma2 = gamma_nrtl(x1, x2, Aij, Aji, Bij, Bji, alpha, T)
    Psat1, Psat2 = antoine_eq(A_i, B_i, C_i, T), antoine_eq(A_j, B_j, C_j, T)
    return x1 * gamma1 * Psat1 + x2 * gamma2 * Psat2 - 760

def plot_vle_manual(smiles_i, A_i, B_i, C_i, smiles_j, A_j, B_j, C_j):
    name_i, name_j = get_compound_name(smiles_i), get_compound_name(smiles_j)
    prediction = predict_values(smiles_i, smiles_j)
    if not prediction:
        return
    
    Aij, Aji, Bij, Bji, alpha = prediction["Aij"], prediction["Aji"], prediction["Bij"], prediction["Bji"], prediction["Alpha"]
    x1_values = np.linspace(0, 1, 20)
    y1_values, T_values = [], []

    for x1 in x1_values:
        x2 = 1 - x1
        T_solution = fsolve(vle_equation, 80, args=(x1, x2, Aij, Aji, Bij, Bji, alpha, A_i, B_i, C_i, A_j, B_j, C_j))[0]
        gamma1, gamma2 = gamma_nrtl(x1, x2, Aij, Aji, Bij, Bji, alpha, T_solution)
        Psat1, Psat2 = antoine_eq(A_i, B_i, C_i, T_solution), antoine_eq(A_j, B_j, C_j, T_solution)
        y1 = (x1 * gamma1 * Psat1) / (x1 * gamma1 * Psat1 + x2 * gamma2 * Psat2)
        y1_values.append(y1)
        T_values.append(T_solution)

    plt.figure(figsize=(8, 6))
    plt.plot(x1_values, T_values, label="Bubble Point", marker="o")
    plt.plot(y1_values, T_values, label="Dew Point", marker="s")
    plt.xlabel(f"Mole Fraction ({name_i})")
    plt.ylabel("Temperature (°C)")
    plt.title(f"T-xy Diagram {name_i} - {name_j}")
    plt.legend()
    plt.grid()
    plt.show()

plot_vle_manual("O", 8.07131, 1730.63, 233.426, "CCO", 8.20417, 1642.89, 230.300)