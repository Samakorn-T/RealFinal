#%% Merge Data
import os
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

# ปิดการแจ้งเตือน
RDLogger.DisableLog('rdApp.warning')
print("Merge Data")
# Phase 1: Read Original and Processed Data
print("Phase 1: Reading Data")

input_folder = 'Import'
input_file = 'Interaction.xlsx'
input_path = os.path.join(input_folder, input_file)


# Read the original file that contains CAS i and CAS j
df_original = pd.read_excel(input_path, usecols=['CAS i', 'CAS j', 'Aij', 'Aji', 'Bij', 'Bji', 'Alpha'])

# Read the processed file that contains CAS, Name, SMILES, and Fingerprint
temp_folder = 'Temp'
temp_file = 'Processed_Interaction.xlsx'
temp_path = os.path.join(temp_folder, temp_file)
df_processed = pd.read_excel(temp_path)

# Phase 2: Merge Processed Data with Original Data
print("Phase 2: Merging Data")

# Merge data for CAS i
df_merged_i = pd.merge(df_original, df_processed, how='left', left_on='CAS i', right_on='CAS', suffixes=('', '_i'))
df_merged_i = df_merged_i.drop(columns=['CAS'])  # Remove the redundant CAS column from processed data

# Rename columns for CAS i
df_merged_i = df_merged_i.rename(columns={
    'Name': 'Name i',
    'SMILES': 'SMILES i',
    'Fingerprint': 'Fingerprint i'
})

# Merge data for CAS j
df_merged = pd.merge(df_merged_i, df_processed, how='left', left_on='CAS j', right_on='CAS', suffixes=('', '_j'))
df_merged = df_merged.drop(columns=['CAS'])  # Remove the redundant CAS column from processed data

# Rename columns for CAS j
df_merged = df_merged.rename(columns={
    'Name': 'Name j',
    'SMILES': 'SMILES j',
    'Fingerprint': 'Fingerprint j'
})

# Reorganize columns in the desired order: CAS i, CAS j, Name i, Name j, SMILES i, SMILES j, Fingerprint i, Fingerprint j
df_merged = df_merged[['CAS i', 'CAS j', 'Name i', 'Name j', 'SMILES i', 'SMILES j', 'Fingerprint i', 'Fingerprint j', 'Aij', 'Aji', 'Bij', 'Bji', 'Alpha']]

# Phase 3: Save the Final Merged Data
print("Phase 3: Saving Merged Data")

temp_folder = 'Temp'
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
Merge_file = 'Merged_Interaction.xlsx'
Merge_path = os.path.join(temp_folder, Merge_file)
df_merged.to_excel(Merge_path, index=False)#%% Merge Data
