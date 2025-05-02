import pandas as pd
import os

# Input and output paths
input_path = 'Temp/Merged_Interaction.xlsx'
output_folder = 'Temp'
os.makedirs(output_folder, exist_ok=True)
output_file = 'Expanded_Interaction_Data.xlsx'
output_path = os.path.join(output_folder, output_file)

# Read the Excel file
print("Reading Excel Data...")
df = pd.read_excel(input_path)

# Ensure required columns are present
required_columns = [
    'CAS i', 'CAS j', 'Name i', 'Name j', 
    'SMILES i', 'SMILES j', 'Fingerprint i', 'Fingerprint j',
    'Aij', 'Aji', 'Bij', 'Bji', 'Alpha'
]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Some required columns are missing from the input file.")

# Create the swapped DataFrame
print("Creating swapped data...")
df_swapped = df.copy()

# Swap the relevant columns
df_swapped[['CAS i', 'CAS j']] = df[['CAS j', 'CAS i']].values
df_swapped[['Name i', 'Name j']] = df[['Name j', 'Name i']].values
df_swapped[['SMILES i', 'SMILES j']] = df[['SMILES j', 'SMILES i']].values
df_swapped[['Fingerprint i', 'Fingerprint j']] = df[['Fingerprint j', 'Fingerprint i']].values
df_swapped[['Aij', 'Aji']] = df[['Aji', 'Aij']].values
df_swapped[['Bij', 'Bji']] = df[['Bji', 'Bij']].values

# Combine the original and swapped DataFrames
print("Combining original and swapped data...")
df_combined = pd.concat([df, df_swapped], ignore_index=True)

# Save the combined data to a new Excel file
print(f"Saving combined data to {output_file}...")
df_combined.to_excel(output_path, index=False)
print(f"Data saved successfully to: {output_path}")

# Display final row count
print(f"Total rows after combining: {len(df_combined)}")