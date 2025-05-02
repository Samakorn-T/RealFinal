import pandas as pd

# Load the Excel file
file_path = "compounds.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Convert CID to numeric to ensure proper comparison
df['CID'] = pd.to_numeric(df['CID'], errors='coerce')

# Define the characters you want to filter out from the 'SMILES' column
filter_out_characters = ['+', '-', '.', '[', ']']

# Create a boolean mask to filter rows that do not contain any of the unwanted characters
mask = ~df['SMILES'].astype(str).str.contains(r'[\+\-\.\[\]]')

# Apply the filter
df_filtered = df[mask]

# Sort the filtered DataFrame by 'SMILES' and 'CID' (ascending order)
#df_sorted = df_filtered.sort_values(by=['SMILES', 'CID'])

# Drop duplicates, keeping the first entry (which is the one with the lowest CID)
df_unique_smiles = df_filtered.drop_duplicates(subset='SMILES', keep='first')

# Save the filtered data to a new Excel file
filtered_file_path = "filtered_compounds.xlsx"
df_unique_smiles.to_excel(filtered_file_path, index=False)