import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the expanded fingerprint data
input_folder = 'Prepare_datavisual'
input_file = 'Expanded_Fingerprints_Data.xlsx'
input_path = os.path.join(input_folder, input_file)

df = pd.read_excel(input_path)
#%%650,Aij

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_650', y='Aij', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_650 vs Aij', fontsize=16)
plt.xlabel('FP_i_650', fontsize=14)
plt.ylabel('Aij', fontsize=14)
# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_650', values='Aij', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_650 vs Aij', fontsize=16)
plt.xlabel('Aij', fontsize=14)
plt.ylabel('FP_i_650', fontsize=14)

# Show the heatmap
plt.show()
#%%650,Aji

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_650', y='Aji', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_650 vs Aji', fontsize=16)
plt.xlabel('FP_i_650', fontsize=14)
plt.ylabel('Aji', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_650', values='Aji', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_650 vs Aji', fontsize=16)
plt.xlabel('Aji', fontsize=14)
plt.ylabel('FP_i_650', fontsize=14)

# Show the heatmap
plt.show()
#%%650,Bij

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_650', y='Bij', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_650 vs Bij', fontsize=16)
plt.xlabel('FP_i_650', fontsize=14)
plt.ylabel('Bij', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_650', values='Bij', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_650 vs Bij', fontsize=16)
plt.xlabel('Bij', fontsize=14)
plt.ylabel('FP_i_650', fontsize=14)

# Show the heatmap
plt.show()
#%%650,Bji

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_650', y='Bji', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_650 vs Bji', fontsize=16)
plt.xlabel('FP_i_650', fontsize=14)
plt.ylabel('Bji', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_650', values='Bji', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_650 vs Bji', fontsize=16)
plt.xlabel('Bji', fontsize=14)
plt.ylabel('FP_i_650', fontsize=14)

# Show the heatmap
plt.show()
#%%650,alpha

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_650', y='Alpha', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_650 vs Alpha', fontsize=16)
plt.xlabel('FP_i_650', fontsize=14)
plt.ylabel('Alpha', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_650', values='Alpha', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_650 vs Alpha', fontsize=16)
plt.xlabel('Alpha', fontsize=14)
plt.ylabel('FP_i_650', fontsize=14)

# Show the heatmap
plt.show()
#%%222,Aij

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_222', y='Aij', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_222 vs Aij', fontsize=16)
plt.xlabel('FP_i_222', fontsize=14)
plt.ylabel('Aij', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_222', values='Aij', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_222 vs Aij', fontsize=16)
plt.xlabel('Aij', fontsize=14)
plt.ylabel('FP_i_222', fontsize=14)

# Show the heatmap
plt.show()
#%%222,Aji

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_222', y='Aji', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_222 vs Aji', fontsize=16)
plt.xlabel('FP_i_222', fontsize=14)
plt.ylabel('Aji', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_222', values='Aji', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_222 vs Aji', fontsize=16)
plt.xlabel('Aji', fontsize=14)
plt.ylabel('FP_i_222', fontsize=14)

# Show the heatmap
plt.show()
#%%222,Bij

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_222', y='Bij', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_222 vs Bij', fontsize=16)
plt.xlabel('FP_i_222', fontsize=14)
plt.ylabel('Bij', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_222', values='Bij', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_222 vs Bij', fontsize=16)
plt.xlabel('Bij', fontsize=14)
plt.ylabel('FP_i_222', fontsize=14)

# Show the heatmap
plt.show()
#%%222,Bji

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_222', y='Bji', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_222 vs Bji', fontsize=16)
plt.xlabel('FP_i_222', fontsize=14)
plt.ylabel('Bji', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_222', values='Bji', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_222 vs Bji', fontsize=16)
plt.xlabel('Bji', fontsize=14)
plt.ylabel('FP_i_222', fontsize=14)

# Show the heatmap
plt.show()
#%%222,alpha

plt.figure(figsize=(8, 6))
sns.scatterplot(x='FP_i_222', y='Alpha', data=df)

# Customize the plot
plt.title('Scatter Plot of FP_i_222 vs Alpha', fontsize=16)
plt.xlabel('FP_i_222', fontsize=14)
plt.ylabel('Alpha', fontsize=14)

# Set x-axis to plain format (normal numbers)
plt.ticklabel_format(style='plain', axis='x')

# Set y-axis to scientific notation (10^x)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Show the scatter plot
plt.show()

# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='FP_i_222', values='Alpha', aggfunc='std')

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")

# Customize the heatmap
plt.title('Heatmap of FP_i_222 vs Alpha', fontsize=16)
plt.xlabel('Alpha', fontsize=14)
plt.ylabel('FP_i_222', fontsize=14)

# Show the heatmap
plt.show()