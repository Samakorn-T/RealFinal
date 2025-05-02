import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import pdist
from minepy import MINE
import numpy as np

# กำหนดเส้นทางไฟล์และโหลดข้อมูล
input_folder = 'Prepare_datavisual'
input_file = 'Expanded_Fingerprints_Data.xlsx'
input_path = os.path.join(input_folder, input_file)

# ตรวจสอบและโหลดข้อมูลจากไฟล์
if os.path.exists(input_path):
    df = pd.read_excel(input_path)
else:
    raise FileNotFoundError(f"The file {input_path} does not exist.")

# ตรวจสอบข้อมูล
print(df.head())  # แสดงแถวแรกของข้อมูล
print(df.isnull().sum())  # ตรวจสอบค่าว่างในข้อมูล

# เลือกคอลัมน์ fingerprint (เช่น FP_i_1, FP_i_2, ...)
fingerprint_cols = [col for col in df.columns if col.startswith('FP_i_')]
Bij = df['Bij']  # กำหนดคอลัมน์ Bij

# สร้างโฟลเดอร์หลักสำหรับบันทึกรูปภาพ
output_folder = "ScatterPlots_ByBit"
os.makedirs(output_folder, exist_ok=True)

# ฟังก์ชันสำหรับคำนวณ Distance Correlation
def distance_correlation(x, y):
    def dist_cov(x, y):
        return np.mean(pdist(x[:, None], 'euclidean') * pdist(y[:, None], 'euclidean'))
    dcov_xx = dist_cov(x, x)
    dcov_yy = dist_cov(y, y)
    dcov_xy = dist_cov(x, y)
    return dcov_xy / (np.sqrt(dcov_xx * dcov_yy))

# ฟังก์ชันสำหรับการบันทึก Scatter Plot
def save_scatter_plot(x, y, bit_name, method, correlation_value):
    method_folder = os.path.join(output_folder, method)
    os.makedirs(method_folder, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, alpha=0.6)
    plt.title(f"Scatter Plot: {bit_name} vs Bij\n{method}: {correlation_value:.4f}")
    plt.xlabel(bit_name)
    plt.ylabel("Bij")
    plt.grid(True)
    
    # บันทึกไฟล์รูปภาพ
    plot_path = os.path.join(method_folder, f"{bit_name}_{method}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

# เรียกใช้การวิเคราะห์สำหรับทุกบิต
for bit in fingerprint_cols:
    x = df[bit]
    y = Bij
    
    # Pearson Correlation
    pearson_corr = x.corr(y, method='pearson')
    save_scatter_plot(x, y, bit, "Pearson", pearson_corr)
    
    # Spearman Correlation
    spearman_corr = x.corr(y, method='spearman')
    save_scatter_plot(x, y, bit, "Spearman", spearman_corr)
    
    # Mutual Information
    mi_score = mutual_info_regression(x.values.reshape(-1, 1), y)[0]
    save_scatter_plot(x, y, bit, "MI", mi_score)
    
    # Distance Correlation
    dcor_score = distance_correlation(x.values, y.values)
    save_scatter_plot(x, y, bit, "dCor", dcor_score)
    
    # Maximal Information Coefficient (MIC)
    mine = MINE()
    mine.compute_score(x, y)
    mic_score = mine.mic()
    save_scatter_plot(x, y, bit, "MIC", mic_score)

print(f"Scatter plots and correlation values saved to '{output_folder}'")