#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:30:12 2024

@author: padprow
"""

import os
import pandas as pd
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

# ฟังก์ชันสำหรับคำนวณ Distance Correlation
def distance_correlation(x, y):
    def dist_cov(x, y):
        return np.mean(pdist(x[:, None], 'euclidean') * pdist(y[:, None], 'euclidean'))
    dcov_xx = dist_cov(x, x)
    dcov_yy = dist_cov(y, y)
    dcov_xy = dist_cov(x, y)
    return dcov_xy / (np.sqrt(dcov_xx * dcov_yy))

# สร้าง DataFrame สำหรับเก็บผลลัพธ์
results = []

# เรียกใช้การวิเคราะห์สำหรับทุกบิต
for bit in fingerprint_cols:
    x = df[bit]
    y = Bij
    
    # Pearson Correlation
    pearson_corr = x.corr(y, method='pearson')
    
    # Spearman Correlation
    spearman_corr = x.corr(y, method='spearman')
    
    # Mutual Information
    mi_score = mutual_info_regression(x.values.reshape(-1, 1), y)[0]
    
    # Distance Correlation
    dcor_score = distance_correlation(x.values, y.values)
    
    # Maximal Information Coefficient (MIC)
    mine = MINE()
    mine.compute_score(x, y)
    mic_score = mine.mic()
    
    # เพิ่มผลลัพธ์ในรูปแบบ Dictionary
    results.append({
        "Fingerprint": bit,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "Mutual Information": mi_score,
        "Distance Correlation": dcor_score,
        "MIC": mic_score
    })

# สร้าง DataFrame สำหรับผลลัพธ์ทั้งหมด
results_df = pd.DataFrame(results)

# สร้างโฟลเดอร์สำหรับบันทึกไฟล์ผลลัพธ์
output_folder = "Correlation_Results"
os.makedirs(output_folder, exist_ok=True)

# บันทึกผลลัพธ์เป็นไฟล์ Excel
output_file = os.path.join(output_folder, "Correlation_Summary.xlsx")
results_df.to_excel(output_file, index=False)

print(f"Correlation values saved to '{output_file}'")