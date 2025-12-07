import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

X_min = X.min()
X_max = X.max()

X_scaled_manual = (X - X_min) / (X_max - X_min)

print("--- ข้อมูลหลัง Normalize (เขียนเอง) ---")
print(X_scaled_manual.head())

threshold_value = 0.035

# Mean
mean_val = X_scaled_manual.mean()

# Var
variance_manual = ((X_scaled_manual - mean_val)**2).mean()

print(f"\n--- ค่า Variance ที่คำนวณเอง ---\n{variance_manual}")

cols_to_keep = variance_manual[variance_manual > threshold_value].index

X_reduced_manual = X_scaled_manual[cols_to_keep]

print(f"\n--- ผลลัพธ์หลังตัด (เหลือ {X_reduced_manual.shape[1]} ฟีเจอร์) ---")
print(X_reduced_manual.head())