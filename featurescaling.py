import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset tanpa outlier
file_path = "data_without_outliers.csv"
df = pd.read_csv(file_path)

# Identifikasi fitur numerik
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

# Scaling menggunakan StandardScaler
scaler_standard = StandardScaler()
df_standard = df.copy()
df_standard[numerical_features] = scaler_standard.fit_transform(df[numerical_features])

# Scaling menggunakan MinMaxScaler
scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numerical_features] = scaler_minmax.fit_transform(df[numerical_features])

# Simpan hasil scaling ke dalam satu dataset
df_standard["ScalingType"] = "StandardScaler"
df_minmax["ScalingType"] = "MinMaxScaler"
df_combined = pd.concat([df_standard, df_minmax])

# Simpan dataset hasil scaling
df_combined.to_csv("dataset_clean_proces.csv", index=False)

# Visualisasi distribusi data sebelum dan sesudah scaling
fig, axes = plt.subplots(len(numerical_features), 3, figsize=(12, len(numerical_features) * 3))

for i, feature in enumerate(numerical_features):
    # Histogram sebelum scaling
    axes[i, 0].hist(df[feature], bins=30, color="blue", alpha=0.7)
    axes[i, 0].set_title(f"Original: {feature}")

    # Histogram setelah StandardScaler
    axes[i, 1].hist(df_standard[feature], bins=30, color="red", alpha=0.7)
    axes[i, 1].set_title(f"StandardScaler: {feature}")

    # Histogram setelah MinMaxScaler
    axes[i, 2].hist(df_minmax[feature], bins=30, color="green", alpha=0.7)
    axes[i, 2].set_title(f"MinMaxScaler: {feature}")

plt.tight_layout()
plt.savefig("feature_scaling_comparison.png")  # Simpan visualisasi
plt.show()

print("Feature scaling selesai. Dataset tersimpan sebagai 'dataset_clean_proces.csv'.")
print("Visualisasi perbandingan scaling disimpan sebagai 'feature_scaling_comparison.png'.")
