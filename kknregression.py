import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Membuat folder untuk menyimpan visualisasi
output_folder = "knn_visualisasi"
os.makedirs(output_folder, exist_ok=True)

# Load dataset
file_with_outliers = "data_with_outliers.csv"
file_without_outliers = "dataset_clean_proces.csv"

try:
    df_with_outliers = pd.read_csv(file_with_outliers)
    df_without_outliers = pd.read_csv(file_without_outliers)
    print("Dataset berhasil dimuat!")
except Exception as e:
    print(f"Error saat membaca dataset: {e}")
    exit()

# Menghapus missing values
df_with_outliers.dropna(inplace=True)
df_without_outliers.dropna(inplace=True)

# Memastikan semua kolom numerik
df_with_outliers = df_with_outliers.select_dtypes(include=[np.number])
df_without_outliers = df_without_outliers.select_dtypes(include=[np.number])

# Memisahkan fitur dan target
target_column = "SalePrice"

if target_column in df_with_outliers.columns and target_column in df_without_outliers.columns:
    X_with_outliers = df_with_outliers.drop(columns=[target_column])
    Y_with_outliers = df_with_outliers[target_column]
    
    X_without_outliers = df_without_outliers.drop(columns=[target_column])
    Y_without_outliers = df_without_outliers[target_column]
else:
    print(f"Kolom '{target_column}' tidak ditemukan dalam dataset.")
    exit()

# Fungsi untuk menerapkan KNN Regression dan membuat visualisasi
def knn_regression(X, Y, label, k):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, Y)
    Y_pred = model.predict(X)
    
    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    
    print(f"KNN Regression {label} (K={k}) - MSE: {mse:.2f}, R²: {r2:.4f}")
    
    # Scatter plot antara nilai aktual dan prediksi
    plt.figure(figsize=(6, 4))
    plt.scatter(Y, Y_pred, alpha=0.5)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"KNN Regression {label} (K={k})")
    filename = os.path.join(output_folder, f"knn_regression_{label.lower()}_k{k}.png")
    plt.savefig(filename)
    plt.close()
    
    # Residual plot
    residuals = Y - Y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot {label} (K={k})")
    filename = os.path.join(output_folder, f"residual_plot_{label.lower()}_k{k}.png")
    plt.savefig(filename)
    plt.close()
    
    # Distribusi Residual
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution {label} (K={k})")
    filename = os.path.join(output_folder, f"residual_distribution_{label.lower()}_k{k}.png")
    plt.savefig(filename)
    plt.close()
    
    return mse, r2

# Evaluasi untuk K = 3, 5, 7
k_values = [3, 5, 7]

for k in k_values:
    print(f"\nEvaluasi dengan Outlier (K={k}):")
    knn_regression(X_with_outliers, Y_with_outliers, "With_Outliers", k)
    
    print(f"\nEvaluasi tanpa Outlier (K={k}):")
    knn_regression(X_without_outliers, Y_without_outliers, "Without_Outliers", k)
