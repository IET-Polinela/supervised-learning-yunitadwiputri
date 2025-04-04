import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Membuat folder untuk menyimpan visualisasi jika belum ada
visualisasi_folder = "visualisasi_linear"
os.makedirs(visualisasi_folder, exist_ok=True)

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
if "SalePrice" in df_with_outliers.columns and "SalePrice" in df_without_outliers.columns:
    X_with_outliers = df_with_outliers.drop(columns=["SalePrice"])
    Y_with_outliers = df_with_outliers["SalePrice"]

    X_without_outliers = df_without_outliers.drop(columns=["SalePrice"])
    Y_without_outliers = df_without_outliers["SalePrice"]
else:
    print("Kolom 'SalePrice' tidak ditemukan dalam dataset.")
    exit()

# Fungsi untuk menerapkan Linear Regression dan membuat visualisasi
def linear_regression(X, Y, label):
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)

    print(f"{label} - MSE: {mse:.2f}, R²: {r2:.4f}")

    # Scatter plot antara nilai aktual dan prediksi
    plt.figure(figsize=(6, 4))
    plt.scatter(Y, Y_pred, alpha=0.5)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Linear Regression {label}")
    
    filename = os.path.join(visualisasi_folder, f"linear_regression_{label.lower()}.png")
    plt.savefig(filename)
    plt.show()

    # Residual plot
    residuals = Y - Y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot {label}")
    
    filename = os.path.join(visualisasi_folder, f"linear_residual_plot_{label.lower()}.png")
    plt.savefig(filename)
    plt.show()

    # Distribusi Residual
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution {label}")

    filename = os.path.join(visualisasi_folder, f"linear_residual_distribution_{label.lower()}.png")
    plt.savefig(filename)
    plt.show()

    return mse, r2

# Evaluasi Linear Regression
print("\nEvaluasi dengan Outlier:")
mse_with, r2_with = linear_regression(X_with_outliers, Y_with_outliers, "With_Outliers")

print("\nEvaluasi tanpa Outlier:")
mse_without, r2_without = linear_regression(X_without_outliers, Y_without_outliers, "Without_Outliers")
