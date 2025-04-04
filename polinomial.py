import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Buat folder untuk menyimpan visualisasi jika belum ada
output_dir = "polynomial_visualisasi"
os.makedirs(output_dir, exist_ok=True)

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

# Fungsi untuk menerapkan Polynomial Regression dan visualisasi hasil
def polynomial_regression(degree, X, Y, label):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, Y)
    Y_pred = model.predict(X_poly)

    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)

    print(f"{label} - Degree {degree} - MSE: {mse:.2f}, R²: {r2:.4f}")

    # Scatter plot (Prediksi vs Aktual)
    plt.figure(figsize=(6, 4))
    plt.scatter(Y, Y_pred, alpha=0.5, label=f"Degree {degree}")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Polynomial Regression {label} - Degree {degree}")
    plt.legend()
    plt.savefig(f"{output_dir}/polynomial_scatter_{label.lower()}_degree_{degree}.png")
    plt.close()

    # Residual Plot
    residuals = Y - Y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot {label} - Degree {degree}")
    plt.savefig(f"{output_dir}/polynomial_residual_{label.lower()}_degree_{degree}.png")
    plt.close()

    # Distribusi Residual
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution {label} - Degree {degree}")
    plt.savefig(f"{output_dir}/polynomial_residual_dist_{label.lower()}_degree_{degree}.png")
    plt.close()

    return mse, r2

# Evaluasi Polynomial Regression
print("\nEvaluasi dengan Outlier:")
mse_wo_d2, r2_wo_d2 = polynomial_regression(2, X_with_outliers, Y_with_outliers, "With_Outliers")
mse_wo_d3, r2_wo_d3 = polynomial_regression(3, X_with_outliers, Y_with_outliers, "With_Outliers")

print("\nEvaluasi tanpa Outlier:")
mse_wo_no_d2, r2_wo_no_d2 = polynomial_regression(2, X_without_outliers, Y_without_outliers, "Without_Outliers")
mse_wo_no_d3, r2_wo_no_d3 = polynomial_regression(3, X_without_outliers, Y_without_outliers, "Without_Outliers")
