import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(file_path):
    """Memuat dataset dari file CSV"""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} tidak ditemukan")
    return pd.read_csv(file_path)

def visualize_outliers(df, numerical_features, save_path="outlier_visualization.png"):
    """Membuat visualisasi boxplot untuk outlier"""
    plt.figure(figsize=(16, 8))
    
    # Membuat subplot dengan orientasi horizontal
    ax = sns.boxplot(data=df[numerical_features], orient="h", palette="Set2")
    plt.title("Distribusi dan Outlier pada Fitur Numerik", pad=20, fontsize=14)
    plt.xlabel("Nilai", fontsize=12)
    plt.ylabel("Fitur Numerik", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Menyesuaikan layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Visualisasi outlier disimpan sebagai {save_path}")

def detect_outliers(df, numerical_features):
    """Mendeteksi outlier menggunakan metode IQR"""
    Q1 = df[numerical_features].quantile(0.25)
    Q3 = df[numerical_features].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (df[numerical_features] < lower_bound) | (df[numerical_features] > upper_bound)
    return outlier_mask

def save_datasets(df_with_outliers, df_without_outliers):
    """Menyimpan dataset dengan dan tanpa outlier"""
    df_with_outliers.to_csv("data_with_outliers.csv", index=False)
    df_without_outliers.to_csv("data_without_outliers.csv", index=False)
    
    print("\n📊 Statistik Dataset:")
    print(f"- Dataset dengan outlier: {df_with_outliers.shape[0]} baris")
    print(f"- Dataset tanpa outlier: {df_without_outliers.shape[0]} baris")
    print(f"- Persentase data dihapus: {100*(1-df_without_outliers.shape[0]/df_with_outliers.shape[0]):.2f}%")

def main():
    # Konfigurasi
    DATA_FILE = "data_encode.csv"
    OUTPUT_DIR = "output"
    
    # Membuat direktori output jika belum ada
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    try:
        # 1. Memuat data
        df = load_data(DATA_FILE)
        
        # 2. Identifikasi fitur numerik
        numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
        
        # 3. Visualisasi outlier
        visualize_outliers(df, numerical_features, f"{OUTPUT_DIR}/outlier_visualization.png")
        
        # 4. Deteksi outlier
        outlier_mask = detect_outliers(df, numerical_features)
        
        # 5. Membuat dataset dengan dan tanpa outlier
        df_with_outliers = df.copy()
        df_without_outliers = df[~outlier_mask.any(axis=1)].copy()
        
        # 6. Menyimpan hasil
        save_datasets(df_with_outliers, df_without_outliers)
        
        print("\n✅ Proses deteksi outlier selesai!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
