import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "train.csv"
df = pd.read_csv(file_path)

# Hitung statistik deskriptif
stats = df.describe(percentiles=[0.25, 0.5, 0.75])
stats = stats.rename(columns={"50%": "Q2 (Median)", "25%": "Q1 (25%)", "75%": "Q3 (75%)"})
stats.loc['count'] = df.count()

# Simpan hasil statistik deskriptif ke CSV
stats.to_csv("statistik_deskriptif.csv")

# Analisis fitur
print("\nAnalisis Fitur:")
print("- Fitur 'Id': Tidak berguna untuk analisis, bisa dihapus")
print("- Fitur 'LotFrontage': Memiliki nilai yang hilang, perlu ditangani dengan metode imputasi")
print("- Fitur 'MSSubClass': Walaupun berupa angka, fitur ini sebenarnya kategori, sehingga perlu encoding")
