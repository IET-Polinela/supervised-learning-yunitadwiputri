import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path, target_column="SalePrice", test_size=0.2, random_state=42):
    """
    Memuat dan memproses data dari file CSV
    
    Parameters:
        file_path (str): Path ke file CSV
        target_column (str): Nama kolom target
        test_size (float): Proporsi data testing
        random_state (int): Seed untuk reproduktibilitas
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, df_encoded, label_encoders)
    """
    # 1. Memuat dataset
    df = pd.read_csv(file_path)
    
    # 2. Identifikasi fitur
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df.select_dtypes(include=["object"]).columns
    
    # 3. Encoding fitur kategorikal
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # 4. Memisahkan fitur dan target
    X = df_encoded.drop(columns=[target_column, "Id"])
    y = df_encoded[target_column]
    
    # 5. Membagi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 6. Menambahkan label DataType
    df_encoded["DataType"] = "Train"
    df_encoded.loc[X_test.index, "DataType"] = "Test"
    
    return X_train, X_test, y_train, y_test, df_encoded, label_encoders

def save_and_report_results(df_encoded, numerical_features, categorical_features, X_train, X_test):
    """Menyimpan data dan menampilkan laporan hasil preprocessing"""
    # 1. Menyimpan data hasil preprocessing
    output_file = "data_encode.csv"
    df_encoded.to_csv(output_file, index=False)
    
    # 2. Menampilkan informasi dataset
    print("\n" + "="*50)
    print("📊 LAPORAN PREPROCESSING DATA")
    print("="*50)
    print(f"📌 Dataset hasil preprocessing disimpan sebagai: {output_file}")
    print(f"📌 Total data: {df_encoded.shape[0]} baris, {df_encoded.shape[1]} kolom")
    print(f"📌 Fitur numerik: {len(numerical_features)}")
    print(f"📌 Fitur kategorikal: {len(categorical_features)}")
    print(f"📌 Data Training: {X_train.shape[0]} sampel")
    print(f"📌 Data Testing: {X_test.shape[0]} sampel")
    
    # 3. Menampilkan preview data
    print("\n" + "="*50)
    print("🖥️ PREVIEW DATA HASIL PREPROCESSING")
    print("="*50)
    print(df_encoded.head())

# Eksekusi utama
if __name__ == "__main__":
    # Konfigurasi
    DATA_FILE = "train.csv"
    TARGET_COLUMN = "SalePrice"
    
    # Memproses data
    X_train, X_test, y_train, y_test, df_encoded, label_encoders = load_and_preprocess_data(
        DATA_FILE, target_column=TARGET_COLUMN
    )
    
    # Menyimpan dan menampilkan hasil
    numerical_features = df_encoded.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df_encoded.select_dtypes(include=["object"]).columns
    save_and_report_results(df_encoded, numerical_features, categorical_features, X_train, X_test)
