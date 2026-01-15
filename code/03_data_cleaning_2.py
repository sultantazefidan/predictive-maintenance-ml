import pandas as pd

# Temizlenmiş veri yolu (step1)
path = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset\ai4i2020_clean_step1.csv"

# Veriyi oku
df = pd.read_csv(path)

print("ONE-HOT ÖNCESİ SÜTUNLAR:")
print(df.columns)
print("Veri boyutu:", df.shape)

# One-Hot Encoding (Type)
df_encoded = pd.get_dummies(df, columns=["Type"], drop_first=True)

print("ONE-HOT ENCODING SONRASI")
print("----------------------------")

print("Yeni sütunlar:")
print(df_encoded.columns)
print("Yeni veri boyutu:", df_encoded.shape)

# Yeni veri setini kaydet (step2)
save_path = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset\ai4i2020_clean_step2.csv"
df_encoded.to_csv(save_path, index=False)

print("\nOne-Hot yapılmış veri kaydedildi:")
print(save_path)
