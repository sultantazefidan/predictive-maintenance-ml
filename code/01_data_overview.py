import pandas as pd

# CSV yolu
path = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset\ai4i2020_clean_step2.csv"

# Veri setini oku
df = pd.read_csv(path)

# İlk 5 satır
print("İlk 5 satır:")
print(df.head(), "\n")

# Boyut bilgisi
print("Veri boyutu (satır, sütun):")
print(df.shape, "\n")

# Sütun isimleri
print("Sütunlar:")
print(df.columns, "\n")

# Hedef değişken dağılımı
print("Machine failure sınıf dağılımı:")
print(df["Machine failure"].value_counts())

print("\nOranlar:")
print(df["Machine failure"].value_counts(normalize=True))
