import pandas as pd

# CSV yolu
path = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset\ai4i2020.csv"
#Modele anlamlı katkısı olmayan iki sütün silindi ve modelin anlayabileceği format kaldı.

# Veri setini oku
df = pd.read_csv(path)

print("ORİJİNAL VERİ BOYUTU (satır, sütun):")
print(df.shape)
print("\nORİJİNAL SÜTUNLAR:")
print(df.columns)

# Anlamsız ID sütunlarını sil
df_clean = df.drop(columns=["UDI", "Product ID"])

print("ID SÜTUNLARI SİLİNDİKTEN SONRA")
print("----------------------------")

print("YENİ VERİ BOYUTU (satır, sütun):")
print(df_clean.shape)

print("\nYENİ SÜTUNLAR:")
print(df_clean.columns)

# Aynı klasöre yeni isimle kaydet
save_path = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset\ai4i2020_clean_step1.csv"

df_clean.to_csv(save_path, index=False)

print("\nTemizlenmiş veri seti kaydedildi:")
print(save_path)

