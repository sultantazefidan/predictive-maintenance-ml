import pandas as pd
import numpy as np

# One-hot yapılmış veri yolu
path = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset\ai4i2020_clean_step2.csv"

df = pd.read_csv(path)

print(" NaN kontrolü:")
print(df.isnull().sum())

print("\n Inf / -Inf kontrolü:")
print(np.isinf(df.select_dtypes(include=[np.number])).sum())

print("\n Duplicate kayıt sayısı:")
print(df.duplicated().sum())
