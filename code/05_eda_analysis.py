import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CSV yolu (Step-2: One-Hot sonrası için)
path = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset\ai4i2020_clean_step2.csv"
save_dir = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset"

# Veri setini oku
df = pd.read_csv(path)

print("Veri boyutu:", df.shape)
print("\nSütunlar:")
print(df.columns)


# tek değişkenlı analiz
print("\n Sayısal sütunlar için özet istatistikler:")
print(df.describe())

# Histogram – Air temperature
plt.figure(figsize=(6,4))
plt.hist(
    df["Air temperature [K]"],
    bins=30,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.85
)
plt.title("Air Temperature Distribution")
plt.xlabel("Air temperature [K]")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)

plt.savefig(
    os.path.join(save_dir, "hist_air_temperature.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Histogram grafiği
plt.figure(figsize=(6,4))
plt.hist(
    df["Tool wear [min]"],
    bins=30,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.85
)
plt.title("Tool Wear Distribution")
plt.xlabel("Tool wear [min]")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)

plt.savefig(
    os.path.join(save_dir, "hist_tool_wear.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# çok değişklneli analiz
plt.figure(figsize=(12,8))
corr = df.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Matrix (Heatmap)")
plt.savefig(
    os.path.join(save_dir, "correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n✅ Histogramlar ve korelasyon matrisi başarıyla kaydedildi.")
