import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


DATA_PATH = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset\ai4i2020_clean_step2.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Kümeleme için veri hazırlanamsı
# Etiket kullanılmıyor çünkü bu denetimsiz bir analiz

X = df.drop(columns=["Machine failure"])

print("\nClustering features:", X.shape[1])

# Normalize etme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA - görselleştirme için kulanıldı
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)


kmeans = KMeans(
    n_clusters=2,       # Arıza ve arıza olmama hipotezi
    random_state=42,
    n_init=1
)

clusters = kmeans.fit_predict(X_pca)


# Silhouette Score- nokta kendı kumesıne ne adar yakın
sil_score = silhouette_score(X_pca, clusters)
print("\nSilhouette Score:", round(sil_score, 4))


plt.figure(figsize=(7, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=15
)
plt.title("KMeans Clustering (EDA – PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster ID")
plt.tight_layout()
plt.show()

print("\nEDA clustering completed successfully.")
