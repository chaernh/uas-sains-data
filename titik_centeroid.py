import pandas as pd
from sklearn.cluster import KMeans

# Baca dataset
df = pd.read_csv('dataset_alfa.csv')

# Tentukan nilai X
X = df.iloc[:,].values

# Tentukan jumlah kluster (3 kluster)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Dapatkan titik centroid dari masing-masing cluster
centroids = kmeans.cluster_centers_

# Tampilkan titik centroid
for i, centroid in enumerate(centroids):
    print(f"Centroid Kluster {i+1}: {centroid}")