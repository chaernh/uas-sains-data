import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Baca dataset
df = pd.read_csv('dataset_alfa.csv')

# Tentukan nilai X
X = df.iloc[:,].values

# Tentukan jumlah kluster (misalnya, 3 kluster berdasarkan Elbow Method)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot scatter plot untuk setiap kluster
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='red', label='Kluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='blue', label='Kluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='green', label='Kluster 3')

# Plot centroid
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroid')

plt.title('Hasil Clustering K-Means')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()