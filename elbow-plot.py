import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Baca dataset
df = pd.read_csv('dataset_alfa.csv')

# Tentukan nilai X
X = df.iloc[:,].values

# Hitung inersia untuk berbagai jumlah kluster
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot grafik Elbow
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Metode Elbow')
plt.xlabel('Jumlah Kluster')
plt.ylabel('Inersia')
plt.show()