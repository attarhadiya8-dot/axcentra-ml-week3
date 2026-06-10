from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data

# -------------------------------
# Assignment 1: K-Means Clustering
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print("Cluster Labels:")
print(clusters)

# -------------------------------
# Assignment 2: PCA
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("\nFirst 5 PCA transformed samples:")
print(X_pca[:5])

# -------------------------------
# Visualize Clusters
# -------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering on Iris Dataset")
plt.savefig("screenshots/kmeans_clusters.png")
plt.show()