from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Flower Clustering Project")
plt.savefig("screenshots/iris_clustering_project.png")
plt.show()

# Compare predicted clusters with true labels
comparison = pd.DataFrame({
    "True Label": y,
    "Predicted Cluster": clusters
})

print("\nFirst 10 Comparisons:")
print(comparison.head(10))

print("\nConfusion Matrix:")
print(confusion_matrix(y, clusters))