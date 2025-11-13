# Clustering - Lab Activity

## Part 1: Hierarchical Clustering
Inspect the following implementation of **Hierarchical clustering** and answer the questions that follow:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import make_blobs

# --- 1. Generate Sample Data ---
# Create synthetic 2D data suitable for visualization
X, y = make_blobs(
    n_samples=20,       # Small number of samples for a readable dendrogram
    n_features=2,
    centers=3,
    cluster_std=0.8,
    random_state=42
)

print(f"Generated data shape: {X.shape}")

# --- 2. Perform Hierarchical Clustering ---

# 2a. Calculate the linkage matrix (Z)
# The linkage function computes the distances between clusters.
#
# Arguments:
# - X: The input data points
# - method='ward': The linkage criterion. Ward minimizes the variance of the clusters being merged.
#                  Other options include 'single', 'complete', or 'average'.
# - metric='euclidean': The distance metric to use.
Z = linkage(X, method='ward', metric='euclidean')

# The resulting matrix Z contains the clustering information:
# Z[i, 0] and Z[i, 1]: indices of the two clusters/points merged in step i
# Z[i, 2]: the distance between the merged clusters
# Z[i, 3]: the number of original observations in the new cluster

# --- 3. Visualize the Result (Dendrogram) ---

plt.figure(figsize=(12, 6))
plt.title('Agglomerative Hierarchical Clustering Dendrogram (Ward Linkage, Euclidean Distance)')
plt.xlabel('Data Point Index (or Cluster Size)')
plt.ylabel('Distance (Linkage Dissimilarity)')

# Plot the dendrogram using the linkage matrix Z
# The orientation='top' displays the hierarchy from bottom-up (Agglomerative)
dendrogram(
    Z,
    orientation='top',
    labels=None, # You can use labels=y to color code final leaves if needed
    distance_sort='descending',
    show_leaf_counts=True,
    color_threshold=3.5 # Draw a horizontal line to visualize 3 clusters (just for visual aid)
)

# Draw a horizontal line at the distance threshold to illustrate cluster selection
plt.axhline(y=3.5, color='r', linestyle='--', label='Cut-off for 3 Clusters')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.show()
# 

# --- 4. Extract Final Clusters (If needed) ---

from scipy.cluster.hierarchy import fcluster

# If we "cut" the dendrogram at a distance threshold of 3.5 (the red dashed line),
# we would obtain the following cluster assignments:
max_d = 3.5
clusters = fcluster(Z, max_d, criterion='distance')

print("\n--- Cluster Assignments ---")
print(f"Number of clusters found at distance threshold {max_d}: {len(np.unique(clusters))}")
print(f"Cluster labels: {clusters}")
```

1. Understand the code in relation to hierarchical clustering algorithm. 
2. Add code to print the dataset (after the points are created). Is the dataset different for each run or same?
3. Cut the dendrogram at different heights and see how it affects the number of clusters and cluster assignments to the data points.
4. Tweak the code to generate datasets of different sizes (say 10, 15, 25, 30) and examine the dendrograms.
5. Tabulate what would be the cut off height for each cases.
6. Change the distance metric from 'Euclidean' to 'Cosine similarity' and compare the cluster assignments for each data point.

## Part 2: K-Means Clustering
Inspect the following implementation of **k-means clustering** and answer the questions that follow:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def _euclidean_distance(self, p1, p2):
        """Calculates the Euclidean distance between two points."""
        return np.sqrt(np.sum((p1 - p2)**2))

    def fit(self, X):
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape

        # 1. Initialization: Randomly select k data points as initial centroids
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. Assignment Step (E-Step): Assign each sample to the closest centroid
            self.labels = self._assign_clusters(X)

            # 3. Update Step (M-Step): Recalculate new centroids
            new_centroids = self._update_centroids(X)

            # 4. Check for Convergence: Stop if centroids haven't moved significantly
            # Check if all new centroids are equal to old ones (or very close)
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids

        return self

    def _assign_clusters(self, X):
        """Assigns each data point to the nearest centroid."""
        labels = np.zeros(X.shape[0])
        for i, sample in enumerate(X):
            distances = [self._euclidean_distance(sample, centroid) for centroid in self.centroids]
            # The label is the index of the minimum distance (the closest centroid)
            labels[i] = np.argmin(distances)
        return labels.astype(int)

    def _update_centroids(self, X):
        """Calculates the mean of all samples assigned to a cluster."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            # Get all samples assigned to the current cluster k
            cluster_points = X[self.labels == k]
            
            if len(cluster_points) > 0:
                # Calculate the mean of these points
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty clusters by keeping the old centroid or re-initializing
                new_centroids[k] = self.centroids[k] 
        return new_centroids

    def predict(self, X):
        return self._assign_clusters(X)

# --- Demonstration ---

# 1. Generate Sample Data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)

# 2. Initialize and Train K-Means
k = 4
kmeans = KMeansFromScratch(n_clusters=k, max_iters=100, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.labels

# 3. Visualize Results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.7)

# Plot the final centroids
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c='red', s=200, marker='X', label='Centroids')

plt.title(f'K-Means Clustering with K={k} (From Scratch)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()
```
7. Understand the code in relation to k-means clustering algorithm.
8. Add code to print the number of iterations the algorithm took to converge.
9. Mark the initial centroids in the plot using blue cross.
10. Do the initial centroids change each time you run?
11. Increase K by 1 (i.e. k = 5) and see its effect on the clusters. Note down your observations.
12. Decrease K by 1 (i.e. k = 3) and see its effect on the clusters. Note down your observations.
13. How do th
