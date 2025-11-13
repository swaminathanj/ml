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
2. Add code to print the dataset (after the points are created).
3. Cut the dendrogram at different heights and see how it reflects in the cluster assignments.
4. Tweak the code to generate datasets of different sizes (say 10, 25, 30, 40) and examine the dendrograms.
5. Tabulate what would be the cut off height for each cases.
6. Change the distance metric from 'Euclidean' to 'Cosine similarity' and compare the cluster assignments for each data point.
