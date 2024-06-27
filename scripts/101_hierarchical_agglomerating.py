"""Test hierarchical clustering with Wasserstein distance."""
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

from src.persistence import load_burst_matrix

burst_extraction_params = "burst_n_bins_50_normalization_integral"
n_clusters = 8  # if None chooses the number of clusters with Davies-Bouldin index
n_bursts = 4000
linkage_method = "complete"

burst_matrix = load_burst_matrix(burst_extraction_params)
# select randomly bursts to reduce computation time
np.random.seed(0)
if n_bursts is not None:
    idx = np.random.choice(burst_matrix.shape[0], n_bursts, replace=False)
    burst_matrix = burst_matrix[idx]
print(burst_matrix.shape)

# %%


def _wasserstein_distance(a, b):
    a_cumsum = np.cumsum(a)
    b_cumsum = np.cumsum(b)
    a_cumsum /= a_cumsum[-1]
    b_cumsum /= b_cumsum[-1]
    return np.sum(np.abs(a_cumsum - b_cumsum))


# %% cluster with linkage
t0 = time()
distance_matrix = pdist(burst_matrix, metric=_wasserstein_distance)
t1 = time()
print(f"Distance matrix: {t1 - t0:.2f} s")
Z = linkage(distance_matrix, method=linkage_method)
t2 = time()
print(f"Linkage: {t2 - t1:.2f} s")

# %% Cross-validate with Davies-Bouldin index
n_clusters_range = range(2, 30)
score_davies_bouldin = np.zeros(len(n_clusters_range))
for _n_clusters in range(2, 30):
    labels = fcluster(Z, t=_n_clusters, criterion="maxclust")
    score_davies_bouldin[_n_clusters - 2] = davies_bouldin_score(burst_matrix, labels)
fig, ax = plt.subplots()
sns.despine()
ax.plot(n_clusters_range, score_davies_bouldin, "o-", label="Davies-Bouldin Index")
# highlight the best number of clusters
best_n_clusters = np.argmin(score_davies_bouldin) + 2
ax.plot(
    best_n_clusters,
    score_davies_bouldin[best_n_clusters - 2],
    "ro",
    label=f"Min: {best_n_clusters} clusters",
)
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Davies-Bouldin Index")
ax.legend(frameon=False)
fig.show()

if n_clusters is None:
    n_clusters = best_n_clusters

# %% Define a color palette for the clusters
palette = sns.color_palette("Set1", n_clusters)
cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
cluster_colors = [
    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
    for c in cluster_colors
]
# %%
# Plot dendrogram with colored clusters
fig, ax = plt.subplots(figsize=(12, 8))
sns.despine()
dendrogram(
    Z,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=10,
    above_threshold_color="black",
    color_threshold=Z[-(n_clusters - 1), 2],
)

# Highlight the clusters with background colors
x = np.arange(len(burst_matrix))
for i, d in enumerate(ax.collections):
    if i >= n_clusters:
        break
    color = cluster_colors[i]
    d.set_color(color)
ax.set_xlabel("Sample Index")
ax.set_ylabel("Distance")
fig.show()
# %% get clusters from linkage
labels = fcluster(Z, t=n_clusters, criterion="maxclust")

# %% PCA
pca = PCA(n_components=2)
pca.fit(burst_matrix)
X_pca = pca.transform(burst_matrix)

# %% plot clusters in PCA space
fig, ax = plt.subplots()
sns.despine()
for cluster in range(1, n_clusters + 1):
    cluster_points = X_pca[labels == cluster]
    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=10,
        color=palette[cluster - 1],
        label=f"Cluster {cluster}",
    )
ax.set_xlabel("1st PC")
ax.set_ylabel("2nd PC")
ax.legend()
fig.show()

# %% histogram of cluster sizes
fig, ax = plt.subplots()
sns.despine()
# Count the number of elements in each cluster
cluster_counts = np.bincount(labels)[1:]  # ignoring cluster 0
# Create a bar plot with the correct colors
ax.bar(range(1, n_clusters + 1), cluster_counts, color=palette)
# write numerical values on top of the bars
for i, count in enumerate(cluster_counts):
    ax.text(i + 1, count, str(count), ha="center", va="bottom")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Bursts")
ax.set_xticks(range(1, n_clusters + 1))
fig.show()

# %% plot average burst of each cluster
fig, ax = plt.subplots()
sns.despine()
for cluster in range(1, n_clusters + 1):
    cluster_bursts = burst_matrix[labels == cluster]
    ax.plot(
        cluster_bursts.mean(axis=0),
        color=palette[cluster - 1],
        label=f"Cluster {cluster}",
    )
ax.set_xlabel("Time [arb. units]")
ax.set_ylabel("Rate [arb. units]")
ax.legend()
fig.show()
