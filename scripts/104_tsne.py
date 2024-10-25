import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.folders import get_results_folder
from src.persistence import load_burst_matrix

burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
)
clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
n_clusters = 2

recompute_tsne = False

# define colors
palette = sns.color_palette(n_colors=n_clusters)
cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
cluster_colors = [
    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
    for c in cluster_colors
]

# load linkage -> labels
print(f"Loading linkage from {get_results_folder()}")
linkage_file = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    clustering_params,
    "linkage.npy",
)
linkage = np.load(linkage_file)
labels = fcluster(linkage, t=n_clusters, criterion="maxclust")

# %% compute or load t-SNE
file_tsne = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    clustering_params,
    "tsne.npy",
)
if recompute_tsne or not os.path.exists(file_tsne):
    distance_matrix_file = os.path.join(
        get_results_folder(),
        burst_extraction_params,
        clustering_params,
        "distance_matrix.npy",
    )
    distance_matrix = squareform(np.load(distance_matrix_file), force="tomatrix")
    burst_matrix = load_burst_matrix(burst_extraction_params)
    file_idx = os.path.join(
        get_results_folder(),
        burst_extraction_params,
        clustering_params,
        "idx.npy",
    )
    if os.path.exists(file_idx):
        idx = np.load(file_idx)
        burst_matrix = burst_matrix[idx]

    # pca initialization
    pca = PCA(n_components=2)
    pca_burst = pca.fit_transform(burst_matrix)
    file_pca = os.path.join(
        get_results_folder(),
        burst_extraction_params,
        clustering_params,
        "pca.npy",
    )
    np.save(file_pca, pca_burst)

    # t-SNE
    n_points = distance_matrix.shape[0]
    tsne_burst = TSNE(
        init=pca_burst,
        n_components=2,
        perplexity=n_points / 100,
        learning_rate=n_points / 12,
        early_exaggeration=4,
        n_jobs=None,  # no impact if precomputed
        verbose=1,
        metric="precomputed",
    ).fit_transform(distance_matrix)
    np.save(file_tsne, tsne_burst)
else:
    print(f"Loading t-SNE from {file_tsne}")
    tsne_burst = np.load(file_tsne)

# %% plot t-SNE
fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
for i in range(n_clusters):
    ax.scatter(
        tsne_burst[labels == i + 1, 0],
        tsne_burst[labels == i + 1, 1],
        s=1,
        color=cluster_colors[i],
        label=f"cluster {i + 1}",
    )
# legend with larger markers
ax.legend(markerscale=10)
ax.axis("off")
fig.show()
