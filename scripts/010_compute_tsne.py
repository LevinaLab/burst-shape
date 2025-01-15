import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.folders import get_results_folder
from src.persistence import (
    load_burst_matrix,
    load_distance_matrix,
    load_tsne,
    save_pca,
    save_tsne,
    tsne_exists,
)
from src.persistence.agglomerative_clustering import get_agglomerative_labels

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
)
agglomerating_clustering_params = (
    None  # not plotting labels
    # "agglomerating_clustering_linkage_complete"
    # "agglomerating_clustering_linkage_ward"
)
n_clusters = 3

recompute_tsne = False

# define colors
palette = sns.color_palette(n_colors=n_clusters)
cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
cluster_colors = [
    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
    for c in cluster_colors
]

# load labels
try:
    labels = get_agglomerative_labels(
        n_clusters, burst_extraction_params, agglomerating_clustering_params
    )
except ValueError:
    labels = None
    print(f"Labels not found. Run without labels and just computing embeddings.")

# %% compute or load t-SNE
if recompute_tsne or not tsne_exists(burst_extraction_params):
    distance_matrix = load_distance_matrix(burst_extraction_params, form="matrix")
    burst_matrix = load_burst_matrix(burst_extraction_params)

    # pca initialization
    pca = PCA(n_components=2)
    pca_burst = pca.fit_transform(burst_matrix)
    save_pca(pca_burst, burst_extraction_params)

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
    save_tsne(tsne_burst, burst_extraction_params)
else:
    print(f"Loading t-SNE from file.")  # {file_tsne}")
    tsne_burst = load_tsne(burst_extraction_params)

# %% plot t-SNE
fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
if labels is None:
    ax.scatter(tsne_burst[:, 0], tsne_burst[:, 1], s=1, color="k", label="data")
else:
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
