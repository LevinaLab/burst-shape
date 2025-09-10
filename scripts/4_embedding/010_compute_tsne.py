import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.persistence import (
    load_burst_matrix,
    load_distance_matrix,
    load_spectral_embedding,
    load_tsne,
    save_pca,
    save_tsne,
    tsne_exists,
)
from src.persistence.agglomerative_clustering import get_agglomerative_labels

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
agglomerating_clustering_params = (
    None  # not plotting labels
    # "agglomerating_clustering_linkage_complete"
    # "agglomerating_clustering_linkage_ward"
)
spectral_clustering_params = (  # required for spectral embedding initialization
    # "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
    # "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_60"
    # "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
    "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_55"
)
tsne_params = {
    "initialization": ["pca", "spectral"][0],
}

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
if recompute_tsne or not tsne_exists(burst_extraction_params, tsne_params):
    distance_matrix = load_distance_matrix(burst_extraction_params, form="matrix")
    burst_matrix = load_burst_matrix(burst_extraction_params)

    # initalization
    match tsne_params["initialization"]:
        case "pca":
            # pca initialization
            pca = PCA(n_components=2)
            pca_burst = pca.fit_transform(burst_matrix)
            save_pca(pca_burst, burst_extraction_params)
            init = pca_burst
        case "spectral":
            spectral_embedding = load_spectral_embedding(
                burst_extraction_params, spectral_clustering_params
            )
            init = spectral_embedding
        case _:
            raise ValueError(
                f"Unknown initialization: {tsne_params['initialization']}."
            )

    # t-SNE
    n_points = distance_matrix.shape[0]
    tsne_burst = TSNE(
        init=init,
        n_components=2,
        perplexity=n_points / 100,
        learning_rate=n_points / 12,
        early_exaggeration=4,
        n_jobs=None,  # no impact if precomputed
        verbose=1,
        metric="precomputed",
    ).fit_transform(distance_matrix)
    save_tsne(tsne_burst, burst_extraction_params, tsne_params)
else:
    print(f"Loading t-SNE from file.")  # {file_tsne}")
    tsne_burst = load_tsne(burst_extraction_params, tsne_params)

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
