"""
--- Deprecated ---

This script is now deprecated and will be removed in a future version.
It is no longer needed because the spectral embedding is implicitly
calculated in spectral clustering.
The one in spectral clustering is even better because it has more dimensions.
"""
import warnings

from matplotlib import pyplot as plt
from sklearn.manifold import SpectralEmbedding

from src.persistence import (
    load_affinity_matrix,
    load_burst_matrix,
    load_spectral_embedding,
    save_spectral_embedding,
    spectral_embedding_exists,
)

warnings.warn(
    "This script is now deprecated and will be removed in a future version. "
    "Use instead spectral clustering which implicitly calculates spectral embedding."
)

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
clustering_params = {
    "n_components_max": 30,
    "affinity": "precomputed",
    "metric": "euclidean",  # "wasserstein",
    "n_neighbors": None,  # 21,  # 85,  # 6,  # 60,   # 150,
    "random_state": 0,
}
if clustering_params["n_neighbors"] is None:
    burst_matrix = load_burst_matrix(burst_extraction_params)
    clustering_params["n_neighbors"] = burst_matrix.shape[0] // 100
    del burst_matrix


recompute_spectral_embedding = False

# %% compute or load spectral embedding
if recompute_spectral_embedding or not spectral_embedding_exists(
    burst_extraction_params, clustering_params
):
    try:
        print("Loading affinity matrix...")
        affinity_matrix = load_affinity_matrix(
            clustering_params,
            burst_extraction_params,
        )
        print("Loaded affinity matrix.")
    except FileNotFoundError as e:
        print("Affinity matrix not found. Run first spectral clustering to compute it.")
        raise e
    print("Computing spectral embedding...")
    spectral_embedding = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        gamma=None,
        n_neighbors=None,
    ).fit_transform(affinity_matrix)
    print("Finished.")
    save_spectral_embedding(
        spectral_embedding, burst_extraction_params, clustering_params
    )
else:
    print("Loading spectral embedding...")
    spectral_embedding = load_spectral_embedding(
        burst_extraction_params, clustering_params
    )
    print("Finished.")

# %%
fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
ax.scatter(
    spectral_embedding[:, 0],
    spectral_embedding[:, 1],
    s=1,
    color="k",
    label="data",
)
ax.legend(markerscale=10)
ax.axis("off")
fig.show()
