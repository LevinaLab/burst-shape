import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.folders import get_results_folder
from src.persistence import (
    load_clustering_labels,
    load_clustering_maps,
    load_df_bursts,
    load_spectral_embedding,
)
from src.plot import get_cluster_colors

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
    n_clusters = 4
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
    n_clusters = 4
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
    n_clusters = 4
elif "mossink" in burst_extraction_params:
    dataset = "mossink"
    n_clusters = 4
else:
    dataset = "wagenaar"
    n_clusters = 6
print(f"Detected dataset: {dataset}")

clustering_params = {
    "n_components_max": 30,
    "affinity": "precomputed",
    "metric": "wasserstein",
    "n_neighbors": 85,  # 6,  # 60,   # 150,
    "random_state": 0,
}

clustering = load_clustering_maps(
    clustering_params,
    burst_extraction_params,
    params_cross_validation=None,
    i_split=None,
)

print("Try loading spectral embedding...")
try:
    spectral_embedding = load_spectral_embedding(
        burst_extraction_params, clustering_params
    )
    print("Successfully loaded spectral embedding.")

    print("Comparing spectral embeddings...")
    print("spectral_embedding shape:", spectral_embedding.shape)
    print("clustering.maps_.shape:", clustering.maps_.shape)

    e1 = spectral_embedding[:, 0]
    m1 = clustering.maps_[:, 1]
    print("First dim same:", np.allclose(e1, m1))
except Exception as e:
    print(e)
    print("Failed to load spectral embedding - skipping.")
    print("This is not an issue because it's redundant.")

# %% Assign clusters and assign spectral embedding
df_bursts = load_df_bursts(burst_extraction_params)
df_bursts.drop(labels=df_bursts.columns, axis="columns", inplace=True)
clustering = load_clustering_labels(
    clustering_params,
    burst_extraction_params,
    "labels",
    None,
    i_split=None,
)
n_clusters_list = np.arange(2, 10, 1)
for n_clusters_ in n_clusters_list:
    df_bursts[f"cluster_{n_clusters_}"] = [
        f"Cluster {label + 1}" for label in clustering.labels_[n_clusters_]
    ]
spectral_columns = [f"Spec.-Dim. {dim}" for dim in range(1, 11)]
df_bursts.loc[:, spectral_columns] = clustering.maps_[:, 1:11]

df_bursts.to_csv(os.path.join(get_results_folder(), "df_spectral_embedding.csv"))

df_bursts_test = pd.read_csv(
    os.path.join(get_results_folder(), "df_spectral_embedding.csv")
)

ndims = 5
fig, axs = plt.subplots(
    nrows=ndims,
    ncols=ndims,
    sharex="col",
    sharey="row",
    constrained_layout=True,
    figsize=(10, 10),
)
for i in range(ndims):
    for j in range(ndims):
        sns.scatterplot(
            df_bursts_test,
            x=f"Spec.-Dim. {j+1}",
            y=f"Spec.-Dim. {i+1}",
            s=0.3,
            alpha=0.5,
            ax=axs[i, j],
            hue=f"cluster_{n_clusters}",
            hue_order=sorted(df_bursts_test[f"cluster_{n_clusters}"].unique()),
            palette=get_cluster_colors(n_clusters),
            legend=False,
        )
fig.show()
