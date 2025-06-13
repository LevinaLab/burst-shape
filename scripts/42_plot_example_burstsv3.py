import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence import load_clustering_labels, load_df_bursts, load_df_cultures
from src.persistence.spike_times import get_inhibblock_spike_times
from src.plot import get_cluster_colors, prepare_plotting

cm = prepare_plotting()

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
    n_clusters = 4
elif "hommersom_test" in burst_extraction_params:
    dataset = "hommersom_test"
    n_clusters = 4
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
    n_clusters = 4
else:
    dataset = "wagenaar"
    n_clusters = 6
print(f"Detected dataset: {dataset}")

# which clustering to plot
col_cluster = f"cluster_{n_clusters}"

clustering_params = (
    # "agglomerating_clustering_linkage_complete"
    # "agglomerating_clustering_linkage_ward"
    # "agglomerating_clustering_linkage_average"
    # "agglomerating_clustering_linkage_single"
    # "spectral_affinity_precomputed_metric_wasserstein"
    # "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
    # "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_60"
    # "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_6"
    "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
)
labels_params = "labels"
cv_params = "cv"  # if cv_split is not None, chooses the cross-validation split
cv_split = (
    None  # set to None for plotting the whole clustering, set to int for specific split
)
# load bursts
df_bursts = load_df_bursts(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)
np.random.seed(0)

match dataset:
    case "kapucu":
        index_names = ["culture_type", "mea_number", "well_id", "DIV"]
    case "wagenaar":
        index_names = ["batch", "culture", "day"]
    case "hommersom_test":
        index_names = ["batch", "clone", "well_idx"]
    case "inhibblock":
        index_names = ["drug_label", "div", "well_idx"]
    case _:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
#  get clusters from linkage
# print("Getting clusters from linkage...")
# labels = get_agglomerative_labels(
#     n_clusters, burst_extraction_params, agglomerating_clustering_params
# )
clustering = load_clustering_labels(
    clustering_params, burst_extraction_params, labels_params, cv_params, cv_split
)
df_bursts["cluster"] = clustering.labels_[n_clusters] + 1

# Define a color palette for the clusters
# palette = sns.color_palette(n_colors=n_clusters)  # "Set1", n_clusters)
# cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
# cluster_colors = [
#     f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
#     for c in cluster_colors
# ]
palette = get_cluster_colors(n_clusters)
cluster_colors = get_cluster_colors(n_clusters)


# %% inhibblock examples
if dataset == "inhibblock":
    examples_indices = [
        # ("control", 17, 12, 124),  # cluster 3 (green) - double peak
        ("control", 17, 6, 167),  # cluster 3 (green) - single peak
        ("bic", 17, 2, 115),
        ("control", 17, 4, 77),
        ("bic", 17, 7, 98),
    ]

    # fig, axs = plt.subplots(
    #     nrows=len(examples_indices), constrained_layout=True, figsize=(3 * cm, 5 * cm)
    # )
    # sns.despine()
    offset_start = 1000
    offset_end = 1000
    for i, index in enumerate(examples_indices):
        fig, ax = plt.subplots(constrained_layout=True, figsize=(4 * cm, 3 * cm))
        # ax = axs[i]
        ax_raster = ax.twinx()
        color = get_cluster_colors(n_clusters)[df_bursts.at[index, "cluster"] - 1]
        start, end = df_bursts.at[index, "start_orig"], df_bursts.at[index, "end_orig"]
        st, gid = get_inhibblock_spike_times(df_cultures, index[:-1])
        selection = (st >= start - offset_start) & (st <= end + offset_end)
        st, gid = st[selection], gid[selection]
        bins = np.linspace(start, end, num=51, endpoint=True)
        bins_mid = (bins[1:] + bins[:-1]) / 2

        ax_raster.scatter(st - start, gid, marker="|", s=15, color="k", alpha=0.1)
        y_time_bar = gid.min() - 2
        x_end = bins[-1] - start
        x_time_bar = [x_end - 500 + offset_end, x_end + offset_end]
        ax_raster.plot(x_time_bar, [y_time_bar, y_time_bar], color="k", linewidth=3)
        ax_raster.set_yticks([])

        ax.plot(
            bins_mid - start, df_bursts.at[index, "burst"], color=color, linewidth=2
        )
        ax.set_yticks([])
        # ax.set_xticks([0, 500])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlim(-offset_start, x_end + offset_end)

        # sns.despine(left=True, bottom=True)
        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_burst_example_{index}.svg"),
            transparent=True,
        )
