import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from src.folders import get_fig_folder
from src.persistence import load_clustering_labels, load_df_bursts, load_df_cultures
from src.persistence.spike_times import get_spike_times_in_milliseconds
from src.plot import get_cluster_colors, get_group_colors, prepare_plotting
from src.settings import (
    get_chosen_spectral_clustering_params,
    get_dataset_from_burst_extraction_params,
)

cm = prepare_plotting()
color_by = ["spectral_cluster", "group"][1]

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Detected dataset: {dataset}")

# load bursts
df_bursts = load_df_bursts(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)
np.random.seed(0)

if color_by == "spectral_cluster":
    # which clustering to plot
    clustering_params, n_clusters = get_chosen_spectral_clustering_params(dataset)
    col_cluster = f"cluster_{n_clusters}"

    labels_params = "labels"
    cv_params = "cv"  # if cv_split is not None, chooses the cross-validation split
    cv_split = None  # set to None for plotting the whole clustering, set to int for specific split

    clustering = load_clustering_labels(
        clustering_params, burst_extraction_params, labels_params, cv_params, cv_split
    )
    df_bursts["cluster"] = clustering.labels_[n_clusters] + 1

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
        match color_by:
            case "spectral cluster":
                color = get_cluster_colors(n_clusters)[
                    df_bursts.at[index, "cluster"] - 1
                ]
            case "group":
                color = get_group_colors(dataset)[index[0]]
            case _:
                raise NotImplementedError(f"color_by={color_by} not implemented.")
        start, end = df_bursts.at[index, "start_orig"], df_bursts.at[index, "end_orig"]
        st, gid = get_spike_times_in_milliseconds(df_cultures, index[:-1], dataset)
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

        # ax.plot(
        #     bins_mid - start, df_bursts.at[index, "burst"], color=color, linewidth=2
        # )

        # plot firing rate
        bin_size = 30
        hist, bin_edges = np.histogram(
            st - start,
            bins=np.linspace(
                -offset_start,
                end - start + offset_end,
                int((end - start + offset_start + offset_end) // bin_size),
            ),
        )
        hist = hist / (bin_edges[1] - bin_edges[0])
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        mask = (bin_centers > 0) & (bin_centers < end - start)
        x = bin_centers
        y = hist
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color: red inside mask, gray outside
        colors = np.where(mask[:-1] | mask[1:], color, "k")

        # Create LineCollection
        lc = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(lc)

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlim(-offset_start, x_end + offset_end)
        ax.set_ylim(-y.max() * 0.15, y.max() * 1.1)

        # sns.despine(left=True, bottom=True)
        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_burst_example_{index}.svg"),
            transparent=True,
        )
