"""Visualize the burst cluster similar to the Figure 3 in Wagenaar et al. 2006."""
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence import load_clustering_labels, load_df_bursts, load_df_cultures
from src.pie_chart.pie_chart import (
    get_df_cultures_subset,
    plot_df_culture_layout,
    prepare_df_cultures_layout,
)
from src.plot import get_cluster_colors, get_group_colors, prepare_plotting
from src.settings import get_dataset_from_burst_extraction_params

cm = prepare_plotting()

# SET TO TRUE TO PLOT ONLY A SUBSET
plot_subset = True

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
match dataset:
    case "kapucu":
        n_clusters = 4
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
        )
    case "hommersom":
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_6"
        )
        n_clusters = 4
    case "inhibblock":
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
        )
        n_clusters = 4
    case "mossink":
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
        )
        n_clusters = 4
    case "wagenaar":
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
        )
        n_clusters = 6
    case _:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
print(f"Detected dataset: {dataset}")

# which clustering to plot
col_cluster = f"cluster_{n_clusters}"

labels_params = "labels"
cv_params = "cv"  # if cv_split is not None, chooses the cross-validation split
cv_split = (
    None  # set to None for plotting the whole clustering, set to int for specific split
)

# load data
clustering = load_clustering_labels(
    clustering_params, burst_extraction_params, labels_params, cv_params, cv_split
)
df_bursts = load_df_bursts(burst_extraction_params, cv_params=cv_params)
if cv_split is not None:
    df_bursts = df_bursts[df_bursts[f"cv_{cv_split}_train"]]
for n_clusters_ in clustering.n_clusters:
    df_bursts[f"cluster_{n_clusters_}"] = clustering.labels_[n_clusters_]

# index of df_bursts is ('batch', 'culture', 'day', 'i_burst')
df_bursts.reset_index(inplace=True)

df_cultures = load_df_cultures(burst_extraction_params)

# %% subsample
if plot_subset is True:
    df_cultures = get_df_cultures_subset(df_cultures, dataset)

# %% for all unique combinations of batch and culture
df_cultures, unique_batch_culture = prepare_df_cultures_layout(df_cultures)
# %% add cluster information
for i_cluster in range(n_clusters):
    col_cluster = f"cluster_{n_clusters}"
    df_cultures[f"cluster_abs_{i_cluster}"] = df_bursts.groupby(
        df_cultures.index.names  # ["batch", "culture", "day"]
    )[col_cluster].agg(lambda x: np.sum(x == i_cluster))
    df_cultures[f"cluster_rel_{i_cluster}"] = df_cultures[
        f"cluster_abs_{i_cluster}"
    ] / df_cultures["n_bursts"].astype(float)

# %% plot a pie chart for each entry in df_cultures
# position of the pie chart in the grid is determined by the day and i_culture
# colors = sns.color_palette("Set1", n_colors=n_clusters)
match dataset:
    case "kapucu":
        if plot_subset is True:
            figsize = (4.5 * cm, 7 * cm)
        else:
            figsize = (12 * cm, 10 * cm)
    case "wagenaar":
        if plot_subset is True:
            figsize = (5 * cm, 8 * cm)
        else:
            figsize = (12 * cm, 10 * cm)
    case "hommersom":
        figsize = (8, 6)
    case "inhibblock":
        figsize = (3.5 * cm, 9 * cm)
    case "mossink":
        if plot_subset is True:
            figsize = (10 * cm, 9 * cm)
        else:
            figsize = (10 * cm, 30 * cm)
    case _:
        figsize = (15, 12)

column_names = [f"cluster_rel_{i_cluster}" for i_cluster in range(n_clusters)]
fig, axs = plot_df_culture_layout(
    df_cultures,
    figsize,
    dataset,
    column_names,
    get_cluster_colors(n_clusters),
    unique_batch_culture,
)

fig.tight_layout()
fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
fig.show()
fig.savefig(
    os.path.join(
        get_fig_folder(), f"{dataset}_pie_chart{'_subset' if plot_subset else ''}.svg"
    ),
    transparent=True,
)

# %% special plate layout for inhibblock data
draw_rectangles = True
if dataset == "inhibblock":
    for div in [17, 18]:
        df_cultures_special = df_cultures.reset_index()
        df_cultures_special = df_cultures_special[df_cultures_special["div"] == div]
        df_cultures_special["layout_column"] = df_cultures_special["well"].str[0]
        df_cultures_special["layout_row"] = df_cultures_special["well"].str[1]
        df_cultures_special.set_index(["layout_row", "layout_column"], inplace=True)

        df_cultures_special, unique_batch_culture_special = prepare_df_cultures_layout(
            df_cultures_special
        )
        fig, axs = plot_df_culture_layout(
            df_cultures_special,
            (4 * cm, 3 * cm),
            dataset,
            column_names,
            get_cluster_colors(n_clusters),
            unique_batch_culture_special,
        )

        fig.tight_layout()
        # fig.subplots_adjust(wspace=-0.15, hspace=-0.15)

        if draw_rectangles:
            # draw Bicuculline rectangle
            margin = 0.045
            bbox1 = axs[0, 1].get_position(fig)
            bbox2 = axs[1, -1].get_position(fig)
            x0 = bbox1.x0  # Left
            y0 = bbox2.y0 - margin  # Bottom
            width = bbox2.x1 - x0
            height = bbox1.y1 - y0
            # Create a rectangle and add it to the figure
            rect = patches.Rectangle(
                (x0, y0),
                width,
                height,
                transform=fig.transFigure,  # Figure-level transformation
                color=get_group_colors(dataset)["bic"],
                linewidth=2,
                fill=False,
            )
            fig.add_artist(rect)

            # draw Control rectangle
            margin_x = 0.07
            margin_y = 0.045
            bbox1 = axs[0, 0].get_position(fig)
            bbox2 = axs[2, 0].get_position(fig)
            bbox3 = axs[3, -1].get_position(fig)
            L_shape = [
                (bbox1.x0, bbox1.y1),  # Top-left
                (bbox1.x1 + margin_x, bbox1.y1),  # Top-right
                (bbox1.x1 + margin_x, bbox2.y1 + margin_y),  # Middle-right
                (bbox3.x1, bbox2.y1 + margin_y),  # Bottom-right
                (bbox3.x1, bbox3.y0),  # Bottom
                (bbox2.x0, bbox3.y0),  # Bottom-left
                (bbox2.x0, bbox1.y1),  # Back to Top-left
            ]
            L_patch = patches.Polygon(
                L_shape,
                transform=fig.transFigure,
                edgecolor=get_group_colors(dataset)["control"],
                fill=False,
                alpha=1,
                linewidth=2,
            )

            fig.patches.append(L_patch)

            # fig.subplots_adjust(wspace=0.0, hspace=-0.0)
        fig.subplots_adjust(wspace=-0.15, hspace=-0.15)

        fig.show()
        fig.savefig(
            os.path.join(
                get_fig_folder(), f"{dataset}_special_div_{div}_pie_chart.svg"
            ),
            transparent=True,
        )
