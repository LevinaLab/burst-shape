"""Visualize the burst cluster similar to the Figure 3 in Wagenaar et al. 2006."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.folders import get_fig_folder
from src.persistence import load_clustering_labels, load_df_bursts, load_df_cultures
from src.plot import get_cluster_colors

# which clustering to plot
n_clusters = 3
col_cluster = f"cluster_{n_clusters}"

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
else:
    dataset = "wagenaar"
print(f"Detected dataset: {dataset}")

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

# %% build new dataframe df_cultures with index ('batch', 'culture', 'day') and columns ('n_bursts', 'cluster_abs', 'cluster_rel')
match dataset:
    case "kapucu":
        index_names = ["culture_type", "mea_number", "well_id", "DIV"]
    case "wagenaar":
        index_names = ["batch", "culture", "day"]
    case "hommersom":
        index_names = ["batch", "clone", "well_idx"]
    case "inhibblock":
        index_names = ["drug_label", "div", "well_idx"]
    case _:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
df_cultures_test = df_bursts.groupby(index_names).agg(
    n_bursts=pd.NamedAgg(column="i_burst", aggfunc="count")
)
try:
    for index in tqdm(df_cultures_test.index):
        assert index in df_cultures.index
        assert (
            df_cultures_test.at[index, "n_bursts"] == df_cultures.at[index, "n_bursts"]
        )
except AssertionError:  # TODO delete legacy code
    print(
        "Still the old implementation where df_cultures is inconsistent with df_bursts."
    )
    for index in tqdm(df_cultures.index):
        if index in df_cultures_test.index:
            continue
        df_cultures_test.at[index, "n_bursts"] = 0
    df_cultures = df_cultures_test
del df_cultures_test

# for all unique combinations of batch and culture
unique_batch_culture = df_cultures.reset_index()[index_names[:-1]].drop_duplicates()
# sort by batch and culture
unique_batch_culture.sort_values(index_names[:-1], inplace=True)
# assign an index to each unique combination
unique_batch_culture["i_culture"] = np.arange(len(unique_batch_culture))  # [::-1]
unique_batch_culture.set_index(index_names[:-1], inplace=True)

df_cultures["i_culture"] = pd.Series(
    data=(
        df_cultures.reset_index().apply(
            lambda x: unique_batch_culture.loc[
                tuple(
                    [x[index_label] for index_label in index_names[:-1]]
                ),  # (x["batch"], x["culture"]),
                "i_culture",
            ],
            axis=1,
        )
    ).values,
    index=df_cultures.index,
    dtype=int,
)

# %% add cluster information
for i_cluster in range(n_clusters):
    col_cluster = f"cluster_{n_clusters}"
    df_cultures[f"cluster_abs_{i_cluster}"] = df_bursts.groupby(
        index_names  # ["batch", "culture", "day"]
    )[col_cluster].agg(lambda x: np.sum(x == i_cluster))
    df_cultures[f"cluster_rel_{i_cluster}"] = df_cultures[
        f"cluster_abs_{i_cluster}"
    ] / df_cultures["n_bursts"].astype(float)

# %% plot a pie chart for each entry in df_cultures
# position of the pie chart in the grid is determined by the day and i_culture
# colors = sns.color_palette("Set1", n_colors=n_clusters)
match dataset:
    case "kapucu" | "wagenaar":
        figsize = (15, 12)
    case "hommersom":
        figsize = (8, 6)
    case "inhibblock":
        figsize = (3, 5)
    case _:
        figsize = (15, 12)

colors = get_cluster_colors(n_clusters)
ncols = df_cultures["i_culture"].max() + 1
row_day = (
    df_cultures.index.get_level_values(index_names[-1]).unique().sort_values().to_list()
)
# consider filling in missing days
# nrows = df_cultures.index.get_level_values("day").max() + 1
nrows = len(row_day)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)

# set all axes to invisible
for ax in axs.flatten():
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.pie([1], colors=["white"], startangle=90)
for index in df_cultures.index:
    i_day = row_day.index(index[len(index_names) - 1])
    i_culture = df_cultures.loc[index, "i_culture"]
    ax = axs[i_day, i_culture]
    ax.axis("on")
    # ax.set_title(f"Day {i_day} - Culture {i_culture}")
    cluster_rel = [
        df_cultures.loc[index, f"cluster_rel_{i_cluster}"]
        for i_cluster in range(n_clusters)
    ]
    if df_cultures.at[index, "n_bursts"] == 0:
        # ax.pie([1], colors=["grey"], startangle=90)
        ax.pie([1], colors=["white"], wedgeprops=dict(width=0, edgecolor="grey"))
    else:
        ax.pie(cluster_rel, colors=colors, startangle=90)


# Add a shared y-axis for days
for i_day, day in enumerate(row_day):
    ax = axs[i_day, 0]
    ax.axis("on")
    ax.set_ylabel(f"{day}", fontsize=12, rotation=0)
    # ax.xaxis.set_label_position("bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.pie([1], colors=["white"], startangle=90)

# write batches on the top
# batch_label_pos = np.linspace(0.04, 0.97, nrows, endpoint=True)[::-1]
for (index), row in unique_batch_culture.iterrows():
    i_culture = row["i_culture"]
    ax = axs[0, i_culture]
    match dataset:
        case "wagenaar":
            (batch, culture) = index
            ax.set_title(f"{batch}-{culture}", fontsize=12)
        case "kapucu":
            (culture_type, mea_number, well_id) = index
            ax.set_title(
                f"{culture_type}-{mea_number}-{well_id}", fontsize=12, rotation=90
            )
        case "hommersom":
            (batch, clone) = index
            ax.set_title(f"{batch}-{clone}", fontsize=12, rotation=90)
        case "inhibblock":
            (drug_label, div) = index
            ax.set_title(f"{drug_label}-{div}", fontsize=12, rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
"""fig.legend(
    [f"Cluster {i_cluster}" for i_cluster in range(n_clusters)],
    loc="center right",
    frameon=False,
)"""
fig.tight_layout()
fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_pie_chart.svg"), transparent=True
)
