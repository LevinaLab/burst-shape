"""
Plot spectral embedding.

There are several options for coloring the spectral embedding:
- 'group' will color by group color (drug group, batches, ...)
- 'cluster' will plot spectral clusters
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.persistence import (
    load_burst_matrix,
    load_clustering_labels,
    load_df_bursts,
    load_df_cultures,
    load_spectral_embedding,
)
from src.plot import get_cluster_colors, get_group_colors, prepare_plotting, savefig
from src.settings import (
    get_chosen_spectral_clustering_params,
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)

# -----------------------------------------------------------------------------
# Settings for plot
cm = prepare_plotting()
plot_density = False
color_by = ["cluster", "group", "relative_peak"][1]
# s = lambda dataset: 1 if dataset == "mossink" else 5
s = lambda dataset: 1
dim1 = 1  # usually should be 1
dim2 = 2  # usually should be 2
figsize = (4 * cm, 4 * cm)

# -----------------------------------------------------------------------------
# Parameters which data to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)

# -----------------------------------------------------------------------------
# Load data
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
match color_by:
    case "cluster":
        clustering_params, n_clusters = get_chosen_spectral_clustering_params(dataset)
    case "group":
        clustering_params = get_chosen_spectral_embedding_params(dataset)
    case "relative_peak":
        burst_matrix = load_burst_matrix(burst_extraction_params)
print(f"Detected dataset: {dataset}")

# manually selecting clustering_params
# clustering_params = "spectral_affinity_precomputed_metric_euclidean_n_neighbors_85"
# clustering_params = "spectral_affinity_precomputed_metric_euclidean_n_neighbors_21"

labels_params = "labels"
cv_params = "cv"  # if cv_split is not None, chooses the cross-validation split
cv_split = (
    None  # set to None for plotting the whole clustering, set to int for specific split
)
# load bursts
df_bursts = load_df_bursts(burst_extraction_params)
np.random.seed(0)

#  get embedding
spectral_columns = [f"Spec.-Dim. {dim}" for dim in range(1, 11)]
df_bursts.loc[:, spectral_columns] = load_spectral_embedding(
    burst_extraction_params,
    clustering_params,
    n_dims=10,
)
if color_by == "cluster":
    # which clustering to plot
    col_cluster = f"cluster_{n_clusters}"
    clustering = load_clustering_labels(
        clustering_params, burst_extraction_params, labels_params, cv_params, cv_split
    )
    df_bursts["cluster"] = clustering.labels_[n_clusters] + 1
if color_by == "relative_peak":
    df_bursts["relative_peak"] = np.argmax(burst_matrix, axis=1)

# %% Plot embedding
hue_order = None
match color_by:
    case "cluster":
        color = "cluster"
        hue_order = sorted(df_bursts["cluster"].unique())
        palette = get_cluster_colors(n_clusters)
    case "group":
        color = "batch"
        match dataset:
            case "inhibblock":
                color = "drug_label"
                palette = get_group_colors(dataset)
            case "wagenaar":
                df_bursts.reset_index(inplace=True)
                color_discrete_map_load = get_group_colors(dataset)
                palette = {}
                for key, value in color_discrete_map_load.items():
                    palette[key] = value
            case "kapucu":
                index_names = df_bursts.index.names
                df_bursts.reset_index(inplace=True)
                df_bursts["batch"] = (
                    df_bursts.reset_index()["culture_type"].astype(str)
                    + "-"
                    + df_bursts.reset_index()["mea_number"].astype(str)
                )
                color_discrete_map_load = get_group_colors(dataset)
                palette = {}
                for key, value in color_discrete_map_load.items():
                    palette["-".join(key)] = value
                df_bursts.set_index(index_names, inplace=True)
            case "mossink":
                color = "group-subject"
                df_bursts.reset_index(inplace=True)
                df_bursts["group-subject"] = (
                    df_bursts.reset_index()["group"].astype(str)
                    + " "
                    + df_bursts.reset_index()["subject_id"].astype(str)
                )
                palette = get_group_colors(dataset)
            case "hommersom_binary":
                df_bursts["group"] = pd.Series()
                df_cultures = load_df_cultures(burst_extraction_params)
                for index_culture in df_cultures.index:
                    df_bursts.loc[index_culture, "group"] = df_cultures.at[
                        index_culture, "group"
                    ]
                color = "group"
                palette = get_group_colors(dataset)
            case _:
                raise NotImplementedError
    case "relative_peak":
        color = "relative_peak"
        palette = "viridis"  # "RedBlue"

fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
sns.despine(left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])
if plot_density:
    sns.kdeplot(
        data=df_bursts,
        x=f"Spec.-Dim. {dim1}",
        y=f"Spec.-Dim. {dim2}",
        levels=7,
        alpha=0.5,
        color="k",
    )
efficient = True
if efficient:
    for hue in (
        df_bursts.reset_index()[color].unique() if hue_order is None else hue_order
    ):
        subset = df_bursts.reset_index()[color] == hue
        ax.plot(
            df_bursts[f"Spec.-Dim. {dim1}"].values[subset],
            df_bursts[f"Spec.-Dim. {dim2}"].values[subset],
            linestyle="",
            marker="o",
            markersize=np.sqrt(s(dataset)),
            alpha=0.4,
            color=palette[hue] if isinstance(palette, dict) else None,
            markerfacecolor=palette[hue] if isinstance(palette, dict) else None,
            markeredgecolor=None,  # border color
            markeredgewidth=0,  # border thickness
        )
else:
    sns.scatterplot(
        data=df_bursts,
        x=f"Spec.-Dim. {dim1}",
        y=f"Spec.-Dim. {dim2}",
        s=s(dataset),
        alpha=0.4,
        hue=color,  # "cluster",
        hue_order=hue_order,  # sorted(df_bursts["cluster"].unique()),
        palette=palette,  # get_cluster_colors(n_clusters),
        legend=False,
    )
fig.show()
savefig(
    fig,
    f"{dataset}_spectral_embedding_{color_by}{'_density' if plot_density else ''}_dim_{dim1}_{dim2}.svg",
    file_format=["pdf", "svg"],
)
