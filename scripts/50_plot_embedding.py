import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence import load_clustering_labels, load_df_bursts, load_df_cultures
from src.plot import get_cluster_colors, get_group_colors, prepare_plotting

cm = prepare_plotting()

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
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
np.random.seed(0)

#  get clusters from linkage
# print("Getting clusters from linkage...")
# labels = get_agglomerative_labels(
#     n_clusters, burst_extraction_params, agglomerating_clustering_params
# )
clustering = load_clustering_labels(
    clustering_params, burst_extraction_params, labels_params, cv_params, cv_split
)
df_bursts["cluster"] = clustering.labels_[n_clusters] + 1
spectral_columns = [f"Spec.-Dim. {dim}" for dim in range(1, 11)]
df_bursts.loc[:, spectral_columns] = clustering.maps_[:, 1:11]

# Define a color palette for the clusters
# palette = sns.color_palette(n_colors=n_clusters)  # "Set1", n_clusters)
# cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
# cluster_colors = [
#     f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
#     for c in cluster_colors
# ]
# palette = get_cluster_colors(n_clusters)
# cluster_colors = get_cluster_colors(n_clusters)

# %% plot embedding
plot_density = True
color_by = ["cluster", "group"][0]

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

s = 5
if dataset == "mossink":
    s = 1

fig, ax = plt.subplots(1, 1, figsize=(4 * cm, 4 * cm), constrained_layout=True)
sns.despine(left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])
if plot_density:
    sns.kdeplot(
        data=df_bursts,
        x="Spec.-Dim. 1",
        y="Spec.-Dim. 2",
        levels=7,
        alpha=0.5,
        color="k",
    )
sns.scatterplot(
    data=df_bursts,
    x="Spec.-Dim. 1",
    y="Spec.-Dim. 2",
    s=s,
    alpha=0.4,
    hue=color,  # "cluster",
    hue_order=hue_order,  # sorted(df_bursts["cluster"].unique()),
    palette=palette,  # get_cluster_colors(n_clusters),
    legend=False,
)
fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_spectral_embedding_{color_by}.svg"),
    transparent=True,
)
