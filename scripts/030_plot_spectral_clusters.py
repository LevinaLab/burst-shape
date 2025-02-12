import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.folders import get_fig_folder
from src.persistence import load_clustering_labels, load_df_bursts
from src.plot import get_cluster_colors, make_cluster_legend

# which clustering to plot
n_clusters = 6
col_cluster = f"cluster_{n_clusters}"

# parameters which clustering to plot
# parameters which clustering to evaluate
burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
)
clustering_params = (
    # "agglomerating_clustering_linkage_complete"
    # "agglomerating_clustering_linkage_ward"
    # "agglomerating_clustering_linkage_average"
    # "agglomerating_clustering_linkage_single"
    # "spectral_affinity_precomputed_metric_wasserstein"
    "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
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

# %% plot clusters
# bar plot of cluster sizes
fig, ax = plt.subplots()
sns.despine()
sns.countplot(x=col_cluster, hue=col_cluster, data=df_bursts, ax=ax, palette="Set1")
fig.show()

for category in ["batch", "day"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.despine()
    sns.countplot(
        x=category,
        hue=col_cluster,
        data=df_bursts,
        ax=ax,
        native_scale=True,
        palette="Set1",
    )
    fig.show()
# %%
facet_grid = sns.catplot(
    kind="count",
    col="batch",
    x="culture",
    hue=col_cluster,
    data=df_bursts,
    palette="Set1",
    height=5,
    aspect=0.4,
    sharey=False,
    sharex=False,
)
facet_grid.set_titles("batch {col_name}")
facet_grid.fig.show()


# %% plot bursts
n_plot = 100
fig, ax = plt.subplots()
fig.suptitle("Example bursts")
sns.despine()
for i in range(n_clusters):
    df_bursts_i = df_bursts[df_bursts[col_cluster] == i]
    n_bursts_i = df_bursts_i.shape[0]
    idx_random = np.random.randint(
        0,
        n_bursts_i,
        n_plot,
    )
    for j, idx in enumerate(idx_random):
        ax.plot(
            df_bursts_i.iloc[idx]["burst"],
            color=sns.color_palette("Set1")[i],
            alpha=0.5,
            label=f"Cluster {i}" if j == 0 else None,
        )
ax.legend()
fig.show()

# %% plot burst with real time
n_plot = 100
fig, ax = plt.subplots()
fig.suptitle("Example bursts")
sns.despine()
for i in range(n_clusters):
    df_bursts_i = df_bursts[df_bursts[col_cluster] == i]
    n_bursts_i = df_bursts_i.shape[0]
    idx_random = np.random.randint(
        0,
        n_bursts_i,
        n_plot,
    )
    for j, idx in enumerate(idx_random):
        bins = np.linspace(
            -50, df_bursts_i.iloc[idx]["time_extend"] - 50, 51, endpoint=True
        )
        bins_mid = (bins[1:] + bins[:-1]) / 2
        ax.plot(
            bins_mid,
            df_bursts_i.iloc[idx]["burst"],
            color=sns.color_palette("Set1")[i],
            alpha=0.5,
            label=f"Cluster {i}" if j == 0 else None,
        )
ax.legend(frameon=False)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Rate [Hz]")
fig.show()

# %% plot average burst per cluster
cm = 1 / 2.54
fig, ax = plt.subplots(constrained_layout=True)
fig.suptitle("Average burst per cluster")
sns.despine()
for i in range(n_clusters):
    df_bursts_i = df_bursts[df_bursts[col_cluster] == i]
    ax.plot(
        df_bursts_i["burst"].mean(),
        # color from set1 palette
        color=get_cluster_colors(n_clusters)[i],
        label=f"Cluster {i}",
        linewidth=2,
    )
ax.legend(frameon=False)
fig.show()

fig.suptitle("")
ax.get_legend().remove()
fig.set_size_inches((4 * cm, 4 * cm))
fig.savefig(os.path.join(get_fig_folder(), "average_bursts.svg"), transparent=True)

# %% box plot of statistics (time_orig, time_extend, peak_height, integral)
for stat in ["time_orig", "time_extend", "peak_height", "integral"]:
    fig, ax = plt.subplots()
    # fig.suptitle(f"Box plot of {stat}")
    sns.despine()
    sns.boxplot(
        x=col_cluster, y=stat, data=df_bursts, ax=ax, palette="Set1", log_scale=True
    )

    # if stat == "peak_height":
    #     ax.set_yscale("log")

    fig.show()

# %% plot average burst per cluster
cm = 1 / 2.54
fig, ax = make_cluster_legend(n_clusters, n_cols=2, symbol="dot")
fig.set_size_inches((4 * cm, 4 * cm))
fig.show()

fig.savefig(
    os.path.join(get_fig_folder(), "cluster_legend.svg"),
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
