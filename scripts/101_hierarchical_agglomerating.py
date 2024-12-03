"""Test hierarchical clustering with Wasserstein distance."""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster

from src import folders
from src.persistence import load_burst_matrix, load_df_bursts
from src.persistence.burst_extraction import _get_burst_folder

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)
agglomerating_clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
np.random.seed(0)

# plot settings
n_clusters = 3  # 9  # 3  # if None chooses the number of clusters with Davies-Bouldin index

# plotting
cm = 1 / 2.54  # centimeters in inches
fig_path = folders.get_fig_folder()

folder_agglomerating_clustering = os.path.join(
    _get_burst_folder(burst_extraction_params),
    agglomerating_clustering_params,
)
file_linkage = os.path.join(folder_agglomerating_clustering, "linkage.npy")

# load bursts
burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
np.random.seed(0)

if not os.path.exists(file_linkage):
    raise FileNotFoundError(f"Linkage file not found: {file_linkage}")
else:
    print(f"Loading linkage from {file_linkage}")
    Z = np.load(file_linkage)

# %% get clusters from linkage
print("Getting clusters from linkage...")
labels = fcluster(Z, t=n_clusters, criterion="maxclust")
df_bursts["cluster"] = labels

# %% Define a color palette for the clusters
palette = sns.color_palette(n_colors=n_clusters)  # "Set1", n_clusters)
cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
cluster_colors = [
    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
    for c in cluster_colors
]

# %% Plot dendrogram with colored clusters
print("Plotting dendrogram...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm))
sns.despine()

color_threshold = Z[-(n_clusters - 1), 2]

dendrogram_properties = dendrogram(
    Z,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=10,
    above_threshold_color="black",
    color_threshold=color_threshold,
)

# Highlight the clusters with background colors
# Count the number of elements in each cluster
cluster_counts = np.bincount(labels)[1:]  # ignoring cluster 0
cluster_index_dendrogram = [i for i in range(n_clusters) if cluster_counts[i] > 1]
# (fix the situation where a cluster is only size 1 -> not present in ax.collections)
for i, d in zip(cluster_index_dendrogram, ax.collections[:-1]):
    if i >= n_clusters:
        break
    color = cluster_colors[i]
    d.set_color(color)
ax.set_xlabel("Samples")
ax.set_ylabel("Distance")

# remove x and y ticks
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "dendrogram.svg"))
fig.savefig(os.path.join(fig_path, "dendrogram.pdf"))

# %% Dendrogram with equally sized clusters (leaves are the clusters themselves)
print("Plotting dendrogram with equally sized clusters...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm))
sns.despine()

color_threshold = Z[-(n_clusters - 1), 2]

dendrogram_properties = dendrogram(
    Z,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=10,
    above_threshold_color="black",
    truncate_mode="lastp",
    p=n_clusters,
    color_threshold=color_threshold,
    show_leaf_counts=False,
)

ax.set_xlabel("Clusters")
ax.set_ylabel("Distance")

# remove x and y ticks
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
fig.show()

# %% histogram of cluster sizes
print("Plotting cluster sizes...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
sns.despine()
# Count the number of elements in each cluster
cluster_counts = np.bincount(labels)[1:]  # ignoring cluster 0
# Create a bar plot with the correct colors
ax.bar(range(1, n_clusters + 1), cluster_counts, color=palette)
# write numerical values on top of the bars
for i, count in enumerate(cluster_counts):
    ax.text(i + 1, count, str(count), ha="center", va="bottom")
ax.set_xlabel("Cluster")
ax.set_ylabel("#Bursts")
ax.set_xticks(range(1, n_clusters + 1))
# fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "cluster_sizes.svg"))
fig.savefig(os.path.join(fig_path, "cluster_sizes.pdf"))

# %% plot average burst of each cluster
print("Plotting average bursts...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm))
sns.despine()
for cluster in range(1, n_clusters + 1):
    cluster_bursts = burst_matrix[labels == cluster]
    ax.plot(
        cluster_bursts.mean(axis=0),
        color=palette[cluster - 1],
        label=f"Cluster {cluster}",
    )
ax.set_xlabel("Time [a.u.]")
ax.set_ylabel("Rate [a.u.]")
# ax.legend()
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "average_bursts.svg"))
fig.savefig(os.path.join(fig_path, "average_bursts.pdf"))

# %% stats of each cluster
print("Computing stats of each cluster...")
for stat in [
    "time_orig",
    # "time_extend",
    # "peak_height",
    # "integral",
]:
    fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
    # fig.suptitle(f"Box plot of {stat}")
    sns.despine()
    sns.violinplot(
        x="cluster",
        y=stat,
        data=df_bursts,
        ax=ax,
        palette="Set1",
        log_scale=True,
        legend=False,
        hue="cluster",
        linewidth=0.5,
        inner=None,  # "quart",
    )

    if stat == "time_orig":
        ax.set_ylabel("Duration [ms]")
        ax.set_xlabel("Cluster")
        fig.savefig(os.path.join(fig_path, "time_orig.svg"))
        fig.savefig(os.path.join(fig_path, "time_orig.pdf"))

    fig.show()


print("Finished.")
