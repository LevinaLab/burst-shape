import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.persistence import load_burst_matrix, load_df_bursts
from src.persistence.agglomerative_clustering import get_agglomerative_labels

burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)
agglomerative_clustering_params = "agglomerating_clustering_linkage_complete"
# load bursts df
df_bursts = load_df_bursts(burst_extraction_params)
burst_matrix = load_burst_matrix(burst_extraction_params)

n_clusters = 9
palette = sns.color_palette(n_colors=n_clusters)
labels = get_agglomerative_labels(
    n_clusters, burst_extraction_params, agglomerative_clustering_params
)
df_bursts["cluster"] = labels

#####################################
# Choose a cluster to plot
i_cluster = 2
n_bursts_to_plot = 1  # number of bursts to plot
index_to_plot = None  # None plots a random one
plot_average = True  # plot average burst

# plot bursts
fig, ax = plt.subplots()
sns.despine()
df_bursts_i = df_bursts[df_bursts["cluster"] == i_cluster]
n_bursts_i = df_bursts_i.shape[0]

if plot_average:
    ax.plot(
        burst_matrix[df_bursts["cluster"] == i_cluster].mean(axis=0),
        color=palette[i_cluster - 1],
        alpha=1,
        linewidth=2,
    )

if index_to_plot is None:
    index_to_plot = np.random.randint(0, n_bursts_i, n_bursts_to_plot)
# if not iterable make a list
if not hasattr(index_to_plot, "__iter__"):
    index_to_plot = [index_to_plot]

for idx in index_to_plot:
    print(df_bursts_i.iloc[idx])
    bins = np.linspace(0, df_bursts_i.iloc[idx]["time_orig"], 51, endpoint=True)
    bins_mid = (bins[1:] + bins[:-1]) / 2
    ax.plot(
        bins_mid,
        df_bursts_i.iloc[idx]["burst"],
        color=palette[i_cluster - 1],
        alpha=1,
        linewidth=1,
    )
ax.set_ylabel("Rate [Hz]")
ax.set_xlabel("Time [ms]")
fig.show()

# %% count labels
print("Number of bursts per cluster")
print(df_bursts["cluster"].value_counts())
