import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster

from src.folders import get_results_folder
from src.persistence import load_df_bursts

burst_extraction_params = "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
# load bursts df
df_bursts = load_df_bursts(burst_extraction_params)
# load labels
linkage_file = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    "agglomerating_clustering_linkage_complete_n_bursts_None",
    "linkage.npy",
)
linkage = np.load(linkage_file)

n_clusters = 9
palette = sns.color_palette('deep', n_colors=n_clusters)
labels = fcluster(linkage, t=n_clusters, criterion="maxclust")
df_bursts["cluster"] = labels

#####################################
# %%
# Choose a cluster to plot
i_cluster = 9
n_bursts_to_plot = 10  # number of bursts to plot
index_to_plot = None  # None plots a random one

# plot bursts
fig, ax = plt.subplots()
sns.despine()
df_bursts_i = df_bursts[df_bursts["cluster"] == i_cluster]
n_bursts_i = df_bursts_i.shape[0]

if index_to_plot is None:
    index_to_plot = np.random.randint(0, n_bursts_i, n_bursts_to_plot)
# if not iterable make a list
if not hasattr(index_to_plot, "__iter__"):
    index_to_plot = [index_to_plot]

for idx in index_to_plot:
    print(df_bursts_i.iloc[idx])
    bins = np.linspace(
        0, df_bursts_i.iloc[idx]["time_orig"], 51, endpoint=True
    )
    bins_mid = (bins[1:] + bins[:-1]) / 2
    ax.plot(
        bins_mid,
        df_bursts_i.iloc[idx]["burst"],
        color=palette[i_cluster - 1],
        alpha=0.4,
        linewidth=1,
    )
ax.set_ylabel("Rate [Hz]")
ax.set_xlabel("Time [ms]")
fig.show()

# %%  Plot examples of all 9 clusters

# %%
# Choose a cluster to plot
from scipy.ndimage import gaussian_filter

idx_bursts_to_plot = [
    [142],  # 1
    [0],
    [201],  # 6,7,59
    [4],  # 4
    [13],  # 5, 13
    [51],  # 17, 40
    [70],
    [21],  # 3,21
    [75],  # 45
]

# plot bursts
fig, ax = plt.subplots(3, 3)
ax = np.hstack(ax)
sns.despine()
for i_cluster in range(1, len(idx_bursts_to_plot) + 1):
    print(i_cluster)
    df_bursts_i = df_bursts[df_bursts["cluster"] == i_cluster].reset_index()
    n_bursts_i = df_bursts_i.shape[0]
    for idx in idx_bursts_to_plot[i_cluster - 1]:
        if idx is None:
            continue

        # Get batch,name,day
        batch, cult, day = list(df_bursts_i.iloc[idx][['batch', 'culture', 'day']])
        # get begining and end
        start = df_bursts_i.iloc[idx]['start_orig']
        end = df_bursts_i.iloc[idx]['end_orig']
        st, gid = np.loadtxt('../data/extracted/%s-%s-%s.spk.txt' % (batch, cult, day)).T
        st = st * 1000
        # fig, ax = plt.subplots()
        ax[i_cluster - 1].plot(st, gid, '|', ms=1, color=palette[i_cluster - 1], alpha=0.5)
        sc, bins = np.histogram(st, np.arange(0, np.max(st), 50))
        ax[i_cluster - 1].plot(bins[1:], 20 * (sc / np.max(sc)) - 30, color=palette[i_cluster - 1])
        ax[i_cluster - 1].set_xlim(start - 800, start + 2000)  # end+500)
        ax[i_cluster - 1].axvline(start, color='gray', linestyle='--', alpha=0.5)
        ax[i_cluster - 1].axis('off')
        if i_cluster == 7:
            ax[i_cluster - 1].plot([start - 800, (start - 800) + 500], [-10, -9], color='k')

    #     bins = np.linspace(
    #         0, df_bursts_i.iloc[idx]["time_orig"], 51, endpoint=True
    #     )
    #     bins_mid = (bins[1:] + bins[:-1]) / 2
    #     ax.plot(
    #         bins_mid,
    #         gaussian_filter(df_bursts_i.iloc[idx]["burst"],2),
    #         color=palette[i_cluster - 1],
    #         alpha=0.4,
    #         linewidth=1,
    #     )
    # ax.set_ylabel("Rate [Hz]")
    # ax.set_xlabel("Time [ms]")
    fig.show()
plt.savefig('../figures/examples_raster3.pdf')

# %% Plot development of one culture
fig, ax = plt.subplots(30, 1, figsize=(4, 10))
ax = np.hstack(ax)
for day in range(1, 30):
    batch, cult = 6, 1
    ax[day - 1].axis('off')
    # get begining and end
    try:
        start = 100000
        end = 600000
        st, gid = np.loadtxt('../data/extracted/%s-%s-%s.spk.txt' % (batch, cult, day)).T
        st = st * 1000
        # fig, ax = plt.subplots()
        # ax[day-1].plot(st,gid,'|',ms=1,color='k',alpha=0.5)
        sc, bins = np.histogram(st, np.arange(0, np.max(st), 50))
        ax[day - 1].plot(bins[1:], sc, color='gray')
        ax[day - 1].set_xlim(start, end)  # end+500)
        # ax[i_cluster-1].axvline(start,color='gray',linestyle='--',alpha=0.5)
        ax[day - 1].set_ylim([0, 2000])
        ax[day - 1].text(100000, 0, day)
    except:
        continue
plt.savefig('../figures/development-6-1.pdf')

# %%
