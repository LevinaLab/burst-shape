import os

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster

from src.folders import get_data_folder, get_data_kapucu_folder
from src.persistence import load_df_bursts, load_df_cultures
from src.persistence.agglomerative_clustering import get_agglomerative_labels
from src.persistence.burst_extraction import _get_burst_folder

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)
agglomerating_clustering_params = "agglomerating_clustering_linkage_complete"
n_clusters = 5

plot_raster = False

dataset = "kapucu" if "kapucu" in burst_extraction_params else "wagenaar"

# load data
df_cultures = load_df_cultures(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)

print("Getting clusters from linkage...")
df_bursts["cluster"] = get_agglomerative_labels(
    n_clusters, burst_extraction_params, agglomerating_clustering_params
)

# Define a color palette for the clusters
palette = sns.color_palette(n_colors=n_clusters)  # "Set1", n_clusters)
cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
cluster_colors = [
    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
    for c in cluster_colors
]

# %% Plot examples of burst extraction
# idx = 110  # batch 1, culture 1, day 19
bin_size = 10  # ms

# choose index
match dataset:
    case "wagenaar":
        idx = (1, 1, 19)
    case "kapucu":
        idx = (
            # ('Rat', 'MEA1', 'C1', 21)
            # ('Rat', 'MEA1', 'C1', 24)
            # ('Rat', 'MEA1', 'C1', 28)
            ("Rat", "MEA1", "C1", 31)
            # ('hPSC', 'MEA1', 'C3', 21)
            # ('hPSC', 'MEA1', 'C3', 24)
            # ('hPSC', 'MEA1', 'C3', 28)
            # ('hPSC', 'MEA1', 'C3', 31)
        )
match dataset:
    case "wagenaar":
        data = (
            np.loadtxt(
                os.path.join(
                    get_data_folder(), "extracted", df_cultures.loc[idx].file_name
                )
            )[:, 0]
            * 1000
        )
    case "kapucu":
        data = df_cultures.loc[idx].times * 1000
bins = np.arange(0, data.max() + bin_size, bin_size)
# histogram and convert to Hz
data = np.histogram(data, bins=bins)[0] / (bin_size / 1000)

if plot_raster:
    match dataset:
        case "wagenaar":
            st, gid = np.loadtxt(
                os.path.join(
                    get_data_folder(), "extracted", df_cultures.loc[idx].file_name
                )
            ).T
            st *= 1000
            gid = gid.astype(int)
        case "kapucu":
            culture_type, mea_number, well_id, div_day = idx
            # print(f"culture_type: {culture_type}, mea_number: {mea_number}, well_id: {well_id}, div_day: {div_day}")

            # Load spike data
            data_folder = get_data_kapucu_folder()
            path_time_series = os.path.join(
                data_folder,
                "_".join(
                    [
                        culture_type,
                        "20517" if culture_type == "hPSC" else "190617",
                        mea_number,
                        f"DIV{div_day}",
                        "spikes.csv",
                    ]
                ),
            )
            # print(path_time_series)
            # print(os.path.exists(path_time_series))
            spikes = pd.read_csv(path_time_series)
            # print(spikes)
            spikes[["well", "ch_n"]] = spikes["Channel"].str.split("_", expand=True)
            # print(spikes)
            spikes = spikes[spikes["well"] == well_id]
            # print(spikes)
            st = spikes["Time"].values
            gid = spikes["ch_n"].values.astype(int)
            st = st * 1000

fig, axs = plt.subplots(
    2 if plot_raster else 1,
    1,
    figsize=(15, 10 if plot_raster else 5),
    constrained_layout=True,
    sharex=True,
)
axs = [axs] if not plot_raster else axs
ax = axs[0]
match dataset:
    case "wagenaar":
        batch, culture, day = df_cultures.loc[idx].name
        fig.suptitle(
            f"Bursts: batch {batch}, culture {culture}, day {day}"
            f"\n{df_cultures.loc[idx].n_bursts} bursts"
        )
    case "kapucu":
        culture_type, mea_number, well_id, DIV = idx  # df_cultures.loc[idx].name
        fig.suptitle(
            f"Bursts: {culture_type} {mea_number}, {well_id}, DIV {DIV}"
            f"\n{df_cultures.loc[idx].n_bursts} bursts"
        )
sns.despine()
ax.plot(bins[:-1], data, color="k")
# highlight burst from df_bursts
ax.set_xlabel(f"Time [ms], {bin_size} ms bins")
ax.set_ylabel("Rate [Hz]")

# add legend for clusters
handles = [
    mpatches.Patch(color=cluster_colors[i], label=f"Cluster {i+1}")
    for i in range(n_clusters)
]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.95, 0.5), frameon=False)

for i_burst in df_bursts.loc[(*idx, slice(None))].index:
    for ax in axs:
        ax.axvspan(
            df_bursts.loc[(*idx, i_burst)].start_extend,
            df_bursts.loc[(*idx, i_burst)].end_extend,
            color=cluster_colors[df_bursts.loc[(*idx, i_burst)].cluster - 1],
            # alpha=0.5,
        )

if plot_raster:
    ax = axs[1]
    ax.plot(st, gid, "|", ms=3, color="k", alpha=0.5)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Neuron ID")
    fig.show()
else:
    fig.show()
