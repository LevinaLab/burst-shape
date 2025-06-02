import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures
from src.persistence.spike_times import get_inhibblock_spike_times
from src.plot import prepare_plotting

cm = prepare_plotting()

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
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
df_cultures = load_df_cultures(burst_extraction_params)

# %% inhibblock examples
if dataset == "inhibblock":
    examples_indices = [
        ("control", 17, 12, 124),
        # ("bic", 17, 2, 115),
        ("control", 17, 4, 77),
        # ("bic", 17, 7, 98),
    ]

    fig, axs = plt.subplots(
        nrows=3,
        constrained_layout=True,
        figsize=(6 * cm, 5 * cm),
    )
    sns.despine()
    # sns.despine()
    for i, index in enumerate(examples_indices):
        start, end = df_bursts.at[index, "start_orig"], df_bursts.at[index, "end_orig"]
        st, gid = get_inhibblock_spike_times(df_cultures, index[:-1])
        selection = (st >= start) & (st <= end)
        st, gid = st[selection], gid[selection]
        # bins = np.linspace(start, end, num=301, endpoint=True)
        bins = np.arange(start, end, 5)
        bins_mid = (bins[1:] + bins[:-1]) / 2
        firing_rate = np.histogram(st, bins=bins)[0] / (bins[1] - bins[0]) * 1000

        bins_norm = np.linspace(start, end, num=51, endpoint=True)
        bins_norm_mid = np.arange(50)  # (bins[1:] + bins[:-1]) / 2
        firing_rate_norm = (
            np.histogram(st, bins=bins_norm)[0] / (bins_norm[1] - bins_norm[0]) * 1000
        )

        color = f"C{i}"
        linestyle = "-"  #  if i == 0 else "--"

        ax = axs[0]
        # ax.scatter(st - start, gid, marker="|", s=2, color="k", alpha=0.3)
        ax.plot(
            bins_mid - start,
            firing_rate,
            linewidth=0.5,
            color=color,
            linestyle=linestyle,
        )

        ax = axs[1]
        ax.plot(
            bins_norm_mid,
            firing_rate_norm,
            linewidth=1,
            color=color,
            linestyle=linestyle,
        )

        ax = axs[2]
        ax.plot(
            df_bursts.at[index, "burst"], linewidth=1, color=color, linestyle=linestyle
        )

    ax = axs[0]
    ax.set_xticks([0, 3000])
    ax.set_xlabel("Time [ms]")
    ax.xaxis.set_label_coords(0.5, -0.35)
    ax.set_ylabel("\n[Hz]")

    ax = axs[1]
    ax.set_xticks([0, 50])
    ax.set_xlabel("Time [a.u.]")
    ax.xaxis.set_label_coords(0.5, -0.35)
    ax.set_ylabel("[Hz]")

    ax = axs[2]
    ax.set_xticks([0, 50])
    ax.set_xlabel("Time [a.u.]")
    ax.xaxis.set_label_coords(0.5, -0.35)
    ax.set_ylabel("[a.u.]")

    fig.text(0.01, 0.5, "Firing rate", rotation=90, verticalalignment="center")

    fig.show()
    fig.savefig(
        os.path.join(get_fig_folder(), f"demo_burst_preprocessing.svg"),
        transparent=True,
    )
