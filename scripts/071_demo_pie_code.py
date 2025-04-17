import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence import load_clustering_labels, load_df_bursts, load_df_cultures
from src.persistence.spike_times import get_inhibblock_spike_times
from src.plot import get_cluster_colors, prepare_plotting

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
clustering = load_clustering_labels(
    clustering_params,
    burst_extraction_params,
    labels_params,
    None,
    i_split=None,
)
df_bursts[f"cluster_{n_clusters}"] = [
    f"Cluster {label}" for label in clustering.labels_[n_clusters]
]
df_cultures = load_df_cultures(burst_extraction_params)

# %% inhibblock example
if dataset == "inhibblock":
    # example_index = ("control", 18, 12)
    example_index = ("control", 18, 4)
    print(df_cultures.at[example_index, "well"])
    df_plot = df_bursts.loc[example_index]

    st, gid = get_inhibblock_spike_times(df_cultures, example_index)
    st /= 1000
    bin_size = 0.100
    times_all = np.arange(0, st.max() + bin_size, bin_size)
    firing_rate = np.histogram(st, bins=times_all)[0] / (bin_size)
    times_all = 0.5 * (times_all[1:] + times_all[:-1])
    # Collect all burst start and end times
    burst_starts = []
    burst_ends = []
    burst_colors = []
    burst_index = []

    color = f"cluster_{n_clusters}"
    color_discrete_sequence = get_cluster_colors(n_clusters)
    for i_burst in df_plot.index:
        burst_starts.append(df_plot.loc[i_burst].start_orig / 1000)
        burst_ends.append(df_plot.loc[i_burst].end_orig / 1000)
        burst_colors.append(
            color_discrete_sequence[int(df_plot.loc[i_burst][color].split(" ")[1])]
        )
        burst_index.append(i_burst)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10 * cm, 3 * cm))
    sns.despine()
    for start, end, color, index in zip(
        burst_starts, burst_ends, burst_colors, burst_index
    ):
        ax.axvspan(start, end, color=color, alpha=0.6, zorder=-1)

    ax.plot(times_all, firing_rate, color="k", label="firing rate", linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Firing Rate [Hz]")
    ax.yaxis.set_label_coords(-0.15, 0.25)

    fig.show()
    fig.savefig(
        os.path.join(get_fig_folder(), f"demo_pie_code.svg"),
        transparent=True,
    )

    fig, ax = plt.subplots(constrained_layout=True, figsize=(3 * cm, 3 * cm))
    fractions = [
        (df_plot[f"cluster_{n_clusters}"] == f"Cluster {i_cluster}").sum()
        / df_cultures.at[example_index, "n_bursts"]
        for i_cluster in range(n_clusters)
    ]

    ax.pie(fractions, colors=color_discrete_sequence, startangle=90)
    fig.show()
    fig.savefig(
        os.path.join(get_fig_folder(), f"demo_pie_code-pie.svg"),
        transparent=True,
    )
