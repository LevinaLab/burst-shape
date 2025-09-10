"""
Plot traditional average burst shape for comparison.

This is for demonstration purposes only.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from tqdm import tqdm

from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures
from src.persistence.spike_times import get_spike_times_in_milliseconds
from src.plot import get_group_colors, get_group_labels, prepare_plotting
from src.prediction.define_target import make_target_label
from src.settings import get_dataset_from_burst_extraction_params

cm = prepare_plotting()

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_KS"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Detected dataset: {dataset}")

# load data
df_cultures = load_df_cultures(burst_extraction_params)

# make target label
df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
df_cultures, target_label = make_target_label(
    dataset, df_cultures, special_target=False
)

# determine longest burst duration
df_bursts = load_df_bursts(burst_extraction_params)
max_duration = df_bursts["time_orig"].max()
del df_bursts

# %% get average burst shape
start_offset = -200  # ms
bin_size = 10  # ms
bins = np.arange(start_offset, max_duration + bin_size, bin_size)

df_cultures["burst_shape_absolute"] = pd.Series(dtype=object)
df_cultures["burst_shape_absolute_mean"] = pd.Series(dtype=object)
for idx in tqdm(df_cultures.index):
    spike_times, _ = get_spike_times_in_milliseconds(df_cultures, idx, dataset)
    burst_shape_absolute = np.zeros((df_cultures.at[idx, "n_bursts"], len(bins) - 1))
    for i, (start, end) in enumerate(df_cultures.at[idx, "burst_start_end"]):
        burst_spikes = spike_times[
            (spike_times >= start + start_offset) & (spike_times < end)
        ]
        burst_spikes -= start
        assert burst_spikes.min() >= start_offset
        assert burst_spikes.max() <= max_duration
        burst_shape_absolute[i] = np.histogram(burst_spikes, bins=bins)[0]
    burst_shape_absolute /= bin_size / 1000  # convert to Hz
    df_cultures.at[idx, "burst_shape_absolute"] = burst_shape_absolute
    df_cultures.at[idx, "burst_shape_absolute_mean"] = burst_shape_absolute.mean(axis=0)
del burst_shape_absolute, spike_times

# %% plot average burst shape
column_burst_shapes = "burst_shape_absolute_mean"
x_values = (bins[:-1] + bins[1:]) / 2  # center of bins

figsize = (7 * cm, 3.5 * cm)
background = [None, "recordings"][1]
element = ["lines", "std", "3sem"][2]
fig, ax = plt.subplots(figsize=(8 * cm, 3.5 * cm), constrained_layout=True)
ax.set_position([0.25, 0.35, 0.4, 0.6])
sns.despine()
for i, group in enumerate(df_cultures["target_label"].unique()):
    color = (
        get_group_colors(dataset)[group]
        if get_group_colors(dataset) is not None
        else f"C{i}"
    )
    df_group = df_cultures[df_cultures["target_label"] == group][column_burst_shapes]
    match background:
        case "recordings":
            match element:
                case "lines":
                    ax.plot(
                        np.vstack(df_group).T,
                        color=color,
                        alpha=0.5,
                        linewidth=0.5,
                        zorder=1,
                    )
                case "std":
                    ax.fill_between(
                        x_values,
                        df_group.mean() - np.vstack(df_group).std(axis=0),
                        df_group.mean() + np.vstack(df_group).std(axis=0),
                        color=color,
                        alpha=0.2,
                    )
                case "3sem":
                    ax.fill_between(
                        x_values,
                        df_group.mean()
                        - 3 * scipy.stats.sem(np.vstack(df_group), axis=0),
                        df_group.mean()
                        + 3 * scipy.stats.sem(np.vstack(df_group), axis=0),
                        color=color,
                        alpha=0.5,
                        edgecolor=None,
                    )
        case _:
            pass
    ax.plot(
        x_values,
        df_group.mean(),
        color=color,
        linewidth=1,
        label=get_group_labels(dataset, group),
    )
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Firing rate [Hz]")
ax.yaxis.set_label_coords(-0.4, 0.4)
# ax.yaxis.set_label_coords(-0.32, 0.4)
ax.set_xlim(start_offset, max_duration)
# ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
# ax.set_xticks([0, 25, 50])
if dataset != "wagenaar":
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.9, 1.2),
        handlelength=1,
    )
fig.show()
for file_type in ["svg", "pdf"]:
    fig.savefig(
        os.path.join(
            get_fig_folder(),
            f"{dataset}_traditional_group_average_burst_background_{background}_{element}.{file_type}",
        ),
        transparent=True,
    )

# %% run statistical tests on average burst shapes
import scipy.stats as stats


def _run_group_comparison(df_cultures, compare_column):
    grouped = df_cultures.groupby("target_label", observed=True)
    groups = list(grouped.groups.keys())

    if len(groups) == 2:
        group1 = grouped.get_group(groups[0])[compare_column]
        group2 = grouped.get_group(groups[1])[compare_column]

        # Independent two-sample t-test
        t_stat, p_val = stats.ttest_ind(
            group1, group2, equal_var=False
        )  # Welch's t-test
        print("Run ttest")
        return_values = (t_stat, p_val)
    else:
        # For more than 2 groups, use ANOVA or Kruskal-Wallis
        values = [grouped.get_group(g)[compare_column] for g in groups]
        f_stat, p_val = stats.f_oneway(*values)
        print("Run f_oneway")
        return_values = (f_stat, p_val)
    fig, ax = plt.subplots(figsize=(4 * cm, 6 * cm), constrained_layout=True)
    sns.despine()
    sns.boxplot(
        data=df_cultures.reset_index(),
        x="target_label",
        hue="target_label",
        y=compare_column,
        palette=get_group_colors(dataset),
    )
    fig.show()

    print(return_values)
    return *return_values, fig, ax


print("Run statistical tests on average burst shapes.")
match dataset:
    case "inhibblock" | "hommersom_binary" | "mossink_KS":
        df_cultures["peak_time"] = (
            df_cultures[column_burst_shapes].apply(np.argmax) * bin_size
        )

        for column in ["peak_time"]:
            print(f"\n{column}:")
            mean_std = df_cultures.groupby("target_label", observed=True)[column].agg(
                ["mean", "std"]
            )
            print(mean_std)
            _, _, fig, ax = _run_group_comparison(df_cultures, column)
            fig.savefig(
                os.path.join(
                    get_fig_folder(),
                    f"{dataset}_traditional_group_average_stats_{column}.svg",
                ),
                transparent=True,
            )

    case _:
        print("Running no statistical tests on average burst shape.")
