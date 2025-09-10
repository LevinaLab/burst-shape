"""
Plot average burst shape of the recordings (groups and individual recordings).

Group average plot has several options for what to plot in the background:
- data can be empty, from all recording averages or even from all individual bursts
- from this data, one can plot the individual lines, the std, or 3*sem

The overview of recording averages is plotted in the usual special layout.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats as stats
import seaborn as sns
from matplotlib import ticker

from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures
from src.pie_chart.pie_chart import (
    get_df_cultures_subset,
    plot_df_culture_layout,
    prepare_df_cultures_layout,
)
from src.plot import (
    get_group_colors,
    get_group_labels,
    label_sig_diff,
    prepare_plotting,
    savefig,
)
from src.prediction.define_target import make_target_label
from src.settings import get_dataset_from_burst_extraction_params

cm = prepare_plotting()

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_mossink_KS"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Detected dataset: {dataset}")

# load data
df_bursts = load_df_bursts(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)

# get average burst_shapes
index_names = df_cultures.index.names
df_cultures["avg_burst"] = df_bursts.groupby(index_names).agg(
    avg_burst=pd.NamedAgg(column="burst", aggfunc="mean")
)
df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
df_cultures, target_label = make_target_label(
    dataset, df_cultures, special_target=False
)

# %% plot average burst shape per group
background = [None, "recordings", "bursts"][1]
element = ["lines", "std", "3sem"][2]
x_values = np.arange(50) + 0.5  # 50 bins for the burst shape
fig, ax = plt.subplots(figsize=(8 * cm, 3.5 * cm), constrained_layout=True)
ax.set_position([0.25, 0.35, 0.4, 0.6])
sns.despine()
for i, group in enumerate(df_cultures["target_label"].unique()):
    color = (
        get_group_colors(dataset)[group]
        if get_group_colors(dataset) is not None
        else f"C{i}"
    )
    df_group = df_cultures[df_cultures["target_label"] == group]["avg_burst"]
    match background:
        case "recordings":
            match element:
                case "lines":
                    ax.plot(
                        # x_values,
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
        case "bursts":
            df_bursts_group = df_bursts[
                df_bursts.index.get_level_values("target_label") == group
            ]["burst"]
            for value in df_bursts_group:
                ax.plot(value, color=color, alpha=0.01, linewidth=0.5, zorder=1)
        case _:
            pass
    ax.plot(
        x_values,
        df_group.mean(),
        color=color,
        linewidth=1,
        label=get_group_labels(dataset, group),
    )
ax.set_xlabel("Time [a.u.]")
ax.set_ylabel("Firing rate [a.u.]")
ax.yaxis.set_label_coords(-0.32, 0.4)
ax.set_xticks([0, 25, 40, 50])
ax.set_xticklabels([0, 0.5, 0.8, 1])
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
if dataset != "wagenaar":
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.9, 1.2),
        handlelength=1,
    )
# plot vertical line indicating peaks, decay measurement location
match dataset:
    case "inhibblock":
        kwargs = {"linestyle": "--", "linewidth": 1}
        ax.axvline(x=39.5, ymax=0.5, color="k", **kwargs)
        ax.axvline(x=0.5 + 2.800000, color=get_group_colors(dataset)["bic"], **kwargs)
        ax.axvline(
            x=0.5 + 10.428571, color=get_group_colors(dataset)["control"], **kwargs
        )
    case "hommersom_binary":
        kwargs = {"linestyle": "--", "linewidth": 1}
        ax.axvline(x=39.5, ymax=0.5, color="k", **kwargs)
        ax.axvline(
            x=0.5 + 12.894737, color=get_group_colors(dataset)["CACNA1A"], **kwargs
        )
        ax.axvline(
            x=0.5 + 8.666667, color=get_group_colors(dataset)["Control"], **kwargs
        )
    case "mossink_KS":
        kwargs = {"linestyle": "--", "linewidth": 1}
        ax.axvline(x=39.5, ymax=0.5, color="k", **kwargs)
        ax.axvline(
            x=0.5 + 13.845614, color=get_group_colors(dataset)["Control"], **kwargs
        )
        ax.axvline(x=0.5 + 7.310345, color=get_group_colors(dataset)["KS"], **kwargs)
    case _:
        pass

fig.show()
for file_type in ["pdf", "svg"]:
    fig.savefig(
        os.path.join(
            get_fig_folder(),
            f"{dataset}_group_average_burst_background_{background}_{element}.{file_type}",
        ),
        transparent=True,
    )


# %% run statistical tests on average burst shapes
def _run_group_comparison(df_cultures, compare_column, plot_significance=True):
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
    fig, ax = plt.subplots(figsize=(4 * cm, 5 * cm), constrained_layout=True)
    ax.set_position([0.5, 0.2, 0.4, 0.7])
    sns.despine()
    sns.boxplot(
        data=df_cultures.reset_index(),
        x="target_label",
        hue="target_label",
        y=compare_column,
        palette=get_group_colors(dataset),
        flierprops=dict(
            marker="o", markerfacecolor="black", markersize=4, linestyle="none"
        ),
    )
    ax.set_xlabel("")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [get_group_labels(dataset, lbl.get_text()) for lbl in ax.get_xticklabels()]
    )
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_y(label.get_position()[1] - 0.1 * (i % 2))
    if plot_significance is True:
        label_sig_diff(
            ax,
            [0, 1],
            df_cultures[compare_column].max(),
            p_val,
            0.1
            * (df_cultures[compare_column].max() - df_cultures[compare_column].min()),
            0.05
            * (df_cultures[compare_column].max() - df_cultures[compare_column].min()),
            "k",
            10,
            1,
        )
    fig.show()

    print(return_values)
    return *return_values, fig, ax


print("Run statistical tests on average burst shapes.")
match dataset:
    case "hommersom_binary" | "inhibblock" | "mossink_KS":
        df_cultures["argmax_bin"] = df_cultures["avg_burst"].apply(np.argmax)
        df_cultures["rel_peak"] = (df_cultures["argmax_bin"] + 0.5) / 50
        df_cultures["value_45th_bin"] = df_cultures["avg_burst"].apply(lambda x: x[44])
        df_cultures["value_40th_bin"] = df_cultures["avg_burst"].apply(lambda x: x[39])

        for column in ["argmax_bin", "rel_peak", "value_45th_bin", "value_40th_bin"]:
            print(f"\n{column}:")
            mean_std = df_cultures.groupby("target_label", observed=True)[column].agg(
                ["mean", "std"]
            )
            print(mean_std)
            _, _, fig, ax = _run_group_comparison(df_cultures, column)
            fig.savefig(
                os.path.join(
                    get_fig_folder(), f"{dataset}_group_average_stats_{column}.svg"
                ),
                transparent=True,
            )
    case "something else":
        df_cultures["argmax_bin"] = df_cultures["avg_burst"].apply(np.argmax)

        for column in ["argmax_bin"]:
            print(f"\n{column}:")
            mean_std = df_cultures.groupby("target_label", observed=True)[column].agg(
                ["mean", "std"]
            )
            print(mean_std)
            _, _, fig, ax = _run_group_comparison(df_cultures, column)
            fig.savefig(
                os.path.join(
                    get_fig_folder(), f"{dataset}_group_average_stats_{column}.svg"
                ),
                transparent=True,
            )

    case _:
        print("Running no statistical tests on average burst shape.")


# %% plot overview of average burst per recording (plotted in subplots in 2D grid)
df_cultures_overview = load_df_cultures(burst_extraction_params)
index_names = df_cultures_overview.index.names
df_cultures_overview["avg_burst"] = df_bursts.groupby(index_names).agg(
    avg_burst=pd.NamedAgg(column="burst", aggfunc="mean")
)

# subsample
plot_subset = True
if plot_subset is True:
    df_cultures_overview = get_df_cultures_subset(df_cultures_overview, dataset)

match dataset:
    case "kapucu":
        if plot_subset is True:
            figsize = (4.5 * cm, 7 * cm)
        else:
            figsize = (12 * cm, 10 * cm)
    case "wagenaar":
        if plot_subset is True:
            figsize = (5 * cm, 8 * cm)
        else:
            figsize = (12 * cm, 10 * cm)
    case "hommersom_test":
        figsize = (8, 6)
    case "inhibblock":
        figsize = (5 * cm, 9 * cm)
    case "mossink":
        if plot_subset is True:
            figsize = (10 * cm, 9 * cm)
        else:
            figsize = (10 * cm, 30 * cm)
    case "hommersom":
        figsize = (7 * cm, 20 * cm)
    case _:
        figsize = (15, 12)

df_cultures_overview, unique_batch_culture = prepare_df_cultures_layout(
    df_cultures_overview
)
fig, axs = plot_df_culture_layout(
    df_cultures_overview,
    figsize,
    dataset,
    "avg_burst",
    None,
    unique_batch_culture,
    plot_type="avg_burst",
)

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
sns.despine(fig=fig, bottom=True, left=True)
fig.show()
savefig(fig, f"{dataset}_average_burst_overview", file_format=["pdf", "svg"])

# %% special plate layout for inhibblock data
if dataset == "inhibblock":
    for div in [17, 18]:
        df_cultures_special = df_cultures.reset_index()
        df_cultures_special = df_cultures_special[df_cultures_special["div"] == div]
        df_cultures_special["layout_column"] = df_cultures_special["well"].str[0]
        df_cultures_special["layout_row"] = df_cultures_special["well"].str[1]
        df_cultures_special.set_index(["layout_row", "layout_column"], inplace=True)

        df_cultures_special, unique_batch_culture_special = prepare_df_cultures_layout(
            df_cultures_special
        )
        fig, axs = plot_df_culture_layout(
            df_cultures_special,
            (4 * cm, 3 * cm),
            dataset,
            "avg_burst",
            None,
            unique_batch_culture_special,
            plot_type="avg_burst",
        )

        fig.tight_layout()
        # fig.subplots_adjust(wspace=-0.15, hspace=-0.15)

        fig.subplots_adjust(wspace=-0.15, hspace=-0.15)

        fig.show()
        fig.savefig(
            os.path.join(
                get_fig_folder(),
                f"{dataset}_average_burst_overview_special_div_{div}.pdf",
            ),
            transparent=True,
        )
