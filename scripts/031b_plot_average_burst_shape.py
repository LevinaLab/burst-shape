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
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures
from src.pie_chart.pie_chart import (
    get_df_cultures_subset,
    plot_df_culture_layout,
    prepare_df_cultures_layout,
)
from src.plot import get_group_colors, prepare_plotting
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

# %%
group_column = None  # defaults to first level of index
group_labels = None
match dataset:
    case "kapucu" | "wagenaar":
        pass
    case "hommersom_test":
        group_column = "clone"
    case "inhibblock":
        group_labels = {
            "bic": "Block.-\ninhib.",
            "control": "Contr.",
        }
    case "mossink":
        df_cultures, df_bursts, _ = make_target_label(
            dataset,
            df_cultures,
            df_bursts=df_bursts,
            special_target=True,
            target_column_name="group-subject",
        )
        df_cultures.reset_index(inplace=True)
        df_bursts.reset_index(inplace=True)
        index_names = ["group-subject", "well_idx"]
        df_cultures.set_index(index_names, inplace=True)
        df_bursts.set_index(index_names + ["i_burst"], inplace=True)
    case _:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
if group_column is None:
    group_column = index_names[0]

# %% plot average burst shape per group
figsize = (7 * cm, 3.5 * cm)
background = [None, "recordings", "bursts"][1]
element = ["lines", "std", "3sem"][0]
fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
sns.despine()
for i, group in enumerate(df_cultures.index.get_level_values(group_column).unique()):
    color = (
        get_group_colors(dataset)[group]
        if get_group_colors(dataset) is not None
        else f"C{i}"
    )
    df_group = df_cultures[df_cultures.index.get_level_values(group_column) == group][
        "avg_burst"
    ]
    label = group_labels[group] if group_labels is not None else group
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
                        np.arange(50),
                        df_group.mean() - np.vstack(df_group).std(axis=0),
                        df_group.mean() + np.vstack(df_group).std(axis=0),
                        color=color,
                        alpha=0.2,
                    )
                case "3sem":
                    ax.fill_between(
                        np.arange(50),
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
                df_bursts.index.get_level_values(group_column) == group
            ]["burst"]
            for value in df_bursts_group:
                ax.plot(value, color=color, alpha=0.01, linewidth=0.5, zorder=1)
        case _:
            pass
    ax.plot(df_group.mean(), color=color, linewidth=1, label=label)
ax.set_xlabel("Time [a.u.]")
ax.set_ylabel("Firing rate [a.u.]")
ax.yaxis.set_label_coords(-0.32, 0.4)
ax.set_xticks([0, 25, 50])
if dataset != "mossink":
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.9, 1.2),
        handlelength=1,
    )
fig.show()
fig.savefig(
    os.path.join(
        get_fig_folder(),
        f"{dataset}_group_average_burst_background_{background}_{element}.svg",
    ),
    transparent=True,
)

# %%
df_cultures = load_df_cultures(burst_extraction_params)
index_names = df_cultures.index.names
df_cultures["avg_burst"] = df_bursts.groupby(index_names).agg(
    avg_burst=pd.NamedAgg(column="burst", aggfunc="mean")
)

# subsample
plot_subset = True
if plot_subset is True:
    df_cultures = get_df_cultures_subset(df_cultures, dataset)

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
    case _:
        figsize = (15, 12)

df_cultures, unique_batch_culture = prepare_df_cultures_layout(df_cultures)
fig, axs = plot_df_culture_layout(
    df_cultures,
    figsize,
    dataset,
    "avg_burst",
    None,
    unique_batch_culture,
    plot_type="avg_burst",
)

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_average_burst_overview.svg"),
    transparent=True,
)

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
                f"{dataset}_average_burst_overview_special_div_{div}.svg",
            ),
            transparent=True,
        )
