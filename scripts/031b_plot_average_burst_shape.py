import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures
from src.plot import get_group_colors, prepare_plotting

cm = prepare_plotting()

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
elif "mossink" in burst_extraction_params:
    dataset = "mossink"
else:
    dataset = "wagenaar"
print(f"Detected dataset: {dataset}")

# load data
df_bursts = load_df_bursts(burst_extraction_params)

# index of df_bursts is ('batch', 'culture', 'day', 'i_burst')
# df_bursts.reset_index(inplace=True)

df_cultures = load_df_cultures(burst_extraction_params)

# %% get average burst_shapes
group_column = None
group_labels = None
match dataset:
    case "kapucu":
        index_names = ["culture_type", "mea_number", "well_id", "DIV"]
    case "wagenaar":
        index_names = ["batch", "culture", "day"]
    case "hommersom":
        index_names = ["batch", "clone", "well_idx"]
        group_column = "clone"
    case "inhibblock":
        index_names = ["drug_label", "div", "well_idx"]
        group_labels = {
            "bic": "BIC",
            "control": "Contr.",
        }
    case "mossink":
        index_names = ["group", "subject_id", "well_idx"]
    case _:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
if group_column is None:
    group_column = index_names[0]

df_cultures["avg_burst"] = df_bursts.groupby(index_names).agg(
    avg_burst=pd.NamedAgg(column="burst", aggfunc="mean")
)
# %% plot average burst shape per group
background = [None, "cultures", "burst"][0]
fig, ax = plt.subplots(figsize=(7 * cm, 3.5 * cm), constrained_layout=True)
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
        case "cultures":
            for value in df_group:
                ax.plot(value, color=color, alpha=0.5, linewidth=0.5, zorder=1)
        case "burst":
            df_bursts_group = df_bursts[
                df_bursts.index.get_level_values(group_column) == group
            ]["burst"]
            for value in df_bursts_group:
                ax.plot(value, color=color, alpha=0.01, linewidth=0.5, zorder=1)
        case _:
            pass
    ax.plot(df_group.mean(), color=color, linewidth=2, label=label)
ax.set_xlabel("Time [a.u.]")
ax.set_ylabel("Firing rate [a.u.]")
ax.yaxis.set_label_coords(-0.32, 0.4)
ax.set_xticks([0, 25, 50])
ax.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.9, 1.2),
    handlelength=1,
)
# fig.tight_layout()
fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_group_average_burst.svg"),
    transparent=True,
)
# %% prepare df_cultures indices for plot

# for all unique combinations of batch and culture
unique_batch_culture = df_cultures.reset_index()[index_names[:-1]].drop_duplicates()
# sort by batch and culture
unique_batch_culture.sort_values(index_names[:-1], inplace=True)
# assign an index to each unique combination
unique_batch_culture["i_culture"] = np.arange(len(unique_batch_culture))  # [::-1]
unique_batch_culture.set_index(index_names[:-1], inplace=True)
df_cultures["i_culture"] = pd.Series(
    data=(
        df_cultures.reset_index().apply(
            lambda x: unique_batch_culture.loc[
                tuple(
                    [x[index_label] for index_label in index_names[:-1]]
                ),  # (x["batch"], x["culture"]),
                "i_culture",
            ],
            axis=1,
        )
    ).values,
    index=df_cultures.index,
    dtype=int,
)
# %% plot average burst shape in df_cultures
# position of the pie chart in the grid is determined by the day and i_culture
# colors = sns.color_palette("Set1", n_colors=n_clusters)
match dataset:
    case "kapucu" | "wagenaar":
        figsize = (15, 12)
    case "hommersom":
        figsize = (8, 6)
    case "inhibblock":
        figsize = (3, 5)
    case _:
        figsize = (15, 12)

ncols = df_cultures["i_culture"].max() + 1
row_day = (
    df_cultures.index.get_level_values(index_names[-1]).unique().sort_values().to_list()
)
# consider filling in missing days
# nrows = df_cultures.index.get_level_values("day").max() + 1
nrows = len(row_day)
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True
)
# sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)
# sns.despine(fig=fig, top=True, right=True, left=False, bottom=False)

# set all axes to invisible
for ax in axs.flatten():
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines["left"].set_visible(False)
    # ax.spines["top"].set_visible(False)
for index in df_cultures.index:
    i_day = row_day.index(index[len(index_names) - 1])
    i_culture = df_cultures.loc[index, "i_culture"]
    ax = axs[i_day, i_culture]
    ax.axis("on")
    # ax.set_title(f"Day {i_day} - Culture {i_culture}")
    if df_cultures.at[index, "n_bursts"] == 0:
        pass
    else:
        color = (
            get_group_colors(dataset)[index[0]]
            if get_group_colors(dataset) is not None
            else "black"
        )
        ax.plot(df_cultures.at[index, "avg_burst"], color=color)

# Add a shared y-axis for days
for i_day, day in enumerate(row_day):
    ax = axs[i_day, 0]
    ax.axis("on")
    ax.set_ylabel(f"{day}", rotation=0)
    # ax.xaxis.set_label_position("bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines["left"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # ax.pie([1], colors=["white"], startangle=90)

# write batches on the top
# batch_label_pos = np.linspace(0.04, 0.97, nrows, endpoint=True)[::-1]
for (index), row in unique_batch_culture.iterrows():
    i_culture = row["i_culture"]
    ax = axs[0, i_culture]
    match dataset:
        case "wagenaar":
            (batch, culture) = index
            ax.set_title(f"{batch}-{culture}")
        case "kapucu":
            (culture_type, mea_number, well_id) = index
            ax.set_title(f"{culture_type}-{mea_number}-{well_id}", rotation=90)
        case "hommersom":
            (batch, clone) = index
            ax.set_title(f"{batch}-{clone}", rotation=90)
        case "inhibblock":
            (drug_label, div) = index
            ax.set_title(f"{drug_label}-{div}", rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines["left"].set_visible(False)
    # ax.spines["top"].set_visible(False)

axs[0, 0].set_ylim(0, None)

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.show()
"""
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_pie_chart.svg"), transparent=True
)
"""
