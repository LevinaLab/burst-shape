import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.plot import get_cluster_colors, get_group_colors


def prepare_df_cultures_layout(df_cultures):
    """Prepares df_culture to be plotted in a special layout.

    Determines the unique_batch_culture combinations -> columns.
    Assigns an index 'i_culture' identifying the order of the unique combinations.
    This value determines the column position.
    Adds a column 'i_culture' to the df_cultures dataframe.

    :param df_cultures:
    :return: df_cultures, unique_batch_culture
    """
    index_names = df_cultures.index.names
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
    return df_cultures, unique_batch_culture


def plot_df_culture_layout(
    df_cultures, figsize, dataset, column_names, colors, unique_batch_culture
):
    # prepare setup
    index_names = df_cultures.index.names
    ncols = df_cultures["i_culture"].max() + 1
    row_day = (
        df_cultures.index.get_level_values(index_names[-1])
        .unique()
        .sort_values()
        .to_list()
    )
    nrows = len(row_day)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)

    # set all axes to invisible
    # plot white circle to ensure alignment of subplots
    for ax in axs.flatten():
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.pie([1], colors=["white"], startangle=90)

    # plot data
    for index in df_cultures.index:
        i_day = row_day.index(index[len(index_names) - 1])
        i_culture = df_cultures.loc[index, "i_culture"]
        ax = axs[i_day, i_culture]
        ax.axis("on")
        if df_cultures.at[index, "n_bursts"] == 0:
            # empty circle if no bursts are detected
            ax.pie([1], colors=["white"], wedgeprops=dict(width=0, edgecolor="grey"))
        else:
            relative_shares = np.array(df_cultures.loc[index, column_names])
            ax.pie(relative_shares, colors=colors, startangle=90)

    # Add a shared y-axis for days
    _write_row_count_label(axs, dataset, row_day)

    # write batches on the top
    _write_column_group_label(axs, dataset, unique_batch_culture)

    return fig, axs


def _write_row_count_label(axs, dataset, row_day):
    for i_day, day in enumerate(row_day):
        ax = axs[i_day, 0]
        ax.axis("on")
        match dataset:
            case "mossink":
                ax.set_ylabel("")
            case _:
                ax.set_ylabel(f"{day}", rotation=0, fontsize=9)
        ax.yaxis.set_label_coords(-0.5, 0.15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
    return


def _write_column_group_label(axs, dataset, unique_batch_culture):
    for (index), row in unique_batch_culture.iterrows():
        i_culture = row["i_culture"]
        ax = axs[0, i_culture]
        match dataset:
            case "wagenaar":
                (batch, culture) = index
                ax.set_title(
                    f"{batch}-{culture}",
                    rotation=90,
                    fontsize=10,
                    color=get_group_colors(dataset)[batch],
                )
            case "kapucu":
                (culture_type, mea_number, well_id) = index
                ax.set_title(
                    f"{culture_type}",
                    rotation=90,
                    fontsize=10,
                    color=get_group_colors(dataset)[(culture_type, mea_number)],
                )
            case "hommersom":
                (batch, clone) = index
                ax.set_title(f"{batch}-{clone}", rotation=90)
            case "inhibblock":
                if len(index) == 1:
                    ax.set_title(
                        index,
                        fontsize=10,
                        pad=0,
                    )
                else:
                    (drug_label, div) = index
                    drug_str = "BIC" if drug_label == "bic" else "Contr."
                    ax.set_title(
                        f"{drug_str} ({div})",
                        rotation=90,
                        fontsize=10,
                        color=get_group_colors(dataset)[drug_label],
                    )
            case "mossink":
                (group, subject_id) = index
                ax.set_title(
                    f"{group} {subject_id}",
                    rotation=90,
                    fontsize=10,
                    color=get_group_colors(dataset)[f"{group} {subject_id}"],
                )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
    return


def get_df_cultures_subset(df_cultures, dataset):
    df_cultures_subset = df_cultures.copy()
    match dataset:
        case "wagenaar":
            df_cultures_subset = df_cultures_subset[
                df_cultures_subset.index.get_level_values("day") >= 10
            ]
            df_cultures_subset = df_cultures_subset[
                df_cultures_subset.index.get_level_values("day") <= 26
            ]
            list_batch_culture = [
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 1),
                (3, 3),
                (3, 4),
            ]
            df_cultures_subset = df_cultures_subset[
                [
                    (batch, culture) in list_batch_culture
                    for batch, culture in zip(
                        df_cultures_subset.index.get_level_values("batch"),
                        df_cultures_subset.index.get_level_values("culture"),
                    )
                ]
            ]
        case "kapucu":
            df_cultures_subset = df_cultures_subset[
                df_cultures_subset.index.get_level_values("DIV") >= 7
            ]
            df_cultures_subset = df_cultures_subset[
                df_cultures_subset.index.get_level_values("DIV") <= 45
            ]
            list_select = [
                ("Rat", "MEA1", "A1"),
                ("Rat", "MEA1", "A2"),
                ("Rat", "MEA1", "A3"),
                ("Rat", "MEA1", "A4"),
                ("hPSC", "MEA1", "A3"),
                ("hPSC", "MEA1", "A4"),
                ("hPSC", "MEA2", "A3"),
                ("hPSC", "MEA2", "A4"),
            ]
            df_cultures_subset = df_cultures_subset[
                [
                    (culture_type, mea_number, well_id) in list_select
                    for culture_type, mea_number, well_id in zip(
                        df_cultures_subset.index.get_level_values("culture_type"),
                        df_cultures_subset.index.get_level_values("mea_number"),
                        df_cultures_subset.index.get_level_values("well_id"),
                    )
                ]
            ]
        case "mossink":
            df_cultures_subset = df_cultures_subset[
                df_cultures_subset.index.get_level_values("well_idx") <= 12
            ]
        case _:
            warnings.warn(
                f"Taking subset is not implemented for dataset {dataset}. "
                "Returning full dataset instead and continuing."
            )
    return df_cultures_subset
