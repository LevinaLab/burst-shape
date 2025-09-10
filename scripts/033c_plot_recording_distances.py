import os
import warnings
from collections.abc import Iterable
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from src.folders import get_fig_folder
from src.persistence import (
    load_burst_matrix,
    load_clustering_labels,
    load_df_bursts,
    load_df_cultures,
    load_spectral_embedding,
)
from src.plot import label_sig_diff, prepare_plotting, savefig
from src.prediction.knn_clustering import get_recording_mask
from src.settings import (
    get_chosen_spectral_clustering_params,
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)

cm = prepare_plotting()

metric = (
    # "cosine-distance"
    "Wasserstein"
    # "Wasserstein-individual-bursts"
    # "Embedding"
)

# parameters which clustering to plot
burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Detected dataset: {dataset}")

# define df_distance_metric
match metric:
    case "cosine-distance":
        # which clustering to plot
        clustering_params, n_clusters = get_chosen_spectral_clustering_params(dataset)

        col_cluster = f"cluster_{n_clusters}"
        labels_params = "labels"
        cv_params = "cv"  # if cv_split is not None, chooses the cross-validation split
        cv_split = None  # set to None for plotting the whole clustering, set to int for specific split

        # load bursts
        df_bursts = load_df_bursts(burst_extraction_params)
        index_names = df_bursts.index.names[:-1]

        #  get clusters from linkage
        clustering = load_clustering_labels(
            clustering_params,
            burst_extraction_params,
            labels_params,
            cv_params,
            cv_split,
        )
        df_bursts["cluster"] = clustering.labels_[n_clusters] + 1

        #  build new dataframe df_cultures
        # with index ('batch', 'culture', 'day') and columns ('n_bursts', 'cluster_abs', 'cluster_rel')
        print("Building df_cultures...")
        df_bursts_reset = df_bursts.reset_index(
            drop=False
        )  # reset index to access columns in groupby()
        df_cultures = df_bursts_reset.groupby(index_names).agg(
            n_bursts=pd.NamedAgg(column="i_burst", aggfunc="count")
        )

        # dd cluster information
        for i_cluster in range(1, n_clusters + 1):
            col_cluster = "cluster"
            df_cultures[f"cluster_abs_{i_cluster}"] = df_bursts.groupby(index_names)[
                col_cluster
            ].agg(lambda x: np.sum(x == i_cluster))
            df_cultures[f"cluster_rel_{i_cluster}"] = (
                df_cultures[f"cluster_abs_{i_cluster}"] / df_cultures["n_bursts"]
            )

        columns = [f"cluster_rel_{i_cluster}" for i_cluster in range(1, n_clusters + 1)]

        distance_matrix = squareform(
            pdist(df_cultures[columns].values, metric="cosine")
        )
        df_distance_matrix = pd.DataFrame(
            distance_matrix,
            index=df_cultures.index,
            columns=df_cultures.index,
        )
    case "Wasserstein":
        df_cultures = load_df_cultures(burst_extraction_params)
        df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
        df_bursts = load_df_bursts(burst_extraction_params, cv_params=None)
        burst_matrix = load_burst_matrix(burst_extraction_params)

        #
        df_cultures["average_burst"] = pd.Series(dtype="object")
        for index in df_cultures.index:
            mask_recording = get_recording_mask(df_bursts, index)
            df_cultures.at[index, "average_burst"] = burst_matrix[mask_recording].mean(
                axis=0
            )
        burst_matrix_cultures = np.vstack(df_cultures["average_burst"])

        #
        def _wasserstein_distance(a, b):
            a_cumsum = np.cumsum(a)
            b_cumsum = np.cumsum(b)
            a_cumsum /= a_cumsum[-1]
            b_cumsum /= b_cumsum[-1]
            return np.sum(np.abs(a_cumsum - b_cumsum))

        distance_matrix = squareform(
            pdist(burst_matrix_cultures, metric=_wasserstein_distance)
        )
        df_distance_matrix = pd.DataFrame(
            distance_matrix,
            index=df_cultures.index,
            columns=df_cultures.index,
        )
    case "Wasserstein-individual-bursts":
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")
    case "Embedding":
        df_cultures = load_df_cultures(burst_extraction_params)
        df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
        df_bursts = load_df_bursts(burst_extraction_params, cv_params=None)

        spectral_clustering_params = get_chosen_spectral_embedding_params(dataset)
        spectral_embedding = load_spectral_embedding(
            burst_extraction_params, spectral_clustering_params
        )
        n_spectral_dims = spectral_embedding.shape[1]
        for i_dim in range(n_spectral_dims):
            name_dim = f"shape_{i_dim + 1}"
            df_bursts[name_dim] = spectral_embedding[:, i_dim]
            df_cultures[name_dim] = pd.Series(dtype=float)
            for index in df_cultures.index:
                mask_recording = get_recording_mask(df_bursts, index)
                df_cultures.at[index, name_dim] = (
                    df_bursts[name_dim].values[mask_recording].mean()
                )

        columns = [f"shape_{i_dim + 1}" for i_dim in range(n_spectral_dims)]

        distance_matrix = squareform(
            pdist(df_cultures[columns].values, metric="euclidean")
        )
        df_distance_matrix = pd.DataFrame(
            distance_matrix,
            index=df_cultures.index,
            columns=df_cultures.index,
        )
    case _:
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")


# %%
def _select_distances(
    df_distance_matrix: pd.DataFrame,
    column_separate_combo: List[str] = None,
    column_comparison_combo: List[str] = None,
    column_day_to_day: str = None,
    special_function=None,
    special_columns: List[str] = None,
):
    """
    Selects the distances from the full distance matrix according to the constraints.

    This function is for example used to compute distances between and within groups.
    Each given column has to be in the index.

    :param df_distance_matrix: distances in squareform.
        Index and (!) column must be indexed with df_cultures.index.
    :param column_separate_combo: columns for which only matching combinations
        should be selected from the distance matrix (within-group distances).
    :param column_comparison_combo: columns for which only disagreeing combinations
        should be selected from the distance matrix (between-group distances).
    :param column_day_to_day: column for which consecutive samples' distances should be considered.
        This column must have an order.
    :return: df_selected_distances: dataframe with subsampled distances.
        It has the same index as df_distance_matrix.
        It has a single column "distances" that contains all the selected distances as a list.
    """
    index = df_distance_matrix.index
    level_names = df_distance_matrix.index.names

    # Prepare meshgrid of all combinations (i, j) where i ≠ j
    row_idx, col_idx = zip(*[(i, j) for i in index for j in index if i != j])
    i_values = pd.DataFrame(
        row_idx, columns=pd.MultiIndex.from_product([["i"], level_names])
    )
    j_values = pd.DataFrame(
        col_idx, columns=pd.MultiIndex.from_product([["j"], level_names])
    )
    row_pos = index.get_indexer(row_idx)
    col_pos = index.get_indexer(col_idx)
    distances = df_distance_matrix.values[row_pos, col_pos]

    # create df_pairs
    df_pairs = pd.concat([i_values, j_values], axis=1)
    df_pairs["distance"] = distances

    # Apply within-group (separate) filter: i[col] == j[col]
    for col in column_separate_combo or []:
        df_pairs[("filter", col)] = df_pairs[("i", col)] == df_pairs[("j", col)]

    # Apply between-group (comparison) filter: i[col] != j[col]
    for col in column_comparison_combo or []:
        df_pairs[("filter", col)] = df_pairs[("i", col)] != df_pairs[("j", col)]

    # Apply day-to-day filter (consecutive timepoints)
    if column_day_to_day:
        unique_days = df_pairs[("i", column_day_to_day)].unique()
        unique_days.sort()
        day_map = {day: i for i, day in enumerate(unique_days)}
        day_diff = (
            df_pairs[("i", column_day_to_day)].map(day_map)
            - df_pairs[("j", column_day_to_day)].map(day_map)
        ).abs()
        df_pairs[("filter", column_day_to_day)] = day_diff == 1

    if special_function:
        df_pairs[("filter", "special_function")] = special_function(
            *[df_pairs[("i", special_column)] for special_column in special_columns],
            *[df_pairs[("j", special_column)] for special_column in special_columns],
        )

    # Combine all filters (AND logic)
    filter_cols = [col for col in df_pairs.columns if col[0] == "filter"]
    if filter_cols:
        combined_mask = df_pairs[filter_cols].all(axis=1)
        df_pairs = df_pairs[combined_mask]

    # Group distances by i
    df_pairs[("i", "tuple")] = list(
        zip(*[df_pairs[("i", name)] for name in level_names])
    )
    df_selected = df_pairs.groupby(("i", "tuple"))["distance"].apply(list)

    # Ensure all original rows are present
    # df_selected_distances = df_selected.reindex(df_distance_matrix.index, fill_value=[])
    df_selected_distances = df_selected.reindex(df_distance_matrix.index)
    df_selected_distances = df_selected_distances.apply(
        lambda x: [] if not isinstance(x, list) else x
    )
    df_selected_distances = df_selected_distances.to_frame(name="distances")

    # Compute average
    df_selected_distances["distances_avg"] = df_selected_distances["distances"].apply(
        np.mean
    )

    return df_selected_distances.dropna()


# %%
def _plot_distance_distributions(
    similarities: List[pd.DataFrame],
    figsize,
    xtick_labels,
    xtick_labels_args=None,
    y_min=0,
    plot_statistical_test=False,
    plot_stats_lims=None,
):
    if xtick_labels_args is None:
        xtick_labels_args = {}
    data = [
        np.concatenate(df_similarity["distances"].values)
        for df_similarity in similarities
    ]
    data_avg = [df_similarity["distances_avg"].values for df_similarity in similarities]
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    sns.despine()

    """sns.violinplot(
        data=data,
        ax=ax,
        inner=None,  # Remove internal boxplot
        facecolor=(1, 1, 1, 0),  # Transparent fill
        edgecolor="grey",
        linewidth=1.5,
    )"""
    # Overlay boxplot in red
    """sns.boxplot(
        data=data,
        ax=ax,
        showcaps=False,
        width=0.2,
        color="grey",
        boxprops={"facecolor": "grey", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        medianprops={"color": "red"},
        flierprops={"markerfacecolor": "grey", "markeredgecolor": "grey"},
        fliersize=0,
    )"""
    sns.pointplot(
        data=data_avg,
        ax=ax,
        errorbar="ci",
        linestyle="",
        color="k",
        markers="D",  # Diamond marker
        capsize=0.2,  # Adds caps to error bars
        errwidth=1.5,  # (Optional) make error bars thicker
        markersize=2,
    )

    # Overlay dots for individual data points
    # sns.stripplot(data=data, ax=ax, color="black", size=0.3, jitter=True)
    # sns.boxplot(data=data, ax=ax, color="grey", size=0.3)

    # Labeling
    ax.set_xticks(list(range(len(data))))
    ax.set_xticklabels(
        xtick_labels,
        **xtick_labels_args,
    )
    ax.get_xticklabels()[0].set_transform(
        ax.get_xticklabels()[0].get_transform()
        + matplotlib.transforms.Affine2D().translate(-7, 0)
    )
    ax.get_xticklabels()[2].set_transform(
        ax.get_xticklabels()[2].get_transform()
        + matplotlib.transforms.Affine2D().translate(4, 0)
    )
    ax.set_xlim(-0.5, len(xtick_labels) - 0.5)

    for i, distribution1 in enumerate(data_avg):
        for j, distribution2 in enumerate(data_avg):
            if i >= j:
                continue
            print(f"{i} vs {j}")
            test_mannwhitneyu = scipy.stats.mannwhitneyu(
                distribution1,
                distribution2,
            )
            test_t_ind = scipy.stats.ttest_ind(
                distribution1,
                distribution2,
            )
            print(test_t_ind)
            if plot_statistical_test:
                if isinstance(test_t_ind, Iterable):
                    if (i, j) in plot_statistical_test:
                        pass
                    else:
                        continue
                label_sig_diff(
                    ax=ax,
                    inds=(i, j),
                    data=plot_stats_lims
                    if plot_stats_lims is not None
                    else ax.get_ylim(),
                    text_sig=test_t_ind.pvalue,
                )

    _set_yaxis(ax, y_min=y_min)
    return fig, ax


def _set_yaxis(ax, y_min=0):
    match metric:
        case "cosine-distance":
            ax.set_ylabel("Cosine\ndistance")
            ax.set_yticks([0, 1])
        case "Wasserstein":
            ax.set_ylabel("Wasserstein\ndistance")
            ax.set_ylim(y_min, None)
        case "Wasserstein-individual-bursts":
            warnings.warn(f"Undefined function _set_yaxis() for metric {metric}.")
        case "Embedding":
            ax.set_ylabel("Embedding\ndistance")
            ax.set_ylim(y_min, None)
        case _:
            ax.set_ylabel("Distance")
            warnings.warn(f"Undefined function _set_yaxis() for metric {metric}.")


# %%
match dataset:
    case "inhibblock":
        similarity_random = _select_distances(df_distance_matrix)
        similarity_in_group = _select_distances(
            df_distance_matrix,
            column_separate_combo=["drug_label"],
        )
        similarity_between_group = _select_distances(
            df_distance_matrix,
            column_comparison_combo=["drug_label"],
        )
        fig, ax = _plot_distance_distributions(
            [similarity_random, similarity_between_group, similarity_in_group],
            (6 * cm, 4 * cm),
            ["random", "betw.-\ngroup", "within-\ngroup"],
        )
        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_distances_{metric}.pdf"),
            transparent=True,
        )
    case "kapucu":
        similarity_random = _select_distances(df_distance_matrix)
        similarity_in_group = _select_distances(
            df_distance_matrix,
            column_separate_combo=["culture_type"],
        )
        similarity_between_group = _select_distances(
            df_distance_matrix,
            column_comparison_combo=["culture_type"],
        )
        similarity_day_to_day = _select_distances(
            df_distance_matrix,
            column_separate_combo=["culture_type", "mea_number", "well_id"],
            column_day_to_day="DIV",
        )
        fig, ax = _plot_distance_distributions(
            [
                similarity_random,
                similarity_between_group,
                similarity_in_group,
                similarity_day_to_day,
            ],
            (5 * cm, 4.5 * cm),
            ["random", "betw.-group", "in-group", "day-over-day"],
            xtick_labels_args={"rotation": 30, "ha": "right"},
        )
        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_distances_{metric}.pdf"),
            transparent=True,
        )
    case "wagenaar":
        similarity_random = _select_distances(df_distance_matrix)
        similarity_in_group = _select_distances(
            df_distance_matrix,
            column_separate_combo=["batch"],
            column_comparison_combo=["culture"],
        )
        similarity_between_group = _select_distances(
            df_distance_matrix,
            column_comparison_combo=["batch"],
        )
        similarity_day_to_day = _select_distances(
            df_distance_matrix,
            column_separate_combo=["batch", "culture"],
            column_day_to_day="day",
        )
        similarity_culture = _select_distances(
            df_distance_matrix,
            column_separate_combo=["batch", "culture"],
        )
        similarity_day_to_day_across_group = _select_distances(
            df_distance_matrix,
            column_separate_combo=["batch"],
            column_day_to_day="day",
        )
        fig, ax = _plot_distance_distributions(
            [
                similarity_random,
                # similarity_between_group,
                similarity_in_group,
                similarity_culture,
                similarity_day_to_day,
                # similarity_day_to_day_across_group,
            ],
            (6 * cm, 6 * cm),
            [
                "Random",
                # "Betw.-litter",
                "Litter",
                "Culture",
                "Day-over-day",
                # "Day-to-day\n(across litters)"
            ],
            xtick_labels_args={"rotation": 45, "ha": "right"},
            y_min=None,
            plot_statistical_test=[(0, 1), (1, 2), (2, 3)],
            plot_stats_lims=(2, 4.5),
        )
        ax.set_position([0.3, 0.5, 0.65, 0.45])
        fig.show()
        savefig(fig, f"{dataset}_distances_{metric}", file_format=["pdf", "svg"])

    case "mossink":
        # add additional info to the index to evaluate them in terms of distance/similarity
        df_cultures_metadata = load_df_cultures(burst_extraction_params)
        df_cultures_metadata = df_cultures_metadata[
            df_cultures_metadata["n_bursts"] > 0
        ]
        assert len(df_cultures_metadata) == len(df_distance_matrix)
        index_names = df_cultures_metadata.index.names
        df_cultures_metadata.reset_index(inplace=True)
        df_cultures_metadata.set_index(
            index_names + ["gender", "age", "coating", "batch"], inplace=True
        )

        # add the index to df_distance_matrix
        df_distance_matrix.index = df_cultures_metadata.index
        df_distance_matrix.columns = df_cultures_metadata.index

        # compute and plot as usual
        # WARNING - this computation takes a couple of minutes because of the many index levels.
        similarity_random = _select_distances(df_distance_matrix)
        similarity_in_group = _select_distances(
            df_distance_matrix,
            column_separate_combo=["group"],
        )
        similarity_between_group = _select_distances(
            df_distance_matrix,
            column_comparison_combo=["group"],
        )
        similarity_in_subject = _select_distances(
            df_distance_matrix,
            column_separate_combo=["group", "subject_id"],
        )
        similarity_in_gender = _select_distances(
            df_distance_matrix,
            column_separate_combo=["gender"],
        )
        similarity_in_coating = _select_distances(
            df_distance_matrix,
            column_separate_combo=["coating"],
        )
        _isogenic_map = {
            ("KS", 3): ("Control", 9),
            ("KS", 4): ("Control", 10),
            ("MELAS", 1): ("Control", 2),
            ("MELAS", 2): ("Control", 4),
            ("MELAS", 3): ("Control", 5),
        }
        similarity_subject_pair = []
        for (group_1, subject_1), (group_2, subject_2) in _isogenic_map.items():

            def _function_subject_pair(
                group1,
                subject_id1,
                group2,
                subject_id2,
            ):
                selection = (
                    (group1 == group_1)
                    & (subject_id1 == subject_1)
                    & (group2 == group_2)
                    & (subject_id2 == subject_2)
                )
                return selection

            similarity_subject_pair.append(
                _select_distances(
                    df_distance_matrix,
                    special_columns=["group", "subject_id"],
                    special_function=_function_subject_pair,
                ).dropna()
            )

        print("\nstandard")
        fig, ax = _plot_distance_distributions(
            [
                similarity_random,
                # similarity_between_group,
                similarity_in_group,
                similarity_in_subject,
                similarity_in_gender,
                similarity_in_coating,
            ],
            (7 * cm, 5 * cm),
            [
                "random",
                # "betw.-\ngroup",
                "group",  # "within-\ngroup",
                "subject",
                "gender",
                "coating",
            ],
            xtick_labels_args={"rotation": 45, "ha": "right"},
            y_min=None,
            plot_statistical_test=[(0, 2)],
        )
        fig.show()
        savefig(fig, f"{dataset}_distances_{metric}", file_format=["pdf", "svg"])

        print("\nreduced for paper")
        fig, ax = _plot_distance_distributions(
            [
                similarity_random,
                similarity_in_group,
                similarity_in_subject,
            ],
            (5 * cm, 6 * cm),
            [
                "Random",
                "Group",
                "Subject",
            ],
            xtick_labels_args={"rotation": 45, "ha": "right"},
            y_min=None,
            plot_statistical_test=[(0, 1), (1, 2)],
            plot_stats_lims=(2.5, 4),
        )
        ax.set_position([0.4, 0.5, 0.65, 0.45])
        fig.show()
        savefig(
            fig, f"{dataset}_distances_{metric}_reduced", file_format=["pdf", "svg"]
        )

        print("\nisogenic pairs")
        fig, ax = _plot_distance_distributions(
            [
                similarity_random,
                *similarity_subject_pair,
            ],
            (6 * cm, 6 * cm),
            [
                "random",
                *[f"{key}-{value}" for key, value in _isogenic_map.items()],
            ],
            xtick_labels_args={"rotation": 45, "ha": "right"},
        )
        fig.show()
        savefig(fig, f"{dataset}_distances_{metric}_pairs", file_format=["pdf", "svg"])

        print("\nisogenic pairs reduced")
        fig, ax = _plot_distance_distributions(
            [
                similarity_between_group,
                *similarity_subject_pair[2:],  # only MELAS pairs
            ],
            (6 * cm, 6 * cm),
            [
                "Between group",
                # *[f"{key}-{value}" for key, value in list(_isogenic_map.items())[2:]],  # only MELAS pairs
                "MELAS 1 - Contr. 2",
                "MELAS 2 - Contr. 4",
                "MELAS 3 - Contr. 5",
            ],
            xtick_labels_args={"rotation": 45, "ha": "right"},
            y_min=None,
            plot_statistical_test=[(0, 1), (0, 2), (0, 3)],
            plot_stats_lims=(2, 4.3),
        )
        ax.set_position([0.35, 0.5, 0.6, 0.45])
        fig.show()
        savefig(
            fig,
            f"{dataset}_distances_{metric}_pairs_reduced",
            file_format=["pdf", "svg"],
        )
    case "hommersom_binary":
        df_cultures_metadata = load_df_cultures(burst_extraction_params)
        df_cultures_metadata = df_cultures_metadata[
            df_cultures_metadata["n_bursts"] > 0
        ]
        assert len(df_cultures_metadata) == len(df_distance_matrix)
        index_names = df_cultures_metadata.index.names
        df_cultures_metadata.reset_index(inplace=True)
        df_cultures_metadata.set_index(index_names + ["group"], inplace=True)

        # add the index to df_distance_matrix
        df_distance_matrix.index = df_cultures_metadata.index
        df_distance_matrix.columns = df_cultures_metadata.index

        # compute and plot as usual
        similarity_random = _select_distances(df_distance_matrix)
        similarity_in_group = _select_distances(
            df_distance_matrix,
            column_separate_combo=["group"],
        )
        similarity_in_batch = _select_distances(
            df_distance_matrix,
            column_separate_combo=["batch"],
        )
        fig, ax = _plot_distance_distributions(
            [
                similarity_random,
                similarity_in_group,
                similarity_in_batch,
            ],
            (5 * cm, 4.5 * cm),
            [
                "random",
                "group",
                "batch",
            ],
            xtick_labels_args={"rotation": 45, "ha": "right"},
        )
        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_distances_{metric}.pdf"),
            transparent=True,
        )
    case _:
        raise NotImplementedError(f"dataset {dataset} not implemented")
