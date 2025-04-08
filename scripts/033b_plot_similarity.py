import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from src import folders
from src.folders import get_fig_folder
from src.persistence import (
    load_burst_matrix,
    load_clustering_labels,
    load_df_bursts,
    load_df_cultures,
)
from src.plot import get_cluster_colors, label_sig_diff, prepare_plotting

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
    n_clusters = 4
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
    n_clusters = 4
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
    n_clusters = 4
elif "mossink" in burst_extraction_params:
    dataset = "mossink"
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

np.random.seed(0)

# plot settings
# n_clusters = 5  # 3  # if None chooses the number of clusters with Davies-Bouldin index

# plotting
fig_path = folders.get_fig_folder()

# load bursts
df_bursts = load_df_bursts(burst_extraction_params)
np.random.seed(0)

match dataset:
    case "kapucu":
        index_names = ["culture_type", "mea_number", "well_id", "DIV"]
    case "wagenaar":
        index_names = ["batch", "culture", "day"]
    case "hommersom":
        index_names = ["batch", "clone", "well_idx"]
    case "inhibblock":
        index_names = ["drug_label", "div", "well_idx"]
    case "mossink":
        index_names = ["group", "subject_id", "well_idx"]
    case _:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
# %% get clusters from linkage
# print("Getting clusters from linkage...")
# labels = get_agglomerative_labels(
#     n_clusters, burst_extraction_params, agglomerating_clustering_params
# )
clustering = load_clustering_labels(
    clustering_params, burst_extraction_params, labels_params, cv_params, cv_split
)
df_bursts["cluster"] = clustering.labels_[n_clusters] + 1

# Define a color palette for the clusters
# palette = sns.color_palette(n_colors=n_clusters)  # "Set1", n_clusters)
# cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
# cluster_colors = [
#     f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
#     for c in cluster_colors
# ]
palette = get_cluster_colors(n_clusters)
cluster_colors = get_cluster_colors(n_clusters)


# %% build new dataframe df_cultures with index ('batch', 'culture', 'day') and columns ('n_bursts', 'cluster_abs', 'cluster_rel')
print("Building df_cultures...")
df_bursts_reset = df_bursts.reset_index(
    drop=False
)  # reset index to access columns in groupby()
df_cultures = df_bursts_reset.groupby(index_names).agg(
    n_bursts=pd.NamedAgg(column="i_burst", aggfunc="count")
)

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

# dd cluster information
for i_cluster in range(1, n_clusters + 1):
    col_cluster = "cluster"
    df_cultures[f"cluster_abs_{i_cluster}"] = df_bursts.groupby(index_names)[
        col_cluster
    ].agg(lambda x: np.sum(x == i_cluster))
    df_cultures[f"cluster_rel_{i_cluster}"] = (
        df_cultures[f"cluster_abs_{i_cluster}"] / df_cultures["n_bursts"]
    )

# %%
columns = [f"cluster_rel_{i_cluster}" for i_cluster in range(1, n_clusters + 1)]


def _cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def _cosine_similarity_selection(df_selection, df_selection2=None):
    # average between all combinations
    is_comparison = df_selection2 is not None
    n_samples = len(df_selection)
    fraction_array = df_selection[columns].to_numpy()
    if is_comparison:
        n_samples2 = len(df_selection2)
        fraction_array2 = df_selection2[columns].to_numpy()
        if n_samples == 0 or n_samples2 == 0:
            return np.array([np.nan])
    else:
        if n_samples <= 1:
            return np.array([np.nan])
        n_samples2 = n_samples
        fraction_array2 = fraction_array
    similarities = np.zeros((n_samples, n_samples2))
    for i_sample in range(n_samples):
        for j_sample in range(n_samples2):
            similarities[i_sample, j_sample] = _cosine_similarity(
                fraction_array[i_sample],
                fraction_array2[j_sample],
            )
    if is_comparison:
        similarities_avg1 = np.mean(similarities, axis=1)
        similarities_avg2 = np.mean(similarities, axis=0)
        similarities_avg = np.concatenate((similarities_avg1, similarities_avg2))
        similarities = similarities.flatten()
    else:
        # average over upper triangle
        similarities_avg = (np.sum(similarities, axis=1) - similarities.diagonal()) / (
            n_samples - 1
        )
        similarities = similarities[np.triu_indices(n_samples, k=1)].flatten()
    return similarities, similarities_avg


def _cosine_similarity_df(
    df,
    column_separate_combo=None,
    column_comparison_combo=None,
    column_paired_comparison_combo=None,
    column_day_to_day=None,
):
    if column_paired_comparison_combo is not None:
        raise NotImplementedError
    similarities_list = []
    for _, value_separate_combo in (
        df.reset_index()[column_separate_combo].drop_duplicates().iterrows()
        if column_separate_combo is not None
        else enumerate([None])
    ):
        if value_separate_combo is None:
            df_separated_combo = df  # No filtering needed
        else:
            conditions = [
                df.index.get_level_values(col) == value_separate_combo[col]
                for col in column_separate_combo
            ]
            df_separated_combo = df.loc[np.logical_and.reduce(conditions)]
        similarity = {}
        for col in column_separate_combo if column_separate_combo is not None else []:
            similarity[col] = value_separate_combo[col]
        if column_comparison_combo is None:
            if column_day_to_day is not None:
                unique_days = df_separated_combo.index.get_level_values(
                    column_day_to_day
                ).unique()
                if len(unique_days) <= 1:
                    continue
                similarities_per_day = []
                for day_pre, day_post in zip(unique_days[:-1], unique_days[1:]):
                    similarities_per_day.append(
                        _cosine_similarity_selection(
                            df_separated_combo.loc[
                                df_separated_combo.index.get_level_values(
                                    column_day_to_day
                                )
                                == day_pre
                            ],
                            df_separated_combo.loc[
                                df_separated_combo.index.get_level_values(
                                    column_day_to_day
                                )
                                == day_post
                            ],
                        )[0]
                    )
                similarity["similarities"] = np.array(similarities_per_day)
                similarity["similarities_avg"] = np.mean(similarity["similarities"])
                similarity["days_pre"] = unique_days[:-1].tolist()
                similarity["days_post"] = unique_days[1:].tolist()
                similarities_list.append(similarity)
            else:
                (
                    similarity["similarities"],
                    similarity["similarities_avg"],
                ) = _cosine_similarity_selection(df_separated_combo)
                similarities_list.append(similarity)
        else:
            unique_comparison_values = (
                df_separated_combo.reset_index()[column_comparison_combo]
                .drop_duplicates()
                .values
            )
            for comparison_combo1 in unique_comparison_values:
                conditions_comparison_combo1 = [
                    df_separated_combo.index.get_level_values(col)
                    == comparison_combo1[i]
                    for i, col in enumerate(column_comparison_combo)
                ]
                for comparison_combo2 in unique_comparison_values:
                    if comparison_combo1 == comparison_combo2:
                        continue
                    conditions_comparison_combo2 = [
                        df_separated_combo.index.get_level_values(col)
                        == comparison_combo2[i]
                        for i, col in enumerate(column_comparison_combo)
                    ]
                    similarity_copy = similarity.copy()
                    for i, col in enumerate(column_comparison_combo):
                        similarity_copy[col] = comparison_combo1[i]
                    for i, col in enumerate(column_comparison_combo):
                        similarity_copy[col + "_comp"] = comparison_combo2[i]
                    (
                        similarity_copy["similarities"],
                        similarity_copy["similarities_avg"],
                    ) = _cosine_similarity_selection(
                        df_separated_combo.loc[
                            np.logical_and.reduce(conditions_comparison_combo1)
                        ],
                        df_separated_combo.loc[
                            np.logical_and.reduce(conditions_comparison_combo2)
                        ],
                    )
                    similarities_list.append(similarity_copy)

    return pd.DataFrame(similarities_list)


# %% mossink: load meta data
if dataset == "mossink":
    # df_cultures.drop(["gender", "age", "coating", "batch"], axis=1, inplace=True)
    df_cultures_metadata = load_df_cultures(burst_extraction_params)
    df_cultures_metadata = df_cultures_metadata[df_cultures_metadata["n_bursts"] > 0]
    df_cultures = df_cultures.join(
        df_cultures_metadata[["gender", "age", "coating", "batch"]], rsuffix="_delete"
    )
    del df_cultures_metadata
    df_cultures.reset_index(inplace=True)
    df_cultures.set_index(
        index_names + ["gender", "age", "coating", "batch"], inplace=True
    )

# %%
match dataset:
    case "inhibblock":
        similarity_random = _cosine_similarity_df(df_cultures)
        similarity_in_group = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["drug_label"],
        )
        similarity_between_group = _cosine_similarity_df(
            df_cultures,
            column_comparison_combo=["drug_label"],
        )

        cm = 1 / 2.54
        fig, ax = plt.subplots(constrained_layout=True, figsize=(5 * cm, 3.5 * cm))
        sns.despine()

        # Data preparation
        data = [
            np.concatenate(similarity_random["similarities"].values),
            np.concatenate(similarity_between_group["similarities"].values),
            np.concatenate(similarity_in_group["similarities"].values),
        ]
        data_avg = [
            np.concatenate(similarity_random["similarities_avg"].values),
            np.concatenate(similarity_between_group["similarities_avg"].values),
            np.concatenate(similarity_in_group["similarities_avg"].values),
        ]

        # Violin plot with white fill
        sns.violinplot(
            data=data,
            ax=ax,
            inner="box",
            facecolor=(1, 1, 1, 0),
            edgecolor="black",
            inner_kws={"color": "grey", "zorder": 0},
        )

        # Overlay dots for individual data points
        # sns.stripplot(data=data, ax=ax, color="black", size=0.3, jitter=True)
        # sns.boxplot(data=data, ax=ax, color="grey", size=0.3)

        # Labeling
        ax.set_xticks(list(range(len(data))))
        ax.set_xticklabels(
            ["random", "betw.-\ngroup", "within-\ngroup"],
            rotation=0,  # ha="right"
        )
        ax.get_xticklabels()[0].set_transform(
            ax.get_xticklabels()[0].get_transform()
            + matplotlib.transforms.Affine2D().translate(-7, 0)
        )
        ax.get_xticklabels()[2].set_transform(
            ax.get_xticklabels()[2].get_transform()
            + matplotlib.transforms.Affine2D().translate(4, 0)
        )
        ax.set_ylabel("Similarity")
        ax.set_yticks([0, 1])

        for i, distribution1 in enumerate(data_avg):
            for j, distribution2 in enumerate(data_avg):
                if i >= j:
                    continue
                print(f"{i} vs {j}")
                test_mannwhitneyu = scipy.stats.mannwhitneyu(
                    distribution1,
                    distribution2,
                )
                print(test_mannwhitneyu)
                if test_mannwhitneyu.pvalue < 0.001:
                    label_sig_diff(
                        ax,
                        (i, j),
                        1,
                        r"***",
                        0.3 if j - i == 1 else 0.9,
                        0.1,
                        "k",
                        ft_sig=10,
                        lw_sig=1,
                    )

        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_similarity.svg"),
            transparent=True,
        )
    case "kapucu":
        similarity_random = _cosine_similarity_df(df_cultures)
        similarity_in_group = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["culture_type"],
        )
        similarity_between_group = _cosine_similarity_df(
            df_cultures,
            column_comparison_combo=["culture_type"],
        )
        similarity_day_to_day = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["culture_type", "mea_number", "well_id"],
            column_day_to_day="DIV",
        )
        cm = 1 / 2.54
        fig, ax = plt.subplots(constrained_layout=True, figsize=(5 * cm, 4.5 * cm))
        sns.despine()

        # Data preparation
        data = [
            np.concatenate(similarity_random["similarities"].values),
            np.concatenate(similarity_between_group["similarities"].values),
            np.concatenate(similarity_in_group["similarities"].values),
            np.concatenate(similarity_day_to_day["similarities"].values).flatten(),
        ]

        data_avg = [
            np.concatenate(similarity_random["similarities_avg"].values),
            np.concatenate(similarity_between_group["similarities_avg"].values),
            np.concatenate(similarity_in_group["similarities_avg"].values),
            similarity_day_to_day["similarities_avg"].values,
        ]

        xtick_labels = ["random", "betw.-group", "in-group", "day-to-day"]

        # Violin plot with white fill
        sns.violinplot(
            data=data,
            ax=ax,
            inner="box",
            facecolor=(1, 1, 1, 0),
            edgecolor="black",
            inner_kws={"color": "grey", "zorder": 0},
        )
        # sns.violinplot(data=data, ax=ax, inner="box", color="white", edgecolor="black")

        # Overlay dots for individual data points
        # sns.stripplot(data=data, ax=ax, color="black", size=0.08, jitter=True)

        # Labeling
        ax.set_xticks(list(range(len(data))))
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right")
        ax.set_yticks([0, 1])
        ax.set_ylabel("Similarity")

        for i, distribution1 in enumerate(data_avg):
            for j, distribution2 in enumerate(data_avg):
                if i >= j:
                    continue
                print(f"{i} vs {j}")
                test_mannwhitneyu = scipy.stats.mannwhitneyu(
                    distribution1,
                    distribution2,
                )
                print(test_mannwhitneyu)
                if test_mannwhitneyu.pvalue < 0.05:
                    label_sig_diff(
                        ax,
                        (i, j),
                        1,
                        test_mannwhitneyu.pvalue,
                        (j - i) * 0.6 - 0.3,
                        0.1,
                        "k",
                        ft_sig=10,
                        lw_sig=1,
                    )

        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_similarity.svg"),
            transparent=True,
        )

    case "wagenaar":
        similarity_random = _cosine_similarity_df(df_cultures)
        similarity_in_group = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["batch"],
        )
        similarity_between_group = _cosine_similarity_df(
            df_cultures,
            column_comparison_combo=["batch"],
        )
        similarity_day_to_day = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["batch", "culture"],
            column_day_to_day="day",
        )
        cm = 1 / 2.54
        fig, ax = plt.subplots(constrained_layout=True, figsize=(5 * cm, 4.5 * cm))
        sns.despine()

        # Data preparation
        data = [
            np.concatenate(similarity_random["similarities"].values),
            np.concatenate(similarity_between_group["similarities"].values),
            np.concatenate(similarity_in_group["similarities"].values),
            np.concatenate(similarity_day_to_day["similarities"].values).flatten(),
        ]

        data_avg = [
            np.concatenate(similarity_random["similarities_avg"].values),
            np.concatenate(similarity_between_group["similarities_avg"].values),
            np.concatenate(similarity_in_group["similarities_avg"].values),
            similarity_day_to_day["similarities_avg"].values,
        ]

        xtick_labels = ["random", "betw.-group", "in-group", "day-to-day"]

        # Violin plot with white fill
        sns.violinplot(
            data=data,
            ax=ax,
            inner="box",
            facecolor=(1, 1, 1, 0),
            edgecolor="black",
            inner_kws={"color": "grey", "zorder": 0},
        )
        # sns.violinplot(data=data, ax=ax, inner="box", color="white", edgecolor="black")

        # Overlay dots for individual data points
        # sns.stripplot(data=data, ax=ax, color="black", size=0.08, jitter=True)

        # Labeling
        ax.set_xticks(list(range(len(data))))
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right")
        ax.set_yticks([0, 1])
        ax.set_ylabel("Similarity")

        for i, distribution1 in enumerate(data_avg):
            for j, distribution2 in enumerate(data_avg):
                if i >= j:
                    continue
                print(f"{i} vs {j}")
                test_mannwhitneyu = scipy.stats.mannwhitneyu(
                    distribution1,
                    distribution2,
                )
                print(test_mannwhitneyu)
                if test_mannwhitneyu.pvalue < 0.001:
                    label_sig_diff(
                        ax,
                        (i, j),
                        1,
                        r"***",
                        (j - i) * 0.6 - 0.3,
                        0.1,
                        "k",
                        ft_sig=10,
                        lw_sig=1,
                    )

        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_similarity.svg"),
            transparent=True,
        )
    case "mossink":
        similarity_random = _cosine_similarity_df(df_cultures)
        similarity_in_group = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["group"],
        )
        similarity_between_group = _cosine_similarity_df(
            df_cultures,
            column_comparison_combo=["group"],
        )
        similarity_in_subject = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["group", "subject_id"],
        )
        similarity_in_gender = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["gender"],
        )
        similarity_in_coating = _cosine_similarity_df(
            df_cultures,
            column_separate_combo=["coating"],
        )

        cm = 1 / 2.54
        fig, ax = plt.subplots(constrained_layout=True, figsize=(8 * cm, 3.5 * cm))
        sns.despine()

        # Data preparation
        data = [
            np.concatenate(similarity_random["similarities"].values),
            # np.concatenate(similarity_between_group["similarities"].values),
            np.concatenate(similarity_in_group["similarities"].values),
            np.concatenate(similarity_in_subject["similarities"].values),
            np.concatenate(similarity_in_gender["similarities"].values),
            np.concatenate(similarity_in_coating["similarities"].values),
        ]
        data_avg = [
            np.concatenate(similarity_random["similarities_avg"].values),
            # np.concatenate(similarity_between_group["similarities_avg"].values),
            np.concatenate(similarity_in_group["similarities_avg"].values),
            np.concatenate(similarity_in_subject["similarities_avg"].values),
            np.concatenate(similarity_in_gender["similarities_avg"].values),
            np.concatenate(similarity_in_coating["similarities_avg"].values),
        ]
        labels = [
            "random",
            # "betw.-\ngroup",
            "within-\ngroup",
            "subject",
            "gender",
            "coating",
        ]

        # Violin plot with white fill
        sns.violinplot(
            data=data,
            ax=ax,
            inner="box",
            facecolor=(1, 1, 1, 0),
            edgecolor="black",
            inner_kws={"color": "grey", "zorder": 0},
        )

        # Overlay dots for individual data points
        # sns.stripplot(data=data, ax=ax, color="black", size=0.3, jitter=True)
        # sns.boxplot(data=data, ax=ax, color="grey", size=0.3)

        # Labeling
        ax.set_xticks(list(range(len(data))))
        ax.set_xticklabels(
            labels,
            rotation=0,  # ha="right"
        )
        ax.set_ylabel("Similarity")
        ax.set_yticks([0, 1])

        for i, distribution1 in enumerate(data_avg):
            for j, distribution2 in enumerate(data_avg):
                if i >= j:
                    continue
                print(f"{i} vs {j}")
                test_mannwhitneyu = scipy.stats.mannwhitneyu(
                    distribution1,
                    distribution2,
                )
                print(test_mannwhitneyu)
                if test_mannwhitneyu.pvalue < 0.001:
                    label_sig_diff(
                        ax,
                        (i, j),
                        1,
                        r"***",
                        0.3 if j - i == 1 else 0.9,
                        0.1,
                        "k",
                        ft_sig=10,
                        lw_sig=1,
                    )

        fig.show()
        fig.savefig(
            os.path.join(get_fig_folder(), f"{dataset}_similarity.svg"),
            transparent=True,
        )
    case _:
        raise NotImplementedError(f"dataset {dataset} not implemented")
