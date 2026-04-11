import os
import warnings

import baycomp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence.knn_clustering import load_knn_clustering_results_cv
from src.persistence.xgboost import load_xgboost_results
from src.plot import get_group_colors, prepare_plotting, savefig
from src.settings import (
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)

cm = prepare_plotting()

shape_dimensions = np.arange(1, 11)

burst_extraction_params_list = [
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_wagenaar_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4",
    "burst_dataset_mossink_KS",
]

cv_type = "StratifiedShuffleSplit"

distance_metrics_ = [
    "wasserstein",
    "KLDivergence",
    "JensenShannon",
    "euclidean",
    "Correlation",
]

n_classes = {}
df_accuracies = []
for burst_extraction_params in burst_extraction_params_list:
    dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
    for distance_metric in distance_metrics_:
        spectral_clustering_params = get_chosen_spectral_embedding_params(
            dataset, metric=distance_metric
        )

        for feature_set_name in ["combined", "traditional", "shape_manual"] + [
            f"shape_{i}D" for i in shape_dimensions
        ]:
            _, nested_scores, _, _, all_y_test = load_xgboost_results(
                burst_extraction_params,
                spectral_clustering_params,
                feature_set_name,
                cv_type,
            )
            n_classes[dataset] = len(np.unique(all_y_test))
            for i_score, score in enumerate(nested_scores):
                df_accuracies.append(
                    {
                        "dataset": dataset,
                        "distance_metric": distance_metric,
                        "feature_set": feature_set_name,
                        "cv_fold": i_score,
                        "score": score,
                    }
                )
        try:
            nested_scores, _, _ = load_knn_clustering_results_cv(
                burst_extraction_params, spectral_clustering_params, cv_type
            )
            for i_score, score in enumerate(nested_scores):
                df_accuracies.append(
                    {
                        "dataset": dataset,
                        "distance_metric": distance_metric,
                        "feature_set": "knn_clustering",
                        "cv_fold": i_score,
                        "score": score,
                    }
                )
        except FileNotFoundError:
            warnings.warn(
                f"KNN Clustering not found for dataset={dataset}. "
                "Continuing without it."
            )
            pass
df_accuracies = pd.DataFrame(df_accuracies)
df_accuracies["cv_fold"] = df_accuracies["cv_fold"].astype(int)


def _std2_corrected_std(std, kr, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    std : float
        Standard deviation of the score metrics of one model.
    kr : int
        Number of times the model was evaluated.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    corrected_var = (std**2) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def _get_stats(baseline="traditional", groupby=None):
    if groupby is None:
        groupby = ["dataset", "distance_metric", "feature_set"]
    df_stats = df_accuracies.groupby(groupby)["score"].agg(["mean", "std"])
    df_stats["sem"] = df_stats["std"].apply(lambda x: _std2_corrected_std(x, 100, 4, 1))
    df_stats["Z"] = df_stats["mean"] / df_stats["sem"]
    return df_stats


# %% plot mean +- sem accuracy for each dataset, comparing distance metrics
# take feature_set shape_2D
df_stats = _get_stats()
feature_set_ = "shape_2D"  # "combined"
df_stats_feature_set = df_stats.loc[(slice(None), slice(None), feature_set_)]
dataset_order = [
    "inhibblock",
    "hommersom_binary",
    "mossink_KS",
    "wagenaar",
]

custom_label_mapping = {
    "wasserstein": "Wasserstein",
    "euclidean": "Euclidean",
    "Correlation": "Correlation",
    "JensenShannon": "Jensen-Shannon",
    "KLDivergence": "KL Divergence",
}
color_map = sns.color_palette("Set2", n_colors=len(distance_metrics_))

fig, axs = plt.subplots(figsize=(8 * cm, 4 * cm), ncols=len(dataset_order))

sns.despine()
for i, dataset_name in enumerate(dataset_order):
    ax = axs[i]
    stats_dataset = df_stats_feature_set.loc[dataset_name]
    ax.bar(
        distance_metrics_,
        stats_dataset.loc[distance_metrics_, "mean"],
        yerr=stats_dataset.loc[distance_metrics_, "sem"],
        capsize=1.5,
        error_kw={"elinewidth": 1},
        fill=True,
        # alpha=alpha,
        color=color_map,
        # edgecolor=edgecolor,
        # hatch=hatch,
        # linewidth=linewidth,
        label=[custom_label_mapping[fs] for fs in distance_metrics_]
        if i == 0
        else None,
        width=0.6,
        # rasterized=True,
    )
for i, dataset_name in enumerate(dataset_order):
    ax = axs[i]
    ax.set_ylim(1 / n_classes[dataset_name], None)
    if i == 3:
        ax.set_yticks([0.125, 0.25, 0.375, 0.5, 0.625])
    # ax.set_title(dataset_name)
    # ax.set_xlabel("Feature set")
    if i == 0:
        ax.set_ylabel("Balanced accuracy")
    ax.tick_params(bottom=True, labelbottom=False)

fig.tight_layout()
fig.subplots_adjust(bottom=0.28)
fig.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.26),
    ncol=3,
)
fig.show()
savefig(fig, f"accuracies_supplementary_{feature_set_}", file_format=["pdf", "svg"])

# %%plot saturation when using increasingly more dimensions
shape_labels = [f"shape_{i}D" for i in shape_dimensions]
fig, axs = plt.subplots(figsize=(12 * cm, 4 * cm), ncols=len(dataset_order))
sns.despine()
for i, dataset_name in enumerate(dataset_order):
    ax = axs[i]
    stats_dataset = df_stats.loc[dataset_name]
    for i_distance, distance_metric in enumerate(distance_metrics_):
        stats_distance_ = stats_dataset.loc[distance_metric]
        ax.plot(
            shape_dimensions,
            stats_distance_.loc[shape_labels, "mean"],
            # yerr=stats_dataset.loc[distance_metrics_, "sem"],
            # capsize=1.5,
            # error_kw={"elinewidth": 1},
            # fill=True,
            # alpha=alpha,
            color=color_map[i_distance],
            # edgecolor=edgecolor,
            # hatch=hatch,
            # linewidth=linewidth,
            label=custom_label_mapping[distance_metric] if i == 0 else None,
            # width=0.6,
            # rasterized=True,
        )

for i, dataset_name in enumerate(dataset_order):
    ax = axs[i]
    # ax.set_ylim(1 / n_classes[dataset_name], None)
    # if i == 3:
    #     ax.set_yticks([0.125, 0.25, 0.375, 0.5, 0.625])
    # ax.set_title(dataset_name)
    # ax.set_xlabel("Feature set")
    if i == 0:
        ax.set_ylabel("Balanced accuracy")
        ax.set_xlabel("#Shape dim.")
    # ax.tick_params(bottom=True, labelbottom=False)

fig.tight_layout()
fig.subplots_adjust(bottom=0.4)
fig.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.22),
    ncol=3,
)
fig.show()
savefig(fig, f"accuracies_supplementary_shape_dimensions", file_format=["pdf", "svg"])
