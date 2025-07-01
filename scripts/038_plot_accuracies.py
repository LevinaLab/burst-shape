import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence.knn_clustering import load_knn_clustering_results
from src.persistence.xgboost import load_xgboost_results
from src.plot import prepare_plotting
from src.settings import (
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)

cm = prepare_plotting()

burst_extraction_params_list = [
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4",
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4",
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30",
]

n_classes = {}
df_accuracies = []
for burst_extraction_params in burst_extraction_params_list:
    dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
    spectral_clustering_params = get_chosen_spectral_embedding_params(dataset)

    for feature_set_name in ["combined", "shape", "traditional"]:
        _, nested_scores, _, _, all_y_test = load_xgboost_results(
            burst_extraction_params, spectral_clustering_params, feature_set_name
        )
        n_classes[dataset] = len(np.unique(all_y_test))
        for i_score, score in enumerate(nested_scores):
            df_accuracies.append(
                {
                    "dataset": dataset,
                    "feature_set": feature_set_name,
                    "cv_fold": i_score,
                    "score": score,
                }
            )
    try:
        score, _, _, _, _ = load_knn_clustering_results(
            burst_extraction_params, spectral_clustering_params
        )
        df_accuracies.append(
            {
                "dataset": dataset,
                "feature_set": "knn_clustering",
                "cv_fold": 0,  # there is only one sample
                "score": float(score),
            }
        )
    except FileNotFoundError:
        warnings.warn(
            f"KNN Clustering not found for dataset={dataset}. " "Continuing without it."
        )
        pass
df_accuracies = pd.DataFrame(df_accuracies)
df_accuracies["cv_fold"] = df_accuracies["cv_fold"].astype(int)
# %%
dataset_list = df_accuracies["dataset"].unique()
fig, axs = plt.subplots(nrows=len(dataset_list), figsize=(5, 10))
for i, dataset_name in enumerate(dataset_list):
    ax = axs[i]
    sns.lineplot(
        data=df_accuracies[
            (df_accuracies["dataset"] == dataset_name)
            & (df_accuracies["feature_set"] != "knn_clustering")
        ],
        x="cv_fold",
        y="score",
        hue="feature_set",
        palette="Set1",
        marker="o",
        alpha=0.5,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("CV test fold")
    ax.set_ylabel("Model Score")
    ax.tick_params(bottom=True, labelbottom=False)
fig.legend()
fig.tight_layout()
fig.show()
# %%
custom_label_mapping = {
    "knn_clustering": "Shape KNN",
    "traditional": "XGBoost trad.",
    "shape": "XGBoost shape",
    "combined": "XGBoost comb.",
}
# Create the FacetGrid: one subplot per dataset
g = sns.FacetGrid(
    df_accuracies,
    col="dataset",
    # hue="feature_set",
    sharey=False,
    height=9 * cm,
    aspect=0.4,
)
plot_type = (
    # "pointplot"
    "boxplot"
    # "violinplot"
    # "stripplot"
)
match plot_type:
    case "pointplot":
        g.map_dataframe(
            sns.pointplot,
            x="feature_set",
            y="score",
            hue="feature_set",
            palette="Set2",
            errorbar="sd",
            markers="_",  # Horizontal line marker
            markersize=15,
            dodge=0,
            order=["knn_clustering", "traditional", "shape", "combined"],
        )
    case "boxplot":
        g.map_dataframe(
            sns.boxplot,
            x="feature_set",
            y="score",
            hue="feature_set",
            palette="Set2",
            order=["knn_clustering", "traditional", "shape", "combined"],
        )
    case "violinplot":
        g.map_dataframe(
            sns.violinplot,
            x="feature_set",
            y="score",
            hue="feature_set",
            palette="Set2",
            order=["knn_clustering", "traditional", "shape", "combined"],
            inner="box",
        )
    case "stripplot":
        g.map_dataframe(
            sns.stripplot,
            x="feature_set",
            y="score",
            hue="feature_set",
            palette="Set2",
            order=["knn_clustering", "traditional", "shape", "combined"],
            dodge=False,  # No dodging along the x-axis (unless you have multiple hue levels per x)
            jitter=True,  # Add some jitter to spread out points
            size=6,  # Marker size
        )
for ax in g.axes.flat:
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [
            custom_label_mapping.get(label.get_text(), label.get_text())
            for label in ax.get_xticklabels()
        ],
        rotation=90,
        ha="right",
    )

# Add dashed line for random accuracy
for dataset, ax in zip(df_accuracies["dataset"].unique(), g.axes.flat):
    random_level = 1 / n_classes[dataset]
    ax.axhline(random_level, ls="--", color="gray", label="Chance level")

# Adjust the plot
g.set_axis_labels(
    "",  # "Feature Set",
    "Balanced Accuracy",
)
g.set_titles(col_template="{col_name}")
g.tight_layout()

fig = g.figure
fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), f"accuracies_all_data.svg"),
    transparent=True,
)

# %% statistical tests
from itertools import combinations

from scipy.stats import t, ttest_rel, wilcoxon


def _corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
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
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


results = []

# Iterate over each dataset
for dataset, df_dataset in df_accuracies[
    df_accuracies["feature_set"] != "knn_clustering"
].groupby("dataset"):
    # Pivot the table to have feature_set as columns, cv_fold as rows
    pivot = df_dataset.pivot(index="cv_fold", columns="feature_set", values="score")

    # Only compare models with complete score vectors (i.e., drop NaNs)
    pivot = pivot.dropna(axis=0)  # drop cv_folds with missing scores

    # Get all pairs of feature sets
    for fs1, fs2 in combinations(pivot.columns, 2):
        scores1 = pivot[fs1]
        scores2 = pivot[fs2]

        # Store the result
        results.append(
            {
                "dataset": dataset,
                "feature_set_1": fs1,
                "feature_set_2": fs2,
                "diff": (scores1 - scores2),
            }
        )

# Convert results to DataFrame
df_results = pd.DataFrame(results)
df_results["n"] = df_results["diff"].apply(lambda x: len(x) - 1)
df_results["mean"] = df_results["diff"].apply(np.mean)
df_results["std_corr"] = df_results["diff"].apply(lambda x: _corrected_std(x, 4, 1))
df_results["t_stat"] = df_results["mean"] / df_results["std_corr"]
for index in df_results.index:
    df_results.at[index, "p_val"] = 2 * t.sf(
        np.abs(df_results.at[index, "t_stat"]), df_results.at[index, "n"]
    )
df_results["sign."] = df_results["p_val"].apply(lambda x: "*" if x < 0.05 else "n.s.")
print(
    df_results[
        ["dataset", "feature_set_1", "feature_set_2", "t_stat", "p_val", "sign."]
    ]
)


df_results["better"] = df_results["diff"].apply(lambda x: np.mean(x > 0))
df_results["worse"] = df_results["diff"].apply(lambda x: np.mean(x < 0))
df_results["equal"] = df_results["diff"].apply(lambda x: np.mean(x == 0))
print(
    df_results[
        ["dataset", "feature_set_1", "feature_set_2", "better", "worse", "equal"]
    ]
)
