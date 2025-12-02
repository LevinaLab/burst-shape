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

shape_dimensions = np.arange(1, 21)

burst_extraction_params_list = [
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4",
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4",
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_mossink_KS",
    # "burst_dataset_mossink_MELAS",
]

cv_type = (
    # "RepeatedStratifiedKFold"
    "StratifiedShuffleSplit"
)

n_classes = {}
df_accuracies = []
for burst_extraction_params in burst_extraction_params_list:
    dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
    spectral_clustering_params = get_chosen_spectral_embedding_params(dataset)

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
                    "feature_set": "knn_clustering",
                    "cv_fold": i_score,
                    "score": score,
                }
            )
    except FileNotFoundError:
        warnings.warn(
            f"KNN Clustering not found for dataset={dataset}. " "Continuing without it."
        )
        pass
df_accuracies = pd.DataFrame(df_accuracies)
df_accuracies["cv_fold"] = df_accuracies["cv_fold"].astype(int)


def _get_stats(normalize=False, baseline="traditional", groupby=None):
    if groupby is None:
        groupby = ["dataset", "feature_set"]
    if normalize is True:
        df_accuracies_normalized = df_accuracies.copy()
        for dataset in df_accuracies["dataset"].unique():
            traditional_values = df_accuracies_normalized[
                (df_accuracies_normalized["dataset"] == dataset)
                & (df_accuracies_normalized["feature_set"] == baseline)
            ]["score"].values
            for feature_set in df_accuracies["feature_set"].unique():
                df_accuracies_normalized.loc[
                    (df_accuracies_normalized["dataset"] == dataset)
                    & (df_accuracies_normalized["feature_set"] == feature_set),
                    "score",
                ] = (
                    df_accuracies_normalized.loc[
                        (df_accuracies_normalized["dataset"] == dataset)
                        & (df_accuracies_normalized["feature_set"] == feature_set),
                        "score",
                    ]
                    - traditional_values
                )
        df_stats = df_accuracies_normalized.groupby(groupby)["score"].agg(
            ["mean", "std"]
        )
    else:
        df_stats = df_accuracies.groupby(["dataset", "feature_set"])["score"].agg(
            ["mean", "std"]
        )
    df_stats["sem"] = df_stats["std"].apply(lambda x: _std2_corrected_std(x, 100, 4, 1))
    df_stats["Z"] = df_stats["mean"] / df_stats["sem"]
    return df_stats


# %%
dataset_list = df_accuracies["dataset"].unique()
fig, axs = plt.subplots(nrows=len(dataset_list), figsize=(5, 10))
for i, dataset_name in enumerate(dataset_list):
    ax = axs[i]
    sns.lineplot(
        data=df_accuracies[
            (df_accuracies["dataset"] == dataset_name)
            # & (df_accuracies["feature_set"] != "knn_clustering")
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
    "knn_clustering": "    Individual\n    burst shapes",
    "traditional": "Traditional",
    "shape_2D": "shape_2D",
    "combined": "Combined",
    "shape_manual": "Simplified",
}
# Create the FacetGrid: one subplot per dataset
g = sns.FacetGrid(
    df_accuracies,
    col="dataset",
    # hue="feature_set",
    sharey=False,
    height=9 * cm,
    aspect=0.5,
)
plot_type = (
    # "pointplot"
    # "boxplot"
    # "violinplot"
    # "stripplot"
    "boxplot-manual"
)

# Add dashed line for random accuracy
for dataset, ax in zip(df_accuracies["dataset"].unique(), g.axes.flat):
    random_level = 1 / n_classes[dataset]
    ax.axhline(random_level, ls="--", color="gray", label="Chance level")

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
            order=["knn_clustering", "traditional", "shape_2D", "combined"],
        )
    case "boxplot":
        g.map_dataframe(
            sns.boxplot,
            x="feature_set",
            y="score",
            # hue="feature_set",
            # palette="Set2",
            order=["knn_clustering", "traditional", "shape_2D", "combined"],
            boxprops=dict(facecolor="none"),
            medianprops=dict(color="red"),
        )
    case "violinplot":
        g.map_dataframe(
            sns.violinplot,
            x="feature_set",
            y="score",
            hue="feature_set",
            palette="Set2",
            order=["knn_clustering", "traditional", "shape_2D", "combined"],
            inner="box",
        )
    case "stripplot":
        g.map_dataframe(
            sns.stripplot,
            x="feature_set",
            y="score",
            hue="feature_set",
            palette="Set2",
            order=["knn_clustering", "traditional", "shape_2D", "combined"],
            dodge=False,  # No dodging along the x-axis (unless you have multiple hue levels per x)
            jitter=True,  # Add some jitter to spread out points
            size=6,  # Marker size
        )
    case "boxplot-manual":
        positions = [0, 2, 3, 4]

        def _manual_boxplot(data, color, **kwargs):
            order = ["knn_clustering", "traditional", "shape_2D", "combined"]

            # Collect data for each feature_set in order
            grouped_scores = [
                data.loc[data["feature_set"] == fs, "score"] for fs in order
            ]

            plt.boxplot(
                grouped_scores,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor="none"),
                medianprops=dict(color="red"),
            )

            plt.text(
                x=(positions[1] + positions[3]) / 2,
                y=plt.ylim()[0] - 0.5 * (plt.ylim()[1] - plt.ylim()[0]),
                s="Summary\nfeatures",
                ha="center",
                va="top",
                fontsize=10,
            )
            plt.xticks(positions, order)  # Label x-axis ticks manually

        g.map_dataframe(_manual_boxplot)
for ax in g.axes.flat:
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [
            custom_label_mapping.get(label.get_text(), label.get_text())
            for label in ax.get_xticklabels()
        ],
        rotation=90,
        ha="center",
    )


# Adjust the plot
g.set_axis_labels(
    "",  # "Feature Set",
    "Balanced accuracy",
)
g.set_titles(col_template="{col_name}")
g.tight_layout()

fig = g.figure
fig.show()
for filetype in ("svg", "pdf"):
    fig.savefig(
        os.path.join(get_fig_folder(), f"accuracies_all_data.{filetype}"),
        transparent=True,
    )
# %% print accuracies
df_accuracies_mean = df_accuracies.groupby(["dataset", "feature_set"])["score"].agg(
    "mean"
)
print(df_accuracies_mean)
# to copy it directly into latex table
feature_set_order = ["knn_clustering", "traditional", "shape_2D", "combined"]
for dataset in df_accuracies_mean.index.get_level_values("dataset").unique():
    accuracies_mean_dataset = np.array(
        [
            df_accuracies_mean.loc[(dataset, feature_set)]
            for feature_set in feature_set_order
        ]
    )
    max_accuracy_dataset = np.argmax(accuracies_mean_dataset)
    print(
        dataset,
        " & "
        + " & ".join(
            [
                f"\\textbf{{{accuracy:.3f}}}"
                if max_accuracy_dataset == i
                else f"{accuracy:.3f}"
                for i, accuracy in enumerate(accuracies_mean_dataset)
            ]
        ),
        "\\\\",
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


results = []

# Iterate over each dataset
for dataset, df_dataset in df_accuracies[
    np.ones(
        len(df_accuracies), dtype=bool
    )  # df_accuracies["feature_set"] != "knn_clustering"
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
df_results["n"] = df_results["diff"].apply(lambda x: len(x))
df_results["mean"] = df_results["diff"].apply(np.mean)
df_results["std_corr"] = df_results["diff"].apply(lambda x: _corrected_std(x, 4, 1))
df_results["t_stat"] = df_results["mean"] / df_results["std_corr"]
for index in df_results.index:
    df_results.at[index, "p_val"] = 2 * t.sf(
        np.abs(df_results.at[index, "t_stat"]), df_results.at[index, "n"] - 1
    )
df_results["sign."] = df_results["p_val"].apply(lambda x: "*" if x < 0.05 else "n.s.")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
print("\nCorrected t-test (Nadeau and Bengio, 2003):")
print(
    df_results[
        ["dataset", "feature_set_1", "feature_set_2", "t_stat", "p_val", "sign."]
    ].to_string(line_width=1000)
)


df_results["better"] = df_results["diff"].apply(lambda x: np.mean(x > 0))
df_results["worse"] = df_results["diff"].apply(lambda x: np.mean(x < 0))
df_results["equal"] = df_results["diff"].apply(lambda x: np.mean(x == 0))
print("\nProportion of CV folds where first model was better/worse/equal:")
print(
    df_results[
        ["dataset", "feature_set_1", "feature_set_2", "better", "worse", "equal"]
    ].to_string(line_width=1000)
)

df_results["std"] = df_results["diff"].apply(np.std)
df_results["t_stat_naive"] = (
    df_results["mean"] / df_results["std"] * df_results["n"].apply(np.sqrt)
)
for index in df_results.index:
    df_results.at[index, "p_val_naive"] = 2 * t.sf(
        np.abs(df_results.at[index, "t_stat_naive"]), df_results.at[index, "n"]
    )
df_results["sign. naive"] = df_results["p_val_naive"].apply(
    lambda x: "*" if x < 0.05 else "n.s."
)
print("\nNaive t-test (not corrected for CV):")
print(
    df_results[
        [
            "dataset",
            "feature_set_1",
            "feature_set_2",
            "t_stat_naive",
            "p_val_naive",
            "sign. naive",
        ]
    ].to_string(line_width=1000)
)
# %% overall statistical testt
datasets = ["inhibblock", "hommersom_binary", "mossink_KS", "wagenaar"]  # , "kapucu"]

df_reml = []

for first, second in [
    ("combined", "traditional"),
    ("combined", "shape_2D"),
    ("shape_2D", "traditional"),
    ("knn_clustering", "shape_2D"),
    ("knn_clustering", "traditional"),
    ("shape_2D", "shape_manual"),
    ("knn_clustering", "shape_manual"),
    ("shape_5D", "shape_manual"),
    ("shape_10D", "shape_manual"),
]:
    print("\nFirst dataset:", first)
    print("Second dataset:", second)
    df_stats = _get_stats(normalize=True, baseline=second)
    df_stats_reduced = df_stats[
        (df_stats.index.get_level_values("feature_set") == first)
        & (df_stats.index.get_level_values("dataset").isin(datasets))
    ]
    print(df_stats_reduced)

    import statsmodels.stats.meta_analysis as meta

    res_REML = meta.combine_effects(
        df_stats_reduced["mean"].values,
        (df_stats_reduced["sem"].values ** 2),
        method_re="dl",
        alpha=0.001,
    )
    res_REML = res_REML.summary_frame()

    print("\nRandom-effects (REML):")
    print(res_REML.round(4))

    import scipy.stats

    z = res_REML.at["random effect", "eff"] / res_REML.at["random effect", "sd_eff"]
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
    print("pvalue=", p_value)

    # my approximate p-value
    my_z = (
        df_stats_reduced["mean"].values.mean()
        / np.sqrt((df_stats_reduced["sem"] ** 2).values.sum())
        * np.sqrt(len(df_stats_reduced))
    )
    my_p_value = 2 * (1 - scipy.stats.norm.cdf(abs(my_z)))
    print("my approximate p-value", my_p_value)

    df_reml.append(
        {
            "first": first,
            "second": second,
            "eff": res_REML.at["random effect", "eff"],
            "sd_eff": res_REML.at["random effect", "sd_eff"],
            "p_value": p_value,
        }
    )

df_reml = pd.DataFrame(df_reml)

# %% plot comparisons
custom_label_mapping = {
    "knn_clustering": "Individual",
    "traditional": "Traditional",
    "shape_2D": "Burst shape-2D",
    "shape_5D": "Burst shape-5D",
    "shape_10D": "Burst shape-10D",
    "combined": "Combined",
    "shape_manual": "Designed",
}
custom_dataset_mapping = {
    "inhibblock": "Inhib. block",
    "hommersom_binary": "CACNA1A",
    "mossink_KS": "Kleefstra",
    "wagenaar": "Rat Cortex",
    # "kapucu": "Rat vs hPSC",
}


def _plot_comparison(ax, first, second):
    ax.set_title(
        f"{custom_label_mapping[first]} vs. {custom_label_mapping[second]}", fontsize=7
    )
    df_stats = _get_stats(normalize=True, baseline=second)
    x_meta_analysis = -2
    ax.bar(
        x_meta_analysis,  # "Meta-analysis",  # \nRandom effect",
        df_reml.loc[
            (df_reml["first"] == first) & (df_reml["second"] == second), "eff"
        ].values[0],
        color="lightblue",
        fill=True,
        edgecolor="black",
        yerr=df_reml.loc[
            (df_reml["first"] == first) & (df_reml["second"] == second), "sd_eff"
        ].values[0],
        capsize=3,
        error_kw={"elinewidth": 1},
    )
    for i_dataset, dataset in enumerate(custom_dataset_mapping.keys()):
        mean_accuracy = df_stats.loc[(dataset, first), "mean"]
        std_accuracy = df_stats.loc[(dataset, first), "std"]
        ax.bar(
            i_dataset,  # custom_dataset_mapping[dataset],
            mean_accuracy,
            color="lightgrey" if mean_accuracy < 0 else "grey",
            fill=True,
            edgecolor="black",
            yerr=_std2_corrected_std(std_accuracy, 100, 4, 1),
            capsize=3,
            error_kw={"elinewidth": 1},
        )
    ax.set_xticks([x_meta_analysis] + list(range(len(custom_dataset_mapping.keys()))))
    ax.set_xticklabels(
        ["Meta-analysis"]
        + [custom_dataset_mapping[dataset] for dataset in custom_dataset_mapping.keys()]
    )
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("center")
        if label.get_text() == "CACNA1A":
            label.set_fontstyle("italic")

    y_label_string = (
        f"({custom_label_mapping[first]}) - ({custom_label_mapping[second]})"
    )
    ax.set_ylabel("Accuracy improvement\n" + y_label_string)
    return None


fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize=(9.5 * cm, 5 * cm), constrained_layout=True
)
sns.despine()
for i, (first, second) in enumerate(
    [
        ("combined", "traditional"),
        ("knn_clustering", "shape_2D"),
        ("knn_clustering", "traditional"),
    ]
):
    _plot_comparison(axs[i], first, second)
    if i > 0:
        axs[i].set_ylabel(axs[i].get_ylabel().split("\n")[1])
fig.show()
savefig(fig, f"accuracies_improvement", file_format=["pdf", "svg"])

fig, axs = plt.subplots(ncols=3, figsize=(9.5 * cm, 5 * cm), constrained_layout=True)
sns.despine()
for i, (first, second) in enumerate(
    [
        ("shape_2D", "shape_manual"),
        ("shape_5D", "shape_manual"),
        ("shape_10D", "shape_manual"),
    ]
):
    _plot_comparison(axs[i], first, second)
fig.show()
savefig(fig, f"accuracies_improvement_suppl", file_format=["pdf", "svg"])

# %% Bayesian comparison
# https://jmlr.org/papers/volume18/16-305/16-305.pdf
# https://github.com/janezd/baycomp
# https://baycomp.readthedocs.io/en/latest/
np.set_printoptions(precision=3, suppress=True)

df_bayesian = []
for first, second in [
    ("combined", "traditional"),
    ("knn_clustering", "traditional"),
    ("knn_clustering", "shape_2D"),
]:
    print(f"\nBayesian test: {first} better than {second}?")
    for dataset in df_accuracies["dataset"].unique():
        probs, fig = baycomp.two_on_single(
            df_accuracies[
                (df_accuracies["dataset"] == dataset)
                & (df_accuracies["feature_set"] == first)
            ]["score"].values,
            df_accuracies[
                (df_accuracies["dataset"] == dataset)
                & (df_accuracies["feature_set"] == second)
            ]["score"].values,
            plot=True,
            runs=20,
            rope=0.01,
        )
        print(dataset, np.array(probs))
        df_bayesian.append(
            {
                "dataset": dataset,
                "feature_set_1": first,
                "feature_set_2": second,
                "prob_first_better": probs[0],
                "prob_equal": probs[1],
                "prob_second_better": probs[2],
            }
        )
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        ax.set_xlabel(xlabel[:10] + "\n" + xlabel[11:], fontsize=10, ha="left", x=0.0)
        fig.set_size_inches(6.5 * cm, 4 * cm)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 0.4),
            ncol=1,
            fontsize=10,
            frameon=False,
        )
        # ax.set_xlabel(ax.get_label(), fontsize=10)
        fig.tight_layout()
        fig.text(
            0.5,
            0.95,
            f"{dataset}\n"
            f"yes   ={probs[0]:.3f},\nequal={probs[1]:.3f},\nno    ={probs[2]:.3f}",
            ha="left",
            va="top",
            fontsize=10,
        )
        fig.subplots_adjust(right=0.5)
        fig.show()
        fig.savefig(
            os.path.join(
                get_fig_folder(),
                f"accuracies_all_data_bayesian_test_{dataset}_{first}_vs_{second}.svg",
            ),
            transparent=True,
        )
df_bayesian = pd.DataFrame(df_bayesian)
print(df_bayesian.to_string(line_width=1000))

# %% plot as mean +- std as bars with colors
custom_label_mapping = {
    "knn_clustering": "Individual",
    "traditional": "Traditional",
    "shape_2D": "Burst shape",
    "combined": "Combined",
    "shape_manual": "On/off feature",
}

plot_bayesian = False
bayesian_thresholds = [0.0, 0.7, 0.9, 0.99][::-1]
bayesian_symbols = ["n.d.", "†", "‡", "#"][::-1]


def _get_bayesian_symbol(prob):
    for threshold, symbol in zip(bayesian_thresholds, bayesian_symbols):
        if prob > threshold:
            return symbol
    return None


normalize = False
stats = _get_stats(normalize=normalize, baseline="traditional")

color_shape = "C3"  # "#00AA00"
color_traditional = "grey"  # "#800000"
feature_sets = ["traditional", "shape_2D", "combined", "knn_clustering"]
color = [color_traditional, color_shape, color_shape, "white"]
edgecolor = [color_traditional, color_shape, color_traditional, color_shape]
hatch = ["", "", "//", ""]
linewidth = 1
alpha = 1
matplotlib.rcParams["hatch.linewidth"] = 4

dataset_order = [
    "inhibblock",
    "hommersom_binary",
    "mossink_KS",
    "wagenaar",
]  # , "kapucu"]

fig, axs = plt.subplots(
    figsize=(8 * cm, 4 * cm), constrained_layout=True, ncols=len(dataset_order)
)

sns.despine()
for i, dataset_name in enumerate(dataset_order):
    ax = axs[i]
    stats_dataset = stats.loc[dataset_name]
    ax.bar(
        feature_sets,
        stats_dataset.loc[feature_sets, "mean"],
        yerr=stats_dataset.loc[feature_sets, "sem"],
        capsize=2,
        error_kw={"elinewidth": 1},
        fill=True,
        alpha=alpha,
        color=color,
        edgecolor=edgecolor,
        hatch=hatch,
        linewidth=linewidth,
        label=[custom_label_mapping[fs] for fs in feature_sets],
        width=0.6,
        # rasterized=True,
    )

    if plot_bayesian is True:
        # add bayesian symbols
        for j, (first, second) in enumerate(
            [
                ("combined", "traditional"),
                ("knn_clustering", "traditional"),
                ("knn_clustering", "shape_2D"),
            ]
        ):
            try:
                prob_first_better = df_bayesian[
                    (df_bayesian["dataset"] == dataset_name)
                    & (df_bayesian["feature_set_1"] == first)
                    & (df_bayesian["feature_set_2"] == second)
                ]["prob_first_better"].values[0]
                symbol = _get_bayesian_symbol(prob_first_better)
                if symbol is not None:
                    x_first = feature_sets.index(first)
                    x_second = feature_sets.index(second)
                    x = (x_first + x_second) / 2
                    y = max(
                        [
                            stats_dataset.loc[fs, "mean"]
                            + stats_dataset.loc[fs, "std"]
                            + 0.01
                            for fs in [first, second]
                        ]
                    )
                    ax.plot(
                        [x_first, x_first, x_second, x_second],
                        [y, y + 0.01, y + 0.01, y],
                        color="k",
                        linewidth=1,
                    )
                    # ax.text(x, y+0.01, symbol, ha="center", va="bottom")
                    ax.text(
                        x,
                        y + 0.01,
                        f"P={prob_first_better:.2f}",
                        ha="center",
                        va="bottom",
                    )
            except IndexError:
                # no bayesian result available
                pass

for i, dataset_name in enumerate(dataset_order):
    ax = axs[i]
    if normalize is False:
        ax.set_ylim(1 / n_classes[dataset_name], None)
        if i == 3:
            ax.set_yticks([0.125, 0.25, 0.375, 0.5, 0.625])
    # ax.set_title(dataset_name)
    # ax.set_xlabel("Feature set")
    if i == 0:
        ax.set_ylabel("Balanced accuracy")
    ax.tick_params(bottom=True, labelbottom=False)


fig.show()
savefig(fig, f"accuracies_barplot", file_format=["pdf", "svg"])

fig_legend = plt.figure(constrained_layout=True, figsize=(8 * cm, 2 * cm))
# --- Build manual legend patches ---
handles = []
labels = []
for c, ec, h, fs in zip(color, edgecolor, hatch, feature_sets):
    patch = matplotlib.patches.Patch(
        facecolor=c,
        edgecolor=ec,
        hatch=h,
        linewidth=linewidth,
        alpha=alpha,
    )
    # patch.set_rasterized(True)  # rasterize the legend swatch
    handles.append(patch)
    labels.append(custom_label_mapping[fs])

leg = fig_legend.legend(
    handles,
    labels,
    loc="lower center",
    # bbox_to_anchor=(.5, 0),
    ncol=2,
    frameon=False,
)
fig_legend.show()
savefig(fig_legend, f"accuracies_barplot_legend", file_format=["pdf", "svg"])

# %% plot accuracy over number of dimensions for summary shapes
shades_for_error = False
use_subplots = False
normalize = False
baseline = "shape_manual"
df_stats = _get_stats(normalize=normalize, baseline=baseline)
if use_subplots:
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9 * cm, 5 * cm))
else:
    fig, ax = plt.subplots(figsize=[6.5 * cm, 5 * cm], constrained_layout=True)
sns.despine()
dimensions = np.arange(1, 11)

colormap = {
    "inhibblock": get_group_colors("inhibblock")["bic"],
    "hommersom_binary": get_group_colors("hommersom_binary")["CACNA1A"],
    "mossink_KS": get_group_colors("mossink_KS")["KS"],
    "wagenaar": get_group_colors("wagenaar")[2],
}

for dataset, ax in zip(
    [
        "inhibblock",
        "hommersom_binary",
        "mossink_KS",
        "wagenaar",
    ],
    axs.flatten() if use_subplots else [ax] * 4,
):
    ax.errorbar(
        dimensions,
        [df_stats.loc[(dataset, f"shape_{dim}D"), "mean"] for dim in dimensions],
        yerr=None
        if shades_for_error is True
        else [df_stats.loc[(dataset, f"shape_{dim}D"), "sem"] for dim in dimensions],
        marker="",
        markersize=3,
        # linestyle="",
        linewidth=1,
        label=custom_dataset_mapping[dataset],
        capsize=2,
        color=colormap[dataset],
    )
    if shades_for_error is True:
        ax.fill_between(
            dimensions,
            [
                df_stats.loc[(dataset, f"shape_{dim}D"), "mean"]
                - df_stats.loc[(dataset, f"shape_{dim}D"), "sem"]
                for dim in dimensions
            ],
            [
                df_stats.loc[(dataset, f"shape_{dim}D"), "mean"]
                + df_stats.loc[(dataset, f"shape_{dim}D"), "sem"]
                for dim in dimensions
            ],
            color=colormap[dataset],
            alpha=0.2,
        )
    if normalize is False:
        ax.axhline(
            df_stats.loc[(dataset, baseline), "mean"],
            ls=":",
            color=colormap[dataset],
            alpha=0.7,
        )
    ax.set_xticks([1] + np.arange(5, dimensions.max() + 1, 5).tolist())
    ax.set_xticks(dimensions, minor=True)
    ax.set_xlabel("Number of data-driven dimensions")
    ax.set_ylabel("Accuracy")
if normalize is True:
    ax.axhline(
        0, ls="--", color="gray", label=f"{custom_label_mapping[baseline]}\nlevel"
    )
else:
    ax.plot(
        [], [], ls=":", color="gray", label=f"{custom_label_mapping[baseline]}\nlevel"
    )

if use_subplots:
    ax = axs.flatten()[-1]
    # make space for legend
    fig.tight_layout(rect=[0, 0, 0.75, 1])
    leg = fig.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper right")
else:
    leg = ax.legend(frameon=False, bbox_to_anchor=(1, 0.5), loc="center left")
for label in leg.get_texts():
    if label.get_text() == "CACNA1A":
        label.set_fontstyle("italic")
if normalize is True:
    ax.set_ylabel("Accuracy improvement\nvs. " + custom_label_mapping[baseline])
else:
    ax.set_ylabel("Accuracy")
fig.show()
savefig(fig, f"accuracies_shape_dimensions", file_format=["pdf", "svg"])
