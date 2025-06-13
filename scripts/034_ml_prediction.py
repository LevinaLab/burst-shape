import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

from src import folders
from src.folders import get_fig_folder
from src.persistence import (
    load_burst_matrix,
    load_clustering_labels,
    load_df_bursts,
    load_df_cultures,
)
from src.plot import get_cluster_colors, get_group_colors, prepare_plotting
from src.settings import get_dataset_from_burst_extraction_params
from src.utils.classical_features import get_classical_features

cm = prepare_plotting()

# Choose whether to predict from clusters or classical features
data_columns_to_load = [
    "clusters",  # burst shape cluster info
    "classical",  # compute classical features from spike info
    "combo",  # combine the information
]
plot_legend = True
special_target = True  # for mossink: if True chooses subjects as target instead, if False chooses group

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
match dataset:
    case "kapucu":
        n_clusters = 4
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
        )
    case "hommersom_test":
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_6"
        )
        n_clusters = 4
    case "inhibblock":
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
        )
        n_clusters = 4
    case "mossink":
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
        )
        n_clusters = 4
    case "wagenaar":
        clustering_params = (
            "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
        )
        n_clusters = 6
    case _:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
print(f"Detected dataset: {dataset}")

# which clustering to plot
col_cluster = f"cluster_{n_clusters}"

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
burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
np.random.seed(0)

match dataset:
    case "kapucu":
        index_names = ["culture_type", "mea_number", "well_id", "DIV"]
    case "wagenaar":
        index_names = ["batch", "culture", "day"]
    case "hommersom_test":
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

# %% classical feature information
if "classical" in data_columns_to_load or "combo" in data_columns_to_load:
    df_cultures_all_data = load_df_cultures(burst_extraction_params)
    df_cultures_all_data = df_cultures_all_data[df_cultures_all_data["n_bursts"] > 0]
    df_cultures_all_data, classical_features = get_classical_features(
        df_cultures_all_data, df_bursts, dataset
    )

    df_cultures_all_data = df_cultures_all_data[classical_features]
    df_cultures.drop(classical_features, axis=1, inplace=True, errors="ignore")
    df_cultures = df_cultures.join(df_cultures_all_data)
    # df_cultures[classical_features] = df_cultures_all_data[classical_features]
    del df_cultures_all_data


# %% PCA
def _compute_PCA(select_data_column):
    global df_cultures

    # select feature columns
    match select_data_column:
        case "clusters":
            feature_columns = [
                f"cluster_rel_{i_cluster}" for i_cluster in range(1, n_clusters + 1)
            ]
        case "classical":
            feature_columns = classical_features
        case "combo":
            feature_columns = [
                f"cluster_rel_{i_cluster}" for i_cluster in range(1, n_clusters + 1)
            ] + classical_features
        case _:
            raise NotImplementedError(
                f"select_data_column {select_data_column} not implemented."
            )

    # normalize data
    if select_data_column in ["classical", "combo"]:
        scaler = StandardScaler()
        data = scaler.fit_transform(df_cultures[feature_columns])
    else:
        data = df_cultures[feature_columns].values

    n_features = len(feature_columns)
    pc_column_names = [f"PC{i}_{select_data_column}" for i in range(1, n_features + 1)]

    pca = PCA(n_components=n_features)  # Reduce to 2D for visualization
    principal_components = pca.fit_transform(data)

    # Create a DataFrame for the PCA results
    df_pca = pd.DataFrame(
        principal_components, columns=pc_column_names, index=df_cultures.index
    )
    df_cultures.drop(pc_column_names, axis=1, inplace=True, errors="ignore")
    df_cultures = df_cultures.join(df_pca)
    return pca, pc_column_names, feature_columns


def _plot_PCA(pca, pc_column_names, select_data_column, feature_columns=None):
    s = None
    match dataset:
        case "inhibblock":
            hue = "drug_label"
            s = 15
        case "kapucu":
            hue = [
                (culture_type, mea_number)
                for culture_type, mea_number in zip(
                    df_cultures.index.get_level_values("culture_type").astype(str),
                    df_cultures.index.get_level_values("mea_number").astype(str),
                )
            ]
            s = 8
        case "wagenaar":
            hue = "batch"
            s = 8
        case "mossink":
            hue = "group"
            s = 10
        case _:
            raise NotImplementedError
    # Plot the PCA components
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3.5 * cm, 3.5 * cm))
    sns.despine()
    sns.scatterplot(
        data=df_cultures.reset_index(),
        x=pc_column_names[0],
        y=pc_column_names[1],
        alpha=0.7,
        hue=hue,
        palette=get_group_colors(dataset),
        legend=False,
        s=s,
    )
    # Plot feature directions
    match select_data_column:
        case "classical":
            feature_vectors = pca.components_.T  # Get eigenvectors
            scaling_factor = np.max(
                np.abs(df_cultures[pc_column_names[:2]])
            )  # Scale for visibility
            for i, feature in enumerate(feature_columns):
                plt.arrow(
                    0,
                    0,
                    feature_vectors[i, 0] * scaling_factor,
                    feature_vectors[i, 1] * scaling_factor,
                    color="k",
                    alpha=0.5,
                    head_width=0.05,
                    head_length=0.1,
                )
                plt.text(
                    feature_vectors[i, 0] * scaling_factor * 1.15,
                    feature_vectors[i, 1] * scaling_factor * 1.15,
                    feature,
                    color="k",
                    fontsize=9,
                    ha="center",
                    va="center",
                )
        case "clusters":
            feature_vectors = pca.components_.T  # Get eigenvectors
            scaling_factor = np.max(
                np.abs(df_cultures[pc_column_names[:2]])
            )  # Scale for visibility
            for i, feature in enumerate(feature_columns):
                color = get_cluster_colors(n_clusters)[i]
                plt.arrow(
                    0,
                    0,
                    feature_vectors[i, 0] * scaling_factor,
                    feature_vectors[i, 1] * scaling_factor,
                    color=color,
                    alpha=1,
                    head_width=0.15,
                    head_length=0.1,
                )

    # ax.scatter(df_pca["PC1"], df_pca["PC2"], )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_xticks([])
    ax.set_yticks([])
    if dataset == "inhibblock" and select_data_column in ["classical"]:
        ax.set_xlim(ax.get_xlim()[0] - 1.3, ax.get_xlim()[1])
        ax.set_ylim(ax.get_ylim()[0] - 0.6, ax.get_ylim()[1])

    fig.show()
    fig.savefig(
        os.path.join(get_fig_folder(), f"{dataset}_{select_data_column}_PCA_group.svg"),
        transparent=True,
    )


if "clusters" in data_columns_to_load:
    pca_clusters, pc_column_names_clusters, feature_columns_clusters = _compute_PCA(
        "clusters"
    )
    _plot_PCA(
        pca_clusters, pc_column_names_clusters, "clusters", feature_columns_clusters
    )
if "classical" in data_columns_to_load:
    pca_classical, pc_column_names_classical, feature_columns_classical = _compute_PCA(
        "classical"
    )
    _plot_PCA(
        pca_classical, pc_column_names_classical, "classical", feature_columns_classical
    )
if "combo" in data_columns_to_load:
    pca_combo, pc_column_names_combo, feature_columns_combo = _compute_PCA("combo")
    _plot_PCA(pca_combo, pc_column_names_combo, "combo")


# %% Classification
test_type = ["cross-validate", "direct"][0]

match dataset:
    case "inhibblock":
        target_label = "drug_label"
        if plot_legend:
            figsize = (4.5 * cm, 3.5 * cm)
        else:
            figsize = (4 * cm, 3.5 * cm)
    case "kapucu":
        target_label = "culture_type"
        figsize = (4 * cm, 3.5 * cm)
    case "wagenaar":
        target_label = "batch"
        figsize = (4 * cm, 3.5 * cm)
    case "mossink":
        if special_target is True:
            df_cultures.reset_index(inplace=True)
            df_cultures["group-subject"] = (
                df_cultures["group"] + " " + df_cultures["subject_id"].astype(str)
            )
            df_cultures.set_index(["group-subject", "well_idx"], inplace=True)
            target_label = "group-subject"
        else:
            target_label = "group"

        # target_label = "group"
        figsize = (4 * cm, 3.5 * cm)
    case _:
        raise NotImplementedError


def _logistic_regression(train_columns):
    global test_type
    print(test_type)
    X = df_cultures[train_columns].values
    y = df_cultures.index.get_level_values(target_label)

    # Check number of unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    if n_classes == 2:
        # Binary case: use LabelBinarizer to maintain old behavior
        lb = LabelBinarizer()
        y = lb.fit_transform(y).ravel()  # Ensures y is 1D

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        match test_type:
            case "cross-validate":
                # Initialize Stratified K-Fold
                cv = StratifiedShuffleSplit(
                    n_splits=100, train_size=0.8, test_size=0.2, random_state=42
                )

                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    clf = LogisticRegression(
                        solver="liblinear", class_weight="balanced"
                    )
                    # TODO: multiclass='ovr'
                    clf.fit(X_train, y_train)
                    y_probs = clf.predict_proba(X_test)[:, 1]

                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    aucs.append(auc(fpr, tpr))

                mean_tpr = np.mean(tprs, axis=0)
                std_tpr = np.std(tprs, axis=0)
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                # return mean_fpr, mean_tpr, std_tpr
            case "direct":
                clf = LogisticRegression(solver="liblinear", class_weight="balanced")
                clf.fit(X, y)
                y_probs = clf.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_probs)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs.append(auc(fpr, tpr))

                mean_tpr = np.mean(tprs, axis=0)
                std_tpr = np.std(tprs, axis=0)
                # return mean_fpr, mean_tpr, std_tpr

    else:
        # Multi-class case: use LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)  # Ensures y is 1D integer labels

        # Initialize Stratified K-Fold
        cv = StratifiedShuffleSplit(
            n_splits=100, train_size=0.8, test_size=0.2, random_state=42
        )
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = LogisticRegression(solver="lbfgs", multi_class="ovr")
            clf.fit(X_train, y_train)
            y_probs = clf.predict_proba(X_test)

            # Compute ROC curves for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_probs[:, i])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        # return mean_fpr, mean_tpr, std_tpr

    # Compute Balanced Accuracy for each threshold
    balanced_accuracy = (mean_tpr + (1 - mean_fpr)) / 2
    best_index = balanced_accuracy.argmax()
    # best_threshold = thresholds[best_index]
    best_balanced_accuracy = balanced_accuracy[best_index]
    return (
        mean_fpr,
        mean_tpr,
        std_tpr,
        (best_balanced_accuracy, mean_fpr[best_index], mean_tpr[best_index]),
    )


def _plot_ROC(select_data_column, pc_column_names):
    # feature_columns = locals()[f"feature_columns_{select_data_column}"]
    # Plot ROC curve
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    sns.despine(fig)
    table_data = {}
    for train_column, label, color, linestyle in zip(
        # [feature_columns, ["PC1", "PC2"], ["PC1"]],
        [pc_column_names, pc_column_names[:2], pc_column_names[:1]],
        ["all", "2 PCs", "1 PC"],
        ["k", "k", "k"],  # ["C0", "C1", "C2"],
        ["-", "--", ":"],
    ):
        (
            mean_fpr,
            mean_tpr,
            std_tpr,
            (balanced_accuracy, fpr_best, tpr_best),
        ) = _logistic_regression(train_column)
        table_data[label] = f"{balanced_accuracy*100:.2f}"
        print(f"{label}:\t{balanced_accuracy:.4f}\t{fpr_best:.2f}\t{tpr_best:.2f}")
        ax.plot(
            mean_fpr, mean_tpr, color=color, lw=2, label=label, linestyle=linestyle
        )  # \n(area = {mean_auc:.2f} ± {std_auc:.2f})')
        if test_type == "cross-validate":
            ax.fill_between(
                mean_fpr,
                mean_tpr - std_tpr,
                mean_tpr + std_tpr,
                color=color,
                alpha=0.2,
                # label="±1 std. dev.",
            )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random classifier
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FPR", labelpad=-10)  # ("False Positive Rate")
    ax.set_ylabel("TPR", labelpad=-10)  # ("True Positive Rate")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    if plot_legend:
        ax.legend(
            loc="lower right",
            frameon=False,
            title="ROC from",
            bbox_to_anchor=(1.3, -0.1),
        )
    fig.show()
    fig.savefig(
        os.path.join(
            get_fig_folder(), f"{dataset}_{select_data_column}_ROC_{test_type}.svg"
        ),
        transparent=True,
    )

    fig_table, ax_table = plt.subplots(figsize=figsize)
    ax_table.set_axis_off()
    table = ax_table.table(
        cellText=list(table_data.items()),
        colLabels=["Feature", "Acc. [%]"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    # table.set_fontsize(12)
    table.scale(1.1, 1)  # Adjust scaling for better visibility
    fig_table.show()
    fig_table.savefig(
        os.path.join(
            get_fig_folder(), f"{dataset}_{select_data_column}_table_{test_type}.svg"
        ),
        transparent=True,
    )


if "clusters" in data_columns_to_load:
    _plot_ROC("clusters", pc_column_names_clusters)
if "classical" in data_columns_to_load:
    _plot_ROC("classical", pc_column_names_classical)
if "combo" in data_columns_to_load:
    _plot_ROC("combo", pc_column_names_combo)

# %% Accuracy by feature
test_type = ["cross-validate", "direct"][0]
plot_type = [
    "stripplot",
    "pointplot",
]

match dataset:
    case "inhibblock":
        target_label = "drug_label"
        if plot_legend:
            figsize = (10 * cm, 5 * cm)
        else:
            figsize = (10 * cm, 5 * cm)
    case "kapucu":
        target_label = "culture_type"
        figsize = (10 * cm, 5 * cm)
    case "wagenaar":
        target_label = "batch"
        figsize = (10 * cm, 5 * cm)
    case "mossink":
        if special_target is True:
            target_label = "group-subject"
        else:
            target_label = "group"
        # target_label = "group"
        figsize = (10 * cm, 5 * cm)
    case _:
        raise NotImplementedError

features_list = (
    pc_column_names_combo[:1]
    + pc_column_names_clusters[:1]
    + pc_column_names_classical[:1]
    + classical_features
)
accuracies = {}
for feature in features_list:
    (
        mean_fpr,
        mean_tpr,
        std_tpr,
        (balanced_accuracy, fpr_best, tpr_best),
    ) = _logistic_regression([feature])
    accuracies[feature] = balanced_accuracy
    print(f"{feature}:\t{balanced_accuracy}")
accuracies = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))
print(accuracies)


def _map_label(feature, newline=True):
    if feature[:2] == "PC":
        parts = feature.split("_")
        label = parts[0]
        if newline:
            label += "\n"
        else:
            label += " "
        if parts[1] == "clusters":
            label += "Shape"
        elif parts[1] == "classical":
            label += "Stand."
        elif parts[1] == "combo":
            label += "Combo"
        else:
            label += parts[1]
        return label
    else:
        return feature


fig, axs = plt.subplots(
    ncols=len(features_list),
    figsize=figsize,  # constrained_layout=True
)
sns.despine()

palette = get_group_colors(dataset)
if dataset == "kapucu":
    palette["Rat"] = palette[("Rat", "MEA1")]
    palette["hPSC"] = palette[("hPSC", "MEA1")]

for i, (feature, accuracy) in enumerate(accuracies.items()):
    ax = axs[i] if len(features_list) > 1 else axs  # Handle single subplot case
    if "stripplot" in plot_type:
        sns.stripplot(
            data=df_cultures,
            hue=target_label,
            x=target_label,
            y=feature,
            ax=ax,
            legend=False,
            size=2,
            palette=palette,
        )
    if "pointplot" in plot_type:
        sns.pointplot(
            data=df_cultures,
            x=target_label,
            hue=target_label,
            y=feature,
            ax=ax,
            errorbar="se",
            # capsize=0.2,
            # markers="x",
            palette=palette,
            dodge=True,
            legend=False,
        )
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xticklabels([])
    # ax.set_xlabel(f"{feature[:3]}\n{feature[4:]}\n{accuracy*100:.1f}%")
    ax.set_title(_map_label(feature) + f"\n{accuracy*100:.1f}%", fontsize=10)
    ax.set_xlabel("")

axs[0].set_ylabel("Feature\nValue")
fig.text(0.5, 0.02, target_label.capitalize(), ha="center", fontsize=10)

fig.tight_layout()
fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_accuracies1_{test_type}.svg"),
    transparent=True,
)

# %%
fig, ax = plt.subplots(figsize=(8 * cm, 5 * cm), constrained_layout=True)
sns.despine()
sns.barplot(
    x=accuracies.keys(),
    y=accuracies.values(),
    fill=False,
    color="k",
    width=0.6,
)
ax.bar_label(ax.containers[0], fontsize=10, fmt="%.2f")
ax.set_xticks(range(len(accuracies)))
ax.set_xticklabels(
    [_map_label(feature, newline=True) for feature in accuracies.keys()], rotation=90
)
ax.set_xlabel("Feature")
ax.set_ylim([0.5, None])
ax.set_ylabel("Accuracy")
fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_accuracies2_{test_type}.svg"),
    transparent=True,
)
