"""
KNN clustering of recordings based on KNN graph of individual bursts.

Algorithm:
1) remove bursts of recording that should be predicted from graph
2) find K nearest neighbors of each burst
3) accumulate 'votes' of these K-nearest neighbours based on their classes.
    Weight for:
    - divide by number of bursts per recording
      (Motivation: correct for a single recording with a large number of bursts having a disproportionate influence.)
    - divide by number of recordings per class
      (Motivation: correct for a single class with a large number of recordings having a disproportionate influence.)
4) accumulate votes
5) assign predicted label based on largest share of votes

Plots:
- confusion matrix of burst level prediction (often bad)
- confusion matrix of recording level prediction (often a lot better)
- overview of predictions plotted as pie chart (fraction = fraction of votes)
"""
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures, load_distance_matrix
from src.persistence.knn_clustering import save_knn_clustering_results
from src.pie_chart.pie_chart import (
    get_df_cultures_subset,
    plot_df_culture_layout,
    prepare_df_cultures_layout,
)
from src.plot import get_group_colors, prepare_plotting
from src.prediction.define_target import make_target_label
from src.prediction.knn_clustering import (
    get_burst_level_predictions,
    get_culture_level_predictions,
)
from src.settings import (
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)

cm = prepare_plotting()
special_target = False  # for mossink: if True chooses subjects as target instead, if False chooses group


# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
clustering_params = get_chosen_spectral_embedding_params(dataset)
match = re.search(r"(\d+)$", clustering_params)
if match:
    kth = int(match.group(1))
    print(f"Using kth value K={kth} for KNN graph")
else:
    raise RuntimeError(
        f"No integer found at the end of clustering_params={clustering_params}."
    )
print(f"Detected dataset: {dataset}")

df_cultures = load_df_cultures(burst_extraction_params)
df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
df_bursts = load_df_bursts(burst_extraction_params)
n_bursts = len(df_bursts)
distance_matrix_square = load_distance_matrix(
    burst_extraction_params, clustering_params, form="matrix"
)


df_cultures, df_bursts, target_label = make_target_label(
    dataset,
    df_cultures,
    df_bursts,
    special_target=special_target,
)
print("Target label", target_label)

match dataset:
    case "inhibblock" | "kapucu" | "wagenaar" | "hommersom" | "hommersom_binary":
        figsize = (6 * cm, 6 * cm)
    case "mossink":
        if special_target is True:
            figsize = (10 * cm, 10 * cm)
        else:
            figsize = (6 * cm, 6 * cm)
    case _:
        raise NotImplementedError

# %% burst-level knn clustering
(
    class_labels,
    relative_votes,
    true_labels,
    predicted_labels,
) = get_burst_level_predictions(df_cultures, df_bursts, distance_matrix_square, kth)
for i in range(1, len(class_labels) + 1):
    predicted_labels_rank = class_labels[np.argsort(relative_votes, axis=1)[:, -i]]
    print(
        f"Share in {i} prediction {np.mean(predicted_labels_rank == true_labels):.3f}"
    )

fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
matrix_confusion = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
matrix_confusion = (
    matrix_confusion.astype("float") / matrix_confusion.sum(axis=1)[:, np.newaxis]
)
sns.heatmap(
    matrix_confusion,
    annot=False,
    xticklabels=class_labels,
    yticklabels=class_labels,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
fig.show()

# %% aggregating to culture level knn-clustering
df_cultures = get_culture_level_predictions(
    df_cultures, df_bursts, relative_votes, class_labels
)
score = balanced_accuracy_score(
    df_cultures["target_label"], df_cultures["predicted_label"]
)
save_knn_clustering_results(
    burst_extraction_params,
    clustering_params,
    score,
    np.array(df_cultures["target_label"].values),
    np.array(df_cultures["predicted_label"].values),
    np.array(class_labels),
    np.vstack(df_cultures["relative_votes"]),
)
print("Saved KNN Clustering to file.")

matrix_confusion = confusion_matrix(
    df_cultures["target_label"], df_cultures["predicted_label"], labels=class_labels
)

fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
matrix_confusion = (
    matrix_confusion.astype("float") / matrix_confusion.sum(axis=1)[:, np.newaxis]
)
sns.heatmap(
    matrix_confusion,
    annot=False,
    xticklabels=class_labels,
    yticklabels=class_labels,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
fig.show()
fig.savefig(
    os.path.join(
        get_fig_folder(),
        f"{dataset}_knn_clustering_confusion_matrix.svg",
    ),
    transparent=True,
)

print(
    f'Accuracy: {np.mean(df_cultures["target_label"] == df_cultures["predicted_label"]):.2f}'
)
print(f"Balanced accuracy: {matrix_confusion.diagonal().mean():.2f}")
for i in range(1, min(len(class_labels) + 1, 5)):
    predicted_labels = class_labels[
        np.argsort(np.vstack(df_cultures["relative_votes"]), axis=1)[:, -i]
    ]
    print(
        f"Share in {i} prediction {np.mean(predicted_labels == df_cultures['target_label']):.3f}"
    )

# %% plot relative_votes as pie chart
plot_subset = True
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
        figsize = (3.5 * cm, 9 * cm)
    case "mossink":
        if plot_subset is True:
            figsize = (10 * cm, 9 * cm)
        else:
            figsize = (10 * cm, 30 * cm)
    case _:
        figsize = (15, 12)

if plot_subset is True:
    df_cultures_pie_chart = get_df_cultures_subset(df_cultures, dataset)
else:
    df_cultures_pie_chart = df_cultures.copy()
df_cultures_pie_chart, unique_batch_culture = prepare_df_cultures_layout(
    df_cultures_pie_chart
)
group_colors = (
    None
    if get_group_colors(dataset) is None
    else [get_group_colors(dataset)[class_label] for class_label in class_labels]
)
fig, axs = plot_df_culture_layout(
    df_cultures=df_cultures_pie_chart,
    figsize=figsize,
    dataset=dataset,
    column_names="relative_votes",
    colors=group_colors,
    unique_batch_culture=unique_batch_culture,
)
fig.tight_layout()
fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
fig.show()
fig.savefig(
    os.path.join(
        get_fig_folder(),
        f"{dataset}_knn_clustering_predictions.svg",
    ),
    transparent=True,
)


# %% special plate layout for inhibblock data
draw_rectangles = True
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
            df_cultures=df_cultures_special,
            figsize=(4 * cm, 3 * cm),
            dataset=dataset,
            column_names="relative_votes",
            colors=[
                get_group_colors(dataset)[class_label] for class_label in class_labels
            ],
            unique_batch_culture=unique_batch_culture_special,
        )

        fig.tight_layout()
        # fig.subplots_adjust(wspace=-0.15, hspace=-0.15)

        if draw_rectangles:
            # draw Bicuculline rectangle
            margin = 0.045
            bbox1 = axs[0, 1].get_position(fig)
            bbox2 = axs[1, -1].get_position(fig)
            x0 = bbox1.x0  # Left
            y0 = bbox2.y0 - margin  # Bottom
            width = bbox2.x1 - x0
            height = bbox1.y1 - y0
            # Create a rectangle and add it to the figure
            rect = patches.Rectangle(
                (x0, y0),
                width,
                height,
                transform=fig.transFigure,  # Figure-level transformation
                color=get_group_colors(dataset)["bic"],
                linewidth=2,
                fill=False,
            )
            fig.add_artist(rect)

            # draw Control rectangle
            margin_x = 0.07
            margin_y = 0.045
            bbox1 = axs[0, 0].get_position(fig)
            bbox2 = axs[2, 0].get_position(fig)
            bbox3 = axs[3, -1].get_position(fig)
            L_shape = [
                (bbox1.x0, bbox1.y1),  # Top-left
                (bbox1.x1 + margin_x, bbox1.y1),  # Top-right
                (bbox1.x1 + margin_x, bbox2.y1 + margin_y),  # Middle-right
                (bbox3.x1, bbox2.y1 + margin_y),  # Bottom-right
                (bbox3.x1, bbox3.y0),  # Bottom
                (bbox2.x0, bbox3.y0),  # Bottom-left
                (bbox2.x0, bbox1.y1),  # Back to Top-left
            ]
            L_patch = patches.Polygon(
                L_shape,
                transform=fig.transFigure,
                edgecolor=get_group_colors(dataset)["control"],
                fill=False,
                alpha=1,
                linewidth=2,
            )

            fig.patches.append(L_patch)

            # fig.subplots_adjust(wspace=0.0, hspace=-0.0)
        fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
        fig.show()
        fig.savefig(
            os.path.join(
                get_fig_folder(),
                f"{dataset}_knn_clustering_predictions_special_div_{div}.svg",
            ),
            transparent=True,
        )

# %% one-vs-rest comparison
if len(class_labels) > 2:
    print("One-vs-rest comparison because we have multiple classes")
    accuracies = []
    for test_class_label in tqdm(class_labels, desc="One-vs-rest comparison"):
        # redefine target label to be binary
        df_cultures[f"{test_class_label}_label"] = (
            df_cultures["target_label"] == test_class_label
        )
        df_bursts[f"{test_class_label}_label"] = (
            df_bursts["target_label"] == test_class_label
        )

        # compute burst-level prediction
        (
            class_labels_binary,
            relative_votes_binary,
            true_test_class_labels,
            predicted_labels,
        ) = get_burst_level_predictions(
            df_cultures,
            df_bursts,
            distance_matrix_square,
            kth,
            target_label=f"{test_class_label}_label",
        )
        df_cultures = get_culture_level_predictions(
            df_cultures, df_bursts, relative_votes_binary, class_labels_binary
        )
        accuracies.append(
            {
                "label": test_class_label,
                "accuracy_pos": np.mean(
                    (
                        df_cultures["predicted_label"]
                        == df_cultures[f"{test_class_label}_label"]
                    )[df_cultures[f"{test_class_label}_label"]]
                ),
                "accuracy_neg": np.mean(
                    (
                        df_cultures["predicted_label"]
                        == df_cultures[f"{test_class_label}_label"]
                    )[~df_cultures[f"{test_class_label}_label"]]
                ),
            }
        )

    # df_accuracies = pd.DataFrame.from_dict(accuracies, orient='index', columns=['accuracy'])
    df_accuracies = pd.DataFrame(accuracies)
    df_accuracies["balanced_accuracy"] = (
        df_accuracies["accuracy_pos"] + df_accuracies["accuracy_neg"]
    ) / 2
    print(df_accuracies)
    print(f"Balanced Accuracy: {df_accuracies['balanced_accuracy'].mean():.2f}")
