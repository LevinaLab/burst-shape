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
import re

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

from src.persistence import load_df_bursts, load_df_cultures, load_distance_matrix
from src.persistence.knn_clustering import (
    exist_knn_clustering_results_cv,
    load_knn_clustering_results_cv,
    save_knn_clustering_results_cv,
)
from src.plot import get_group_colors, prepare_plotting, savefig
from src.prediction.define_target import make_target_label
from src.prediction.knn_clustering import (
    get_burst_level_predictions_cv,
    get_culture_level_predictions_cv,
)
from src.settings import (
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)

cm = prepare_plotting()
special_target = True  # changes target in mossink from disease label to subject label
cv_type = (
    # "RepeatedStratifiedKFold"
    "StratifiedShuffleSplit"
)
random_state = 1234567890
n_splits = 100

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_KS"
    # "burst_dataset_mossink_MELAS"
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

if exist_knn_clustering_results_cv(
    burst_extraction_params, clustering_params, cv_type, special_target
):
    print(
        "KNN clustering results for these parameters already exist. Loading results from disk."
    )
    (
        nested_scores,
        all_y_test,
        all_y_pred,
    ) = load_knn_clustering_results_cv(
        burst_extraction_params, clustering_params, cv_type, special_target
    )
else:
    print("KNN clustering results for these parameters do not exist yet. Computing.")
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

    match cv_type:
        case "StratifiedShuffleSplit":
            outer_cv = StratifiedShuffleSplit(
                n_splits=n_splits, test_size=0.2, random_state=random_state
            )
        case "RepeatedStratifiedKFold":
            outer_cv = RepeatedStratifiedKFold(
                n_splits=5, n_repeats=n_splits // 5, random_state=random_state
            )
        case _:
            raise NotImplementedError(f"Unknown cv type: {cv_type}")

    nested_scores = []
    all_y_test = []
    all_y_pred = []

    y = df_cultures["target_label"].values
    for train_idx, test_idx in tqdm(
        outer_cv.split(np.zeros_like(y), y), total=n_splits, desc="Outer loop of cv"
    ):
        y_test = y[test_idx]
        class_labels, relative_votes, _, _ = get_burst_level_predictions_cv(
            df_cultures, df_bursts, distance_matrix_square, kth, train_idx, test_idx
        )
        y_pred = get_culture_level_predictions_cv(
            df_cultures, df_bursts, relative_votes, class_labels, test_idx
        )

        nested_scores.append(balanced_accuracy_score(y_test, y_pred))
        all_y_test.append(y_test)
        all_y_pred.append(y_pred)

    nested_scores = np.array(nested_scores)
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)

    save_knn_clustering_results_cv(
        burst_extraction_params,
        clustering_params,
        cv_type,
        nested_scores,
        all_y_test,
        all_y_pred,
    )

# %% plot confusion matrix
match dataset:
    case "inhibblock" | "kapucu" | "wagenaar":
        figsize = (7 * cm, 6 * cm)
    case "mossink":
        if special_target is True:
            figsize = (10 * cm, 8 * cm)
        else:
            figsize = (7 * cm, 6 * cm)
    case _:
        figsize = (7 * cm, 6 * cm)

fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
labels = np.unique(np.concatenate((all_y_test, all_y_pred)))
sns.heatmap(
    confusion_matrix(
        all_y_pred.flatten(), all_y_test.flatten(), normalize="true", labels=labels
    ),
    vmin=0,
    vmax=None if dataset == "mossink" and special_target == True else 1,
    annot=False,
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": "Accuracy"},
)
if get_group_colors(dataset) is not None:
    for label_x, label_y, color in zip(
        ax.get_xticklabels(),
        ax.get_yticklabels(),
        [get_group_colors(dataset)[j] for j in list(labels)],
    ):
        label_x.set_color(color)
        label_y.set_color(color)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.show()
savefig(
    fig,
    f"{dataset}_knn_clustering_confusion_matrix{'_special_target' if special_target is True else ''}",
    file_format=["pdf", "svg"],
)

# highlight isogenic pairs in mossink dataset with special target
if dataset == "mossink" and special_target == True:
    _isogenic_map = {
        ("KS", 3): ("Control", 9),
        ("KS", 4): ("Control", 10),
        ("MELAS", 1): ("Control", 2),
        ("MELAS", 2): ("Control", 4),
        ("MELAS", 3): ("Control", 5),
    }
    for (group1, subject1), (group2, subject2) in _isogenic_map.items():
        pos1 = np.where(labels == f"{group1} {subject1}")[0][0]
        pos2 = np.where(labels == f"{group2} {subject2}")[0][0]
        for x_pos, y_pos in [(pos1, pos2), (pos2, pos1)]:
            ax.add_patch(
                plt.Rectangle(
                    (x_pos, y_pos),
                    1,
                    1,
                    fill=False,
                    edgecolor="limegreen",
                    lw=1.5,
                    zorder=100,
                )
            )
    fig.show()
    savefig(
        fig,
        f"{dataset}_knn_clustering_confusion_matrix{'_special_target' if special_target is True else ''}_isogenic_highlighted",
        file_format=["pdf", "svg"],
    )
