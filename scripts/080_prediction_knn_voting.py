import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src import folders
from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures, load_distance_matrix
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

cm = prepare_plotting()
special_target = True  # for mossink: if True chooses subjects as target instead, if False chooses group


# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
)
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
    )
    kth = 150
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
    )
    kth = 85
elif "mossink" in burst_extraction_params:
    dataset = "mossink"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
    )
    kth = 85
else:
    dataset = "wagenaar"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
    )
    kth = 150
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
    case "inhibblock" | "kapucu" | "wagenaar":
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
    predicted_labels = class_labels[np.argsort(relative_votes, axis=1)[:, -i]]
    print(f"Share in {i} prediction {np.mean(predicted_labels == true_labels):.3f}")

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
    case "hommersom":
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
fig, axs = plot_df_culture_layout(
    df_cultures=df_cultures_pie_chart,
    figsize=figsize,
    dataset=dataset,
    column_names="relative_votes",
    colors=[get_group_colors(dataset)[class_label] for class_label in class_labels],
    unique_batch_culture=unique_batch_culture,
)
fig.tight_layout()
fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
fig.show()
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
