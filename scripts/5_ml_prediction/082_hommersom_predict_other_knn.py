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
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

from src.persistence import load_df_bursts, load_df_cultures, load_distance_matrix
from src.persistence.knn_clustering import save_knn_clustering_results_cv
from src.pie_chart.pie_chart import plot_df_culture_layout, prepare_df_cultures_layout
from src.plot import get_group_colors, prepare_plotting
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
special_target = False  # changes target in mossink from disease label to subject label
cv_type = (
    # "RepeatedStratifiedKFold"
    # "StratifiedShuffleSplit"
    "Special"
)
random_state = 1234567890
n_splits = 100

# parameters which clustering to plot
burst_extraction_params = "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
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

y = df_cultures["target_label"].values
train_idx = np.arange(len(df_cultures))[y != "Other"]
test_idx = np.arange(len(df_cultures))[y == "Other"]
y_test = y[test_idx]
class_labels, relative_votes, _, _ = get_burst_level_predictions_cv(
    df_cultures, df_bursts, distance_matrix_square, kth, train_idx, test_idx
)
y_pred, relative_votes_recording = get_culture_level_predictions_cv(
    df_cultures,
    df_bursts,
    relative_votes,
    class_labels,
    test_idx,
    return_relative_votes=True,
)

# %% plot
df_cultures_test = df_cultures.iloc[test_idx].copy()
df_cultures_test["relative_votes"] = pd.Series(dtype="object")
for i, index in enumerate(df_cultures_test.index):
    df_cultures_test.at[index, "predicted_label"] = y_pred[i]
    df_cultures_test.at[index, "relative_votes"] = relative_votes_recording[i]

for batch in df_cultures.index.get_level_values("batch").unique():
    df_cultures_special = df_cultures_test.reset_index()
    df_cultures_special = df_cultures_special[df_cultures_special["batch"] == batch]
    df_cultures_special["layout_column"] = df_cultures_special["well_string"].str[0]
    df_cultures_special["layout_row"] = df_cultures_special["well_string"].str[1]
    df_cultures_special.set_index(["layout_row", "layout_column"], inplace=True)

    df_cultures_special, unique_batch_culture_special = prepare_df_cultures_layout(
        df_cultures_special
    )
    fig, axs = plot_df_culture_layout(
        df_cultures=df_cultures_special,
        figsize=(2.5 * cm, 3 * cm),
        dataset=dataset,
        column_names="relative_votes",
        colors=[get_group_colors(dataset)[class_label] for class_label in class_labels],
        unique_batch_culture=unique_batch_culture_special,
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
    fig.show()
