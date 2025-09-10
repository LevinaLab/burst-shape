import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from src.persistence import load_burst_matrix, load_df_bursts, load_df_cultures
from src.plot import get_group_colors, prepare_plotting, savefig
from src.prediction.define_target import make_target_label
from src.prediction.knn_clustering import get_recording_mask
from src.settings import get_dataset_from_burst_extraction_params

cm = prepare_plotting()

special_target = True

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

# %%
df_cultures = load_df_cultures(burst_extraction_params)
df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
df_bursts = load_df_bursts(burst_extraction_params, cv_params=None)
burst_matrix = load_burst_matrix(burst_extraction_params)

#
df_cultures["average_burst"] = pd.Series(dtype="object")
for index in df_cultures.index:
    mask_recording = get_recording_mask(df_bursts, index)
    df_cultures.at[index, "average_burst"] = burst_matrix[mask_recording].mean(axis=0)
burst_matrix_cultures = np.vstack(df_cultures["average_burst"])


#
def _wasserstein_distance(a, b):
    a_cumsum = np.cumsum(a)
    b_cumsum = np.cumsum(b)
    a_cumsum /= a_cumsum[-1]
    b_cumsum /= b_cumsum[-1]
    return np.sum(np.abs(a_cumsum - b_cumsum))


distance_matrix = squareform(pdist(burst_matrix_cultures, metric=_wasserstein_distance))
df_distance_matrix = pd.DataFrame(
    distance_matrix,
    index=df_cultures.index,
    columns=df_cultures.index,
)

df_cultures, target_label = make_target_label(
    dataset, df_cultures, special_target=special_target
)
labels = df_cultures.loc[df_cultures.index, "target_label"]
df_label_distance = (
    df_distance_matrix.groupby(labels, axis=0)  # group rows by target_label
    .mean()
    .groupby(labels, axis=1)  # group columns by target_label
    .mean()
)

# %%
match dataset:
    case "inhibblock" | "kapucu" | "wagenaar":
        figsize = (7 * cm, 6 * cm)
    case "mossink":
        if special_target is True:
            figsize = (10 * cm, 10 * cm)
        else:
            figsize = (7 * cm, 6 * cm)
    case _:
        figsize = (7 * cm, 6 * cm)

fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
sns.heatmap(
    df_label_distance,
    # vmin=2,
    # vmax=4,
    annot=False,
    xticklabels=df_label_distance.columns,
    yticklabels=df_label_distance.index,
    cbar_kws={"label": "Wasserstein distance"},
)
if get_group_colors(dataset) is not None:
    for label_x, label_y, color in zip(
        ax.get_xticklabels(),
        ax.get_yticklabels(),
        [get_group_colors(dataset)[j] for j in df_label_distance.columns],
    ):
        label_x.set_color(color)
        label_y.set_color(color)
ax.set_xlabel("")
ax.set_ylabel("")
fig.show()
savefig(fig, f"{dataset}_subject_distances_{metric}", file_format=["pdf", "svg"])
