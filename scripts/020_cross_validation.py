import os
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm

from src import folders
from src.folders import get_fig_folder
from src.persistence import (
    load_burst_matrix,
    load_clustering_labels,
    load_cv_params,
    load_df_bursts,
    load_distance_matrix,
)
from src.persistence.agglomerative_clustering import get_agglomerative_labels
from src.persistence.burst_extraction import _get_burst_folder
from src.plot import prepare_plotting

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
)
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
    )
    select_n_clusters = 4
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
    select_n_clusters = 4
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
    )
    select_n_clusters = 4
elif "mossink" in burst_extraction_params:
    dataset = "mossink"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
    )
    select_n_clusters = 4
else:
    dataset = "wagenaar"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
    )
    select_n_clusters = 6
print(f"Detected dataset: {dataset}")


clustering_type = clustering_params.split("_")[0]
labels_params = "labels"  #  needed for spectral clustering if not default "labels"
cv_params = "cv"
n_clusters = np.arange(2, 20)
do_my_davies_bouldin = True  # requires large RAM because distance matrix is loaded

# load cross-validation parameters
cv_params = load_cv_params(burst_extraction_params, cv_params)
n_splits = cv_params["n_splits"]

# plotting
cm = prepare_plotting()
fig_path = folders.get_fig_folder()

# load data
burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params, cv_params=cv_params)

# concat empty columns with dtype int
column_names = [f"cluster_{n_clusters_}" for n_clusters_ in n_clusters]
if cv_params is not None:
    cv_params = load_cv_params(burst_extraction_params, cv_params)
    n_splits = cv_params["n_splits"]
    for i_split in range(n_splits):
        column_names.extend(
            [f"cluster_{n_clusters_}_cv_{i_split}" for n_clusters_ in n_clusters]
        )
df_bursts = pd.concat(
    [df_bursts, pd.DataFrame(columns=column_names, dtype=int)], axis=1
)

# load labels
match clustering_type:
    case "agglomerating":
        folder_agglomerating_clustering = os.path.join(
            _get_burst_folder(burst_extraction_params),
            clustering_params,
        )

        for i_split in tqdm(
            [
                None,
            ]
            + list(range(n_splits)),
            desc="labels to dataframe",
        ):
            cv_string = "" if i_split is None else f"_cv_{i_split}"
            idx_train = (
                slice(None)
                if i_split is None
                else df_bursts.index[df_bursts[f"cv_{i_split}_train"]]
            )
            for n_clusters_ in n_clusters:
                df_bursts.loc[
                    idx_train, f"cluster_{n_clusters_}{cv_string}"
                ] = get_agglomerative_labels(
                    n_clusters_,
                    burst_extraction_params,
                    clustering_params,
                    cv_params,
                    i_split,
                )
    case "spectral":
        for i_split in tqdm(
            [
                None,
            ]
            + list(range(n_splits)),
            desc="labels to dataframe",
        ):
            cv_string = "" if i_split is None else f"_cv_{i_split}"
            idx_train = (
                slice(None)
                if i_split is None
                else df_bursts.index[df_bursts[f"cv_{i_split}_train"]]
            )
            clustering = load_clustering_labels(
                clustering_params,
                burst_extraction_params,
                labels_params,
                cv_params,
                i_split,
            )
            for n_clusters_ in n_clusters:
                df_bursts.loc[idx_train, f"cluster_{n_clusters_}{cv_string}"] = (
                    clustering.labels_[n_clusters_] + 1
                )


###############################################################################
# reference for sklearn methods
# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
###############################################################################
# %%
def _plot_score(score, ylabel, score_abbreviation, highlight: Literal["max", "min"]):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6 * cm, 4 * cm))
    sns.despine()
    if score.ndim == 2:
        ax.plot(
            n_clusters,
            score,
            linewidth=1,
            alpha=0.5,
            color="black",
            # label=[
            #     f"{score_abbreviation} cv-split {i}" for i in range(score.shape[1])
            # ],
            label=["CV splits" if i == 0 else None for i in range(score.shape[1])],
        )
        score_mean = score.mean(axis=1)
        score_std = score.std(axis=1)
    else:
        score_mean = score
        score_std = None
    ax.errorbar(
        n_clusters,
        score_mean,
        yerr=score_std,
        fmt="-o",
        color="k",
        markersize=4,
        # label=f"{score_abbreviation}-mean +- std over splits",
        label="Mean+-std",
    )
    # highlight maximum
    match highlight:
        case "min":
            i_highlight = np.argmin(score_mean)
        case "max":
            i_highlight = np.argmax(score_mean)
    ax.scatter(
        n_clusters[i_highlight],
        score_mean[i_highlight],
        color="r",
        # label=f"{highlight}: {n_clusters[i_highlight]} clusters",
        label=f"{highlight.capitalize()}: {n_clusters[i_highlight]}",
        zorder=10,
        s=15,
    )
    scale = score_mean.max() - score_mean.min()
    ax.arrow(
        select_n_clusters,
        score_mean.max() + 0.4 * scale,
        0,
        -0.3 * scale,
        length_includes_head=True,
        color="red",
        alpha=1,
        head_width=0.5,
        head_length=0.1 * scale,
    )
    # ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_coords(-0.25, 0.4)
    fig.show()
    return fig


# %% plot legend
fig, ax = plt.subplots(figsize=(11 * cm, 3 * cm))
handles = []
labels = []
handle = ax.errorbar(
    [],
    [],
    yerr=[],
    fmt="-o",
    color="k",
    markersize=4,
)
handles.append(handle)
labels.append("Mean±std")

(handle,) = ax.plot(
    [],
    [],
    linewidth=1,
    alpha=0.5,
    color="black",
)
print(handle)
handles.append(handle)
labels.append("CV splits")

handle = ax.scatter(
    [],
    [],
    color="r",
    s=15,
)
handles.append(handle)
labels.append("Max/Min")

handle = ax.scatter(
    [],
    [],
    color="red",
    marker=r"$\downarrow$",
    s=30,
)
handles.append(handle)
labels.append("Selected")

ax.legend(
    handles=handles,
    labels=labels,
    ncol=4,
    loc="center",
    frameon=False,
    handletextpad=0.1,
    columnspacing=0.3,
)

ax.axis("off")  # Hide axes since it's just a legend
fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_cv_legend.svg"), transparent=True
)

# %% mutual information score
# this accounts for permutation of labels
# Note: adjusted mutual information score has exact same result because of large numbers
mi_score = np.zeros((len(n_clusters), n_splits))
for i_n_cluster, n_clusters_ in tqdm(
    enumerate(n_clusters),
    total=len(n_clusters),
    desc="compute mutual information score",
):
    for i in range(n_splits):
        idx_train = df_bursts.index[df_bursts[f"cv_{i}_train"]]
        mi_score[i_n_cluster, i] = metrics.adjusted_mutual_info_score(
            df_bursts.loc[idx_train, f"cluster_{n_clusters_}"],
            df_bursts.loc[idx_train, f"cluster_{n_clusters_}_cv_{i}"],
        )
# plot n_clusters vs mutual information score
fig = _plot_score(
    mi_score,
    ylabel="Mutual Information\nScore",
    score_abbreviation="MI",
    highlight="max",
)
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_cv_mutual_info.svg"), transparent=True
)

# %% adjusted rand index (ARI)
# this accounts for permutation of labels
ari_score = np.zeros((len(n_clusters), n_splits))
for i_n_cluster, n_clusters_ in tqdm(
    enumerate(n_clusters),
    total=len(n_clusters),
    desc="compute adjusted rand index",
):
    for i in range(n_splits):
        idx_train = df_bursts.index[df_bursts[f"cv_{i}_train"]]
        ari_score[i_n_cluster, i] = metrics.adjusted_rand_score(
            df_bursts.loc[idx_train, f"cluster_{n_clusters_}"],
            df_bursts.loc[idx_train, f"cluster_{n_clusters_}_cv_{i}"],
        )

# plot n_clusters vs ARI
fig = _plot_score(
    ari_score, ylabel="Adjusted Rand\nIndex", score_abbreviation="ARI", highlight="max"
)
fig.savefig(os.path.join(get_fig_folder(), f"{dataset}_cv_ARI.svg"), transparent=True)

# %% Fowlkes-Mallows scores
# this accounts for permutation of labels
fm_score = np.zeros((len(n_clusters), n_splits))
for i_n_cluster, n_clusters_ in tqdm(
    enumerate(n_clusters), total=len(n_clusters), desc="compute Fowlkes-Mallows score"
):
    for i in range(n_splits):
        idx_train = df_bursts.index[df_bursts[f"cv_{i}_train"]]
        fm_score[i_n_cluster, i] = metrics.fowlkes_mallows_score(
            df_bursts.loc[idx_train, f"cluster_{n_clusters_}"],
            df_bursts.loc[idx_train, f"cluster_{n_clusters_}_cv_{i}"],
        )

# plot n_clusters vs Fowlkes-Mallows score
fig = _plot_score(
    fm_score,
    ylabel="Fowlkes-Mallows\nScore",
    score_abbreviation="FM",
    highlight="max",
)
fig.savefig(os.path.join(get_fig_folder(), "cv_fowlkes_mallows.svg"), transparent=True)

# %% Cross-validate with Davies-Bouldin index
print("Computing Davies-Bouldin index...")
db_scores = np.zeros(len(n_clusters))
for i_n_cluster, n_clusters_ in tqdm(
    enumerate(n_clusters), total=len(n_clusters), desc="compute Davies-Bouldin score"
):
    db_scores[i_n_cluster] = davies_bouldin_score(
        burst_matrix, df_bursts[f"cluster_{n_clusters_}"]
    )

fig = _plot_score(
    db_scores, ylabel="Davies-Bouldin\nScore", score_abbreviation="DB", highlight="min"
)
fig.savefig(
    os.path.join(get_fig_folder(), f"{dataset}_cv_db_euclidian.svg"), transparent=True
)

# %% Cross-validate with self-built Davies-Bouldin index
if do_my_davies_bouldin:
    print("Computing my Davies-Bouldin index...")

    def _wasserstein_distance(a, b):
        a_cumsum = np.cumsum(a)
        b_cumsum = np.cumsum(b)
        a_cumsum /= a_cumsum[-1]
        b_cumsum /= b_cumsum[-1]
        return np.sum(np.abs(a_cumsum - b_cumsum))

    def _my_davies_bouldin_score(X, distance_matrix, labels):
        n_labels = len(np.unique(labels))
        intra_dists = np.zeros(n_labels)
        centroids = np.zeros((n_labels, X.shape[1]))
        for i in range(n_labels):
            mask = labels == i + 1
            assert np.sum(mask) > 0, f"Empty cluster {i}"
            intra_dists[i] = np.mean(distance_matrix[mask][:, mask])
            centroids[i] = X[mask].mean(axis=0)
        centroid_distances = np.zeros((n_labels, n_labels))
        for i in range(n_labels):
            for j in range(i + 1, n_labels):
                centroid_distance = _wasserstein_distance(centroids[i], centroids[j])
                centroid_distances[i, j] = centroid_distance
                centroid_distances[j, i] = centroid_distance
        centroid_distances[centroid_distances == 0] = np.inf
        combined_intra_dists = intra_dists[:, None] + intra_dists
        scores = np.max(combined_intra_dists / centroid_distances, axis=1)
        return np.mean(scores)

    distance_matrix_square = load_distance_matrix(
        burst_extraction_params, clustering_params, form="matrix"
    )
    my_db_score = np.zeros(len(n_clusters))
    for i_n_cluster, n_clusters_ in tqdm(
        enumerate(n_clusters),
        total=len(n_clusters),
        desc="compute my Davies-Bouldin score",
    ):
        my_db_score[i_n_cluster] = _my_davies_bouldin_score(
            burst_matrix, distance_matrix_square, df_bursts[f"cluster_{n_clusters_}"]
        )
    fig = _plot_score(
        my_db_score,
        ylabel="Davies-Bouldin\nScore",
        score_abbreviation="DB",
        highlight="min",
    )
    fig.savefig(
        os.path.join(get_fig_folder(), f"{dataset}_cv_db_wasserstein.svg"),
        transparent=True,
    )
