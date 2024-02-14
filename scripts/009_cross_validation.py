import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm
import itertools

from src.persistence import (
    load_clustering_labels,
    load_df_bursts,
    load_cv_params,
    load_burst_matrix,
)

# parameters which clustering to evaluate
burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
clustering_params = "spectral"
labels_params = "labels"
cv_params = "cv"

# load cross-validation parameters
cv_params = load_cv_params(burst_extraction_params, cv_params)
n_splits = cv_params["n_splits"]

# load data
burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params, cv_params=cv_params)
clustering = load_clustering_labels(
    clustering_params, burst_extraction_params, labels_params, cv_params, i_split=None
)
n_clusters = clustering.n_clusters
for n_clusters_ in n_clusters:
    df_bursts[f"cluster_{n_clusters_}"] = clustering.labels_[n_clusters_]
for i in range(n_splits):
    idx_train = df_bursts.index[df_bursts[f"cv_{i}_train"]]
    clustering = load_clustering_labels(
        clustering_params, burst_extraction_params, labels_params, cv_params, i_split=i
    )
    for n_clusters_ in n_clusters:
        df_bursts[f"cluster_{n_clusters_}_cv_{i}"] = pd.Series(dtype=int)
        df_bursts.loc[idx_train, f"cluster_{n_clusters_}_cv_{i}"] = clustering.labels_[
            n_clusters_
        ]


# labels of n_clusters is saved in "cluster_{n_clusters}"
# labels of n_clusters in cross-validation split i is saved in "cluster_{n_clusters}_cv_{i}"
# index of training data in cross-validation split i is saved in "cv_{i}_train"
# index of training data can be accessed as df_bursts.index[df_bursts[f"cv_{i}_train"]]

# %% map cross-validated labels based on centroids compared to all data
for n_clusters_ in n_clusters:
    centroid_all = np.zeros((n_clusters_, burst_matrix.shape[1]))
    for i_cluster in range(n_clusters_):
        centroid_all[i_cluster] = burst_matrix[
            df_bursts[f"cluster_{n_clusters_}"] == i_cluster
        ].mean(axis=0)
    for i in range(n_splits):
        idx_train = df_bursts.index[df_bursts[f"cv_{i}_train"]]
        centroids_split = np.zeros((n_clusters_, burst_matrix.shape[1]))
        for i_cluster in range(n_clusters_):
            centroids_split[i_cluster] = burst_matrix[
                df_bursts[f"cluster_{n_clusters_}_cv_{i}"] == i_cluster
            ].mean(axis=0)
        # compute distance of each centroid to all centroids
        distance = np.zeros((n_clusters_, n_clusters_))
        for i_cluster in range(n_clusters_):
            distance[i_cluster] = np.linalg.norm(
                centroid_all - centroids_split[i_cluster], axis=1
            )
        # map labels based on minimum distance
        mapping = distance.argmin(axis=1)
        df_bursts[f"cluster_{n_clusters_}_cv_{i}_map"] = pd.Series(dtype=int)
        df_bursts.loc[idx_train, f"cluster_{n_clusters_}_cv_{i}_map"] = df_bursts.loc[
            idx_train, f"cluster_{n_clusters_}_cv_{i}"
        ].map(dict(zip(range(n_clusters_), mapping)))


# %%
# TODO centroid distance (from Oleg)
# TODO eigenvalues of spectral clustering (change sklearn code such that it saves the eigenvalues)

###############################################################################
# reference for sklearn methods
# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
###############################################################################

# %% mutual information score
# this accounts for permutation of labels
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

# Note: adjusted mutual information score has exact same result because of large numbers

# plot n_clusters vs mutual information score
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.errorbar(
    n_clusters,
    mi_score.mean(axis=1),
    yerr=mi_score.std(axis=1),
    fmt="-o",
    label="MI-mean +- std over splits",
)
# highlight maximum
i_max = np.argmax(mi_score.mean(axis=1))
ax.scatter(
    n_clusters[i_max],
    mi_score.mean(axis=1)[i_max],
    color="r",
    label=f"Max: {n_clusters[i_max]} clusters",
    zorder=10,
)
ax.legend(frameon=False)
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Mutual information score")
fig.show()

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
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.errorbar(
    n_clusters,
    fm_score.mean(axis=1),
    yerr=fm_score.std(axis=1),
    fmt="-o",
    label="FM-mean +- std over splits",
)
# highlight maximum
i_max = np.argmax(fm_score.mean(axis=1))
ax.scatter(
    n_clusters[i_max],
    fm_score.mean(axis=1)[i_max],
    color="r",
    label=f"Max: {n_clusters[i_max]} clusters",
    zorder=10,
)
ax.legend(frameon=False)
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Fowlkes-Mallows score")
fig.show()

# %% Silhouette score (expensive calculation - 20 min)
# permutation is irrelevant
silhouette_score = np.zeros(len(n_clusters))
for i_n_cluster, n_clusters_ in tqdm(
    enumerate(n_clusters), total=len(n_clusters), desc="compute Silhouette score"
):
    silhouette_score[i_n_cluster] = metrics.silhouette_score(
        burst_matrix, df_bursts[f"cluster_{n_clusters_}"], metric="euclidean"
    )

# plot n_clusters vs Silhouette score
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.plot(n_clusters, silhouette_score, "-o", label="Silhouette score")
# highlight maximum
i_max = np.argmax(silhouette_score)
ax.scatter(
    n_clusters[i_max],
    silhouette_score[i_max],
    color="r",
    label=f"Max: {n_clusters[i_max]} clusters",
    zorder=10,
)
ax.legend(frameon=False)
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Silhouette score")
fig.show()

# TODO Calinski-Harabasz Index

# %% Davies-Bouldin Index
# permutation is irrelevant
davies_bouldin_index = np.zeros(len(n_clusters))
for i_n_cluster, n_clusters_ in tqdm(
    enumerate(n_clusters), total=len(n_clusters), desc="compute Davies-Bouldin Index"
):
    davies_bouldin_index[i_n_cluster] = metrics.davies_bouldin_score(
        burst_matrix, df_bursts[f"cluster_{n_clusters_}"]
    )

# plot n_clusters vs Davies-Bouldin Index
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.plot(n_clusters, davies_bouldin_index, "-o", label="Davies-Bouldin Index")
# highlight minimum
i_min = np.argmin(davies_bouldin_index)
ax.scatter(
    n_clusters[i_min],
    davies_bouldin_index[i_min],
    color="r",
    label=f"Min: {n_clusters[i_min]} clusters",
    zorder=10,
)
ax.legend(frameon=False)
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Davies-Bouldin Index")
fig.show()

###############################################################################
# methods from Basecamp
###############################################################################

# %% F1 score
# TODO check if accounts for permutation of labels - it doesn't
max_n_clusters = (
    7  # largest reasonable number in terms of computation time - scales with factorial
)
n_clusters_to_compute_f1 = (
    n_clusters if max_n_clusters is None else range(2, max_n_clusters + 1)
)
f1_score = np.zeros((len(n_clusters), n_splits))
for i_n_cluster, n_clusters_ in tqdm(
    enumerate(n_clusters_to_compute_f1), total=len(n_clusters), desc="compute F1 score"
):
    for i in range(n_splits):
        idx_train = df_bursts.index[df_bursts[f"cv_{i}_train"]]
        for permutation in itertools.permutations(range(n_clusters_)):
            permutation = dict(zip(range(n_clusters_), permutation))
            fs_score_permute = metrics.f1_score(
                df_bursts.loc[idx_train, f"cluster_{n_clusters_}"],
                df_bursts.loc[idx_train, f"cluster_{n_clusters_}_cv_{i}"].map(
                    permutation
                ),
                average="weighted",
            )
            if fs_score_permute > f1_score[i_n_cluster, i]:
                f1_score[i_n_cluster, i] = fs_score_permute

# plot n_clusters vs F1 score
n_plot = len(n_clusters_to_compute_f1)
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.errorbar(
    n_clusters_to_compute_f1,
    f1_score.mean(axis=1)[:n_plot],
    yerr=f1_score.std(axis=1)[:n_plot],
    fmt="-o",
    label="F1-mean +- std over splits",
)
# highlight maximum
i_max = np.argmax(f1_score.mean(axis=1))
ax.scatter(
    n_clusters[i_max],
    f1_score.mean(axis=1)[i_max],
    color="r",
    label=f"Max: {n_clusters[i_max]} clusters",
    zorder=10,
)
ax.legend(frameon=False)
ax.set_xlabel("Number of clusters")
ax.set_ylabel("F1 score")
ax.set_ylim(0.8, 1.05)
fig.show()

# %% F1 score based on mapped labels
f1_score_map = np.zeros((len(n_clusters), n_splits))
for i_n_cluster, n_clusters_ in tqdm(
    enumerate(n_clusters),
    total=len(n_clusters),
    desc="compute F1 score based on mapped labels",
):
    for i in range(n_splits):
        idx_train = df_bursts.index[df_bursts[f"cv_{i}_train"]]
        f1_score_map[i_n_cluster, i] = metrics.f1_score(
            df_bursts.loc[idx_train, f"cluster_{n_clusters_}"],
            df_bursts.loc[idx_train, f"cluster_{n_clusters_}_cv_{i}_map"],
            average="weighted",
        )

# plot n_clusters vs F1 score based on mapped labels
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.errorbar(
    n_clusters,
    f1_score_map.mean(axis=1),
    yerr=f1_score_map.std(axis=1),
    fmt="-o",
    label="F1-mean +- std over splits",
)
# highlight maximum
i_max = np.argmax(f1_score_map.mean(axis=1))
ax.scatter(
    n_clusters[i_max],
    f1_score_map.mean(axis=1)[i_max],
    color="r",
    label=f"Max: {n_clusters[i_max]} clusters",
    zorder=10,
)
ax.legend(frameon=False)
ax.set_xlabel("Number of clusters")
ax.set_ylabel("F1 score based on mapped labels")
fig.show()
