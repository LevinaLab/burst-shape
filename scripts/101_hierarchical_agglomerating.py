"""Test hierarchical clustering with Wasserstein distance."""
import os
import multiprocessing
from time import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

from src import folders
from src.persistence import load_burst_matrix, load_df_bursts
from src.persistence.burst_extraction import _get_burst_folder

burst_extraction_params = "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
n_bursts = None  # if None uses all bursts
compute_parallel = True  # if True uses double the memory but is faster
recompute = False  # if False and available loads the data from disk
linkage_method = "complete"
np.random.seed(0)

# plot settings
n_clusters = None # 3  # if None chooses the number of clusters with Davies-Bouldin index

# plotting
cm = 1 / 2.54  # centimeters in inches
fig_path = folders.get_fig_folder()

burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
# select randomly bursts to reduce computation time
np.random.seed(0)
if n_bursts is not None:
    idx = np.random.choice(burst_matrix.shape[0], n_bursts, replace=False)
    burst_matrix = burst_matrix[idx]
    df_bursts = df_bursts.iloc[idx]
print(burst_matrix.shape)

# %%


def _wasserstein_distance(a, b):
    a_cumsum = np.cumsum(a)
    b_cumsum = np.cumsum(b)
    a_cumsum /= a_cumsum[-1]
    b_cumsum /= b_cumsum[-1]
    return np.sum(np.abs(a_cumsum - b_cumsum))


# %% cluster with linkage
folder_agglomerating_clustering = os.path.join(
    _get_burst_folder(burst_extraction_params),
    f"agglomerating_clustering_linkage_{linkage_method}_n_bursts_{n_bursts}"
)
file_distance_matrix = os.path.join(
    folder_agglomerating_clustering, "distance_matrix.npy"
)
file_linkage = os.path.join(folder_agglomerating_clustering, "linkage.npy")

if not recompute and os.path.exists(file_linkage):
    print(f"Loading linkage from disk: {file_linkage}")
    Z = np.load(file_linkage)
    distance_matrix = squareform(np.load(file_distance_matrix), force="tovector")
elif not recompute and os.path.exists(file_distance_matrix):
    print(f"Loading distance matrix from disk: {file_distance_matrix}")
    distance_matrix = squareform(np.load(file_distance_matrix), force="tovector")
    Z = linkage(distance_matrix, method=linkage_method)
    np.save(file_linkage, Z)
else:
    print("Computing distance matrix and linkage")
    t0 = time()
    if compute_parallel:
        n = burst_matrix.shape[0]
        size = n * (n - 1) // 2
        # initialize shared memory array
        distance_matrix = multiprocessing.Array("d", n * (n - 1) // 2, lock=False)
        # convert the burst matrix to shared memory with lock=False
        burst_matrix_parallel = np.frombuffer(
            multiprocessing.Array("d", burst_matrix.size, lock=False)
        ).reshape(burst_matrix.shape)
        burst_matrix_parallel[:] = burst_matrix
        # create a shared lookup table for the index of the distance matrix
        lookup_table = np.zeros((size, 2), dtype=int)
        k = 0
        for i in tqdm(range(n - 1), desc="Computing lookup table"):
            lookup_table[k : k + n - i - 1, 0] = i
            lookup_table[k : k + n - i - 1, 1] = np.arange(i + 1, n)
            k += n - i - 1

        def _metric_from_index(k):
            i, j = lookup_table[k]
            distance_matrix[k] = _wasserstein_distance(burst_matrix_parallel[i], burst_matrix_parallel[j])


        # parallel computation of the distance matrix
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pool.map(
                _metric_from_index,
                range(size),
            )
    else:
        distance_matrix = pdist(burst_matrix, metric=_wasserstein_distance)
    t1 = time()
    print(f"Distance matrix: {t1 - t0:.2f} s")
    Z = linkage(distance_matrix, method=linkage_method)
    t2 = time()
    print(f"Linkage: {t2 - t1:.2f} s")
    os.makedirs(folder_agglomerating_clustering, exist_ok=True)
    np.save(file_distance_matrix, squareform(distance_matrix, force="tomatrix"))
    np.save(file_linkage, Z)

# %% Cross-validate with Davies-Bouldin index
print("Computing Davies-Bouldin index...")
n_clusters_range = range(2, 30)
score_davies_bouldin = np.zeros(len(n_clusters_range))
for _n_clusters in range(2, 30):
    labels = fcluster(Z, t=_n_clusters, criterion="maxclust")
    score_davies_bouldin[_n_clusters - 2] = davies_bouldin_score(burst_matrix, labels)
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
sns.despine()
ax.plot(
    n_clusters_range,
    score_davies_bouldin,
    "o-",
    label="Davies-Bouldin Index",
    markersize=2,
    linewidth=1,
)
# highlight the best number of clusters
best_n_clusters = np.argmin(score_davies_bouldin) + 2
ax.plot(
    best_n_clusters,
    score_davies_bouldin[best_n_clusters - 2],
    "ro",
    markersize=3,
    label=f"Min: {best_n_clusters} clusters",
)
ax.set_xlabel("Number of Clusters")
ax.set_xticks([3, 10, 20, 30])
ax.set_yticks([])
ax.set_ylabel("Davies-\nBouldin Index")
# ax.legend(frameon=False)
# fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "davies_bouldin.svg"))
fig.savefig(os.path.join(fig_path, "davies_bouldin.pdf"))

# %% Cross-validate with self-built Davies-Bouldin index
print("Computing my Davies-Bouldin index...")
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

distance_matrix_square = squareform(distance_matrix, force="tomatrix")
n_clusters_range = range(2, 30)
score_davies_bouldin = np.zeros(len(n_clusters_range))
for _n_clusters in range(2, 30):
    labels = fcluster(Z, t=_n_clusters, criterion="maxclust")
    score_davies_bouldin[_n_clusters - 2] = _my_davies_bouldin_score(burst_matrix, distance_matrix_square, labels)
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
sns.despine()
ax.plot(
    n_clusters_range,
    score_davies_bouldin,
    "o-",
    label="Davies-Bouldin Index",
    markersize=2,
    linewidth=1,
)
# highlight the best number of clusters
best_n_clusters = np.argmin(score_davies_bouldin) + 2
ax.plot(
    best_n_clusters,
    score_davies_bouldin[best_n_clusters - 2],
    "ro",
    markersize=3,
    label=f"Min: {best_n_clusters} clusters",
)
ax.set_xlabel("Number of Clusters")
ax.set_xticks([3, 10, 20, 30])
ax.set_yticks([])
ax.set_ylabel("Davies-\nBouldin Index")
# ax.legend(frameon=False)
# fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "davies_bouldin_my.svg"))
fig.savefig(os.path.join(fig_path, "davies_bouldin_my.pdf"))

if n_clusters is None:
    print(f"Choosing the best number of clusters: {best_n_clusters} based on my Davies-Bouldin index")
    n_clusters = best_n_clusters

# %% get clusters from linkage
print("Getting clusters from linkage...")
labels = fcluster(Z, t=n_clusters, criterion="maxclust")
df_bursts["cluster"] = labels

# %% Define a color palette for the clusters
palette = sns.color_palette(n_colors=n_clusters)  # "Set1", n_clusters)
cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
cluster_colors = [
    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
    for c in cluster_colors
]
# %%
# Plot dendrogram with colored clusters
print("Plotting dendrogram...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm))
sns.despine()

color_threshold = Z[-(n_clusters - 1), 2]

dendrogram_properties = dendrogram(
    Z,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=10,
    above_threshold_color="black",
    color_threshold=color_threshold,
)

# Highlight the clusters with background colors
# Count the number of elements in each cluster
cluster_counts = np.bincount(labels)[1:]  # ignoring cluster 0
cluster_index_dendrogram = [i for i in range(n_clusters) if cluster_counts[i] > 1]
# (fix the situation where a cluster is only size 1 -> not present in ax.collections)
for i, d in zip(cluster_index_dendrogram, ax.collections[:-1]):
    if i >= n_clusters:
        break
    color = cluster_colors[i]
    d.set_color(color)
ax.set_xlabel("Samples")
ax.set_ylabel("Distance")

# remove x and y ticks
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "dendrogram.svg"))
fig.savefig(os.path.join(fig_path, "dendrogram.pdf"))

# %% PCA
print("Computing PCA...")
pca = PCA(n_components=2)
pca.fit(burst_matrix)
X_pca = pca.transform(burst_matrix)

# %% plot clusters in PCA space
print("Plotting PCA...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm))
sns.despine()
for cluster in range(1, n_clusters + 1):
    cluster_points = X_pca[labels == cluster]
    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=1,
        color=palette[cluster - 1],
        alpha=0.5,
        label=f"Cluster {cluster}",
    )
ax.set_xlabel("1st PC")
ax.set_ylabel("2nd PC")
# ax.legend(frameon=False)

ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "pca_clusters.svg"))
fig.savefig(os.path.join(fig_path, "pca_clusters.pdf"))


# %% histogram of cluster sizes
print("Plotting cluster sizes...")
fig, ax = plt.subplots()
sns.despine()
# Count the number of elements in each cluster
cluster_counts = np.bincount(labels)[1:]  # ignoring cluster 0
# Create a bar plot with the correct colors
ax.bar(range(1, n_clusters + 1), cluster_counts, color=palette)
# write numerical values on top of the bars
for i, count in enumerate(cluster_counts):
    ax.text(i + 1, count, str(count), ha="center", va="bottom")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Bursts")
ax.set_xticks(range(1, n_clusters + 1))
fig.show()
fig.savefig(os.path.join(fig_path, "cluster_sizes.svg"))
fig.savefig(os.path.join(fig_path, "cluster_sizes.pdf"))

# %% plot average burst of each cluster
print("Plotting average bursts...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm))
sns.despine()
for cluster in range(1, n_clusters + 1):
    cluster_bursts = burst_matrix[labels == cluster]
    ax.plot(
        cluster_bursts.mean(axis=0),
        color=palette[cluster - 1],
        label=f"Cluster {cluster}",
    )
ax.set_xlabel("Time [a.u.]")
ax.set_ylabel("Rate [a.u.]")
# ax.legend()
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "average_bursts.svg"))
fig.savefig(os.path.join(fig_path, "average_bursts.pdf"))


# %% build new dataframe df_cultures with index ('batch', 'culture', 'day') and columns ('n_bursts', 'cluster_abs', 'cluster_rel')
print("Building df_cultures...")
df_bursts_reset = df_bursts.reset_index(
    drop=False
)  # reset index to access columns in groupby()
df_cultures = df_bursts_reset.groupby(["batch", "culture", "day"]).agg(
    n_bursts=pd.NamedAgg(column="i_burst", aggfunc="count")
)

# for all unique combinations of batch and culture
unique_batch_culture = df_cultures.reset_index()[["batch", "culture"]].drop_duplicates()
# sort by batch and culture
unique_batch_culture.sort_values(["batch", "culture"], inplace=True)
# assign an index to each unique combination
unique_batch_culture["i_culture"] = np.arange(len(unique_batch_culture))  # [::-1]
unique_batch_culture.set_index(["batch", "culture"], inplace=True)

df_cultures["i_culture"] = pd.Series(
    data=(
        df_cultures.reset_index().apply(
            lambda x: unique_batch_culture.loc[(x["batch"], x["culture"]), "i_culture"],
            axis=1,
        )
    ).values,
    index=df_cultures.index,
    dtype=int,
)
# dd cluster information
for i_cluster in range(1, n_clusters + 1):
    col_cluster = "cluster"
    df_cultures[f"cluster_abs_{i_cluster}"] = df_bursts.groupby(
        ["batch", "culture", "day"]
    )[col_cluster].agg(lambda x: np.sum(x == i_cluster))
    df_cultures[f"cluster_rel_{i_cluster}"] = (
        df_cultures[f"cluster_abs_{i_cluster}"] / df_cultures["n_bursts"]
    )

# %% development plot
print("Plotting development...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm))
sns.despine()
days = df_bursts.index.get_level_values("day").unique().sort_values()
fraction = np.zeros((len(days), n_clusters))
for i, day in enumerate(days):
    for cluster in range(1, n_clusters + 1):
        fraction[i, cluster - 1] = np.mean(
            df_bursts[df_bursts.index.get_level_values("day") == day]["cluster"]
            == cluster
        )
# for cluster in range(1, n_clusters + 1):
# ax.plot(days, fraction[:, cluster - 1], color=palette[cluster - 1], label=f"Cluster {cluster}")
# smooth fraction by moving average over 5 days
for cluster in range(1, n_clusters + 1):
    fraction[:, cluster - 1] = np.convolve(
        fraction[:, cluster - 1], np.ones(5) / 5, mode="same"
    )
days = days[2:-2]
fraction = fraction[2:-2]
# fraction /= fraction.sum(axis=1)[:, None]
for cluster in range(1, n_clusters + 1):
    ax.plot(days, fraction[:, cluster - 1], color=palette[cluster - 1]) # , linestyle="--")
ax.set_xlabel("Day")
ax.set_ylabel("Fraction")
# ax.legend()
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "fraction_clusters.svg"))
fig.savefig(os.path.join(fig_path, "fraction_clusters.pdf"))


# %% development plot based on df_cultures
print("Plotting development based on df_cultures...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
sns.despine()
days = df_cultures.index.get_level_values("day").unique().sort_values()
fraction = np.zeros((len(days), n_clusters))
for i, day in enumerate(days):
    for cluster in range(1, n_clusters + 1):
        fraction[i, cluster - 1] = np.mean(
            df_cultures[df_cultures.index.get_level_values("day") == day][
                f"cluster_rel_{cluster}"
            ]
        )
# smooth fraction by moving average over 5 days
for cluster in range(1, n_clusters + 1):
    fraction[:, cluster - 1] = np.convolve(
        fraction[:, cluster - 1], np.ones(5) / 5, mode="same"
    )
days = days[2:-2]
fraction = fraction[2:-2]
# fraction /= fraction.sum(axis=1)[:, None]
for cluster in range(1, n_clusters + 1):
    ax.plot(
        days,
        fraction[:, cluster - 1],
        color=palette[cluster - 1],
        label=f"Cluster {cluster}",
    )
ax.set_xlabel("Day")
ax.set_ylabel("Fraction")
fig.show()
fig.savefig(os.path.join(fig_path, "fraction_clusters_df_cultures.svg"))
fig.savefig(os.path.join(fig_path, "fraction_clusters_df_cultures.pdf"))

# %% complexity plot (information)
print("Plotting information...")
# compute information based on cluster_rel columns
columns = [f"cluster_rel_{i_cluster}" for i_cluster in range(1, n_clusters + 1)]
df_cultures["information"] = df_cultures.apply(
    lambda x: -np.sum([(p * np.log2(p) if p > 0 else 0) for p in x[columns]]), axis=1
)
# plot average information per day
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
sns.despine()
days = df_cultures.index.get_level_values("day").unique().sort_values()
information = np.zeros(len(days))
for i, day in enumerate(days):
    information[i] = df_cultures[df_cultures.index.get_level_values("day") == day][
        "information"
    ].mean()
# smooth information by moving average over 5 days
information = np.convolve(information, np.ones(5) / 5, mode="valid")
days = days[2:-2]
ax.plot(days, information, color="black")
ax.set_xlabel("Day")
ax.set_ylabel("Info [bits]")
fig.show()
fig.savefig(os.path.join(fig_path, "information.svg"))
fig.savefig(os.path.join(fig_path, "information.pdf"))


# %% similarity between batches (vector product of cluster_rel)
print("Computing similarity between batches...")
def _similarity(x, y):
    return np.sum(
        [
            x[f"cluster_rel_{i_cluster}"] * y[f"cluster_rel_{i_cluster}"]
            for i_cluster in range(1, n_clusters + 1)
        ]
    )


def _similarity(x, y):
    # MSE
    return np.sum(
        [
            (x[f"cluster_rel_{i_cluster}"] - y[f"cluster_rel_{i_cluster}"]) ** 2
            for i_cluster in range(1, n_clusters + 1)
        ]
    )


def _similarity_df(df):
    # average between all combinations
    if len(df) <= 1:
        return np.array([np.nan])
    similarities = np.zeros((len(df), len(df)))
    for i, x in enumerate(df.iterrows()):
        for j, y in enumerate(df.iterrows()):
            similarities[i, j] = _similarity(x[1], y[1])
    # average over upper triangle
    similarities = similarities[np.triu_indices(len(df), k=1)].flatten()
    return similarities


similarity_random = _similarity_df(df_cultures.sample(frac=1))
similarity_batch_random = np.array(
    [
        _similarity_df(
            df_cultures[df_cultures.index.get_level_values("batch") == batch]
        ).mean()
        for batch in df_cultures.index.get_level_values("batch").unique()
    ]
).flatten()
similarity_same_day = np.array(
    [
        _similarity_df(
            df_cultures[df_cultures.index.get_level_values("day") == day]
        ).mean()
        for day in df_cultures.index.get_level_values("day").unique()
    ]
).flatten()
similarity_batch_day = np.array(
    [
        _similarity_df(
            df_cultures[
                (df_cultures.index.get_level_values("day") == day)
                & (df_cultures.index.get_level_values("batch") == batch)
            ]
        ).mean()
        for day in df_cultures.index.get_level_values("day").unique()
        for batch in df_cultures.index.get_level_values("batch").unique()
    ]
).flatten()

fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
sns.despine()
sns.violinplot(
    data=[
        similarity_random,
        similarity_batch_random,
        similarity_same_day,
        similarity_batch_day,
    ],
    ax=ax,
)
ax.set_ylabel("Similarity")
fig.show()


# %% plot a pie chart for each entry in df_cultures
# position of the pie chart in the grid is determined by the day and i_culture
print("Plotting pie charts...")
colors = palette #  sns.color_palette("Set1", n_colors=n_clusters)
nrows = df_cultures["i_culture"].max() + 1
ncols = df_cultures.index.get_level_values("day").max() + 1
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)
# set all axes to invisible
for ax in axs.flatten():
    ax.axis("off")
for index in df_cultures.index:
    i_day = index[2]
    i_culture = df_cultures.loc[index, "i_culture"]
    ax = axs[i_culture, i_day]
    ax.axis("on")
    # ax.set_title(f"Day {i_day} - Culture {i_culture}")
    cluster_rel = [
        df_cultures.loc[index, f"cluster_rel_{i_cluster}"]
        for i_cluster in range(1, n_clusters + 1)
    ]
    ax.pie(cluster_rel, colors=colors, startangle=90)

# Add a shared x-axis for days
for i_day in np.arange(ncols)[::2]:
    ax = axs[-1, i_day]
    ax.axis("on")
    ax.set_xlabel(f"Day {i_day}", fontsize=12)
    ax.xaxis.set_label_position("bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

# write batches on the left
batch_label_pos = np.linspace(0.04, 0.97, nrows, endpoint=True)[::-1]
for (batch, culture), row in unique_batch_culture.iterrows():
    i_culture = row["i_culture"]
    fig.text(
        0.05, batch_label_pos[i_culture], fontsize=12, va="center", s=f"Batch {batch}"
    )

fig.legend(
    [f"Cluster {i_cluster}" for i_cluster in range(1, n_clusters + 1)],
    loc="center right",
    frameon=False,
)
fig.tight_layout()
fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
fig.show()
fig.savefig(os.path.join(fig_path, "pie_charts.svg"))
fig.savefig(os.path.join(fig_path, "pie_charts.pdf"))

# %% stats of each cluster
print("Computing stats of each cluster...")
for stat in [
    "time_orig",
    # "time_extend",
    # "peak_height",
    # "integral",
]:
    fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
    # fig.suptitle(f"Box plot of {stat}")
    sns.despine()
    sns.violinplot(
        x="cluster",
        y=stat,
        data=df_bursts,
        ax=ax,
        palette="Set1",
        log_scale=True,
        legend=False,
        hue="cluster",
        linewidth=0.5,
        inner=None,  # "quart",
    )

    if stat == "time_orig":
        ax.set_ylabel("Duration [ms]")
        ax.set_xlabel("Cluster")
        fig.savefig(os.path.join(fig_path, "time_orig.svg"))
        fig.savefig(os.path.join(fig_path, "time_orig.pdf"))

    fig.show()

# %% example bursts
print("Plotting example bursts...")
np.random.seed(0)
fig, axs = plt.subplots(nrows=3, figsize=(4.6 * cm, 7 * cm), constrained_layout=True)
sns.despine()
for ax, i_cluster in zip(axs, range(1, n_clusters + 1)):
    df_bursts_i = df_bursts[df_bursts["cluster"] == i_cluster]
    n_bursts_i = df_bursts_i.shape[0]
    idx_random = np.random.randint(0, n_bursts_i, 5)
    for idx in idx_random:
        bins = np.linspace(
            -50, df_bursts_i.iloc[idx]["time_extend"] - 50, 51, endpoint=True
        )
        bins_mid = (bins[1:] + bins[:-1]) / 2
        ax.plot(
            bins_mid,
            df_bursts_i.iloc[idx]["burst"],
            color=palette[i_cluster - 1],
            alpha=1,
            linewidth=1,
        )
axs[-1].set_ylabel("Rate [Hz]")
axs[-1].set_xlabel("Time [ms]")
fig.show()
fig.savefig(os.path.join(fig_path, "example_bursts.svg"))
fig.savefig(os.path.join(fig_path, "example_bursts.pdf"))


print("Finished.")
