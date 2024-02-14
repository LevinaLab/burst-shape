import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from src.persistence import load_df_bursts, load_clustering_labels, load_burst_matrix

burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
clustering_params = "spectral"
labels_params = "labels"

# load data
clustering = load_clustering_labels(
    clustering_params,
    burst_extraction_params,
    labels_params,
    params_cross_validation=None,
)
df_bursts = load_df_bursts(burst_extraction_params, cv_params=None)
for n_clusters_ in clustering.n_clusters:
    df_bursts[f"cluster_{n_clusters_}"] = clustering.labels_[n_clusters_]

bursts = load_burst_matrix(burst_extraction_params)
# %% pca on bursts
pca_burst = PCA(n_components=None).fit(bursts)

# %% plot explained variance
fig, ax = plt.subplots(figsize=(10, 5))
sns.despine()
sns.lineplot(
    x=np.arange(len(pca_burst.explained_variance_ratio_)),
    y=pca_burst.explained_variance_ratio_,
    ax=ax,
    label="Explained variance",
)
for threshold in [0.95, 0.99]:
    explained_variance_ratio_cumsum = np.cumsum(pca_burst.explained_variance_ratio_)
    x_threhold = np.where(explained_variance_ratio_cumsum > threshold)[0][0]
    ax.axvline(
        x=x_threhold,
        color="k",
        linestyle="--",
        label=f"{threshold*100}% explained variance ({x_threhold} PCs)",
    )
ax.legend()
ax.set_xlabel("PC")
ax.set_ylabel("Explained variance")
fig.show()

# %% plot first PCs
n_pcs = 4
fig, ax = plt.subplots(figsize=(10, 5))
sns.despine()
ax.plot(pca_burst.components_[:n_pcs, :].T, label=[f"PC {i + 1}" for i in range(n_pcs)])
ax.legend()
ax.set_xlabel("Time [arbitrary units]")
ax.set_ylabel("Firing rate a.u.")
fig.show()

# %% plot data points in first two PCs
for pcs in [[0, 1], [0, 2], [1, 2]]:
    for n_clusters in [8]:  # [2, 3, 4, 5]:
        col_cluster = f"cluster_{n_clusters}"

        burst_transformed = pca_burst.transform(bursts)
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.despine()
        sns.scatterplot(
            x=burst_transformed[:, pcs[0]],
            y=burst_transformed[:, pcs[1]],
            hue=df_bursts[col_cluster],
            palette="Set1",
            ax=ax,
            # smaller sizes
            s=4,
        )
        ax.legend(fontsize=8, markerscale=4, title="Cluster", frameon=False)
        ax.set_xlabel(f"PC {pcs[0] + 1}")
        ax.set_ylabel(f"PC {pcs[1] + 1}")
        fig.show()
