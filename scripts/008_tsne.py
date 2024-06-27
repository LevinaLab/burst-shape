import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from src.persistence import load_burst_matrix, load_clustering_labels, load_df_bursts

burst_extraction_params = "burst_n_bins_50_normalization_integral"
clustering_params = "spectral_affinity_precomputed_metric_wasserstein"
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

# %% tsne on bursts
tsne_burst = TSNE(
    n_components=2,
    perplexity=50,
    n_jobs=12,
    verbose=1,
).fit_transform(bursts)

# %% plot tsne
for n_clusters_ in [8]:  # [2, 3, 4, 5]:
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.despine()
    sns.scatterplot(
        x=tsne_burst[:, 0],
        y=tsne_burst[:, 1],
        hue=df_bursts[f"cluster_{n_clusters_}"],
        palette="Set1",
        ax=ax,
        s=3,
    )
    ax.legend(fontsize=8, markerscale=4, title="Cluster", frameon=False)
    fig.show()
