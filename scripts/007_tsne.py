import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

from src.persistence import get_labels_file, get_burst_folder

burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
clustering_params = "spectral"
labels_params = "004_clustering_labels.pkl"

# load data
with open(
    get_labels_file(
        labels_params,
        clustering_params,
        burst_extraction_params,
    ),
    "rb",
) as f:
    clustering = pickle.load(f)
df_bursts = pd.read_pickle(
    os.path.join(
        get_burst_folder(burst_extraction_params), "002_wagenaar_bursts_df.pkl"
    )
)
for n_clusters_ in clustering.n_clusters:
    df_bursts[f"cluster_{n_clusters_}"] = clustering.labels_[n_clusters_]

bursts = np.load(
    os.path.join(
        get_burst_folder(burst_extraction_params), "002_wagenaar_bursts_mat.npy"
    )
)  # n_burst x time

# %% tsne on bursts
tsne_burst = TSNE(
    n_components=2,
    perplexity=5,
    n_jobs=12,
    verbose=1,
).fit_transform(bursts)

# %% plot tsne
for n_clusters_ in [2, 3, 4, 5]:
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
