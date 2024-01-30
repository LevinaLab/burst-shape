import pickle
import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src.folders import get_results_folder

# which clustering to plot
n_clusters = 5
col_cluster = f"cluster_{n_clusters}"

# load data
with open(
    os.path.join(
        get_results_folder(),
        "004_spectral_clustering",
        f"004_clustering_labels.pkl",
    ),
    "rb",
) as f:
    clustering = pickle.load(f)
df_bursts = pd.read_pickle(
    os.path.join(get_results_folder(), "002_wagenaar_bursts_df.pkl")
)
for n_clusters_ in clustering.n_clusters:
    df_bursts[f"cluster_{n_clusters_}"] = clustering.labels_[n_clusters_]

# convert each category of multi-index to integer
for i, index_name in enumerate(df_bursts.index.names):
    df_bursts.index = df_bursts.index.set_levels(
        df_bursts.index.levels[i].astype(int), level=i
    )

# %% plot clusters
# bar plot of cluster sizes
fig, ax = plt.subplots()
sns.despine()
sns.countplot(x=col_cluster, hue=col_cluster, data=df_bursts, ax=ax, palette="Set1")
fig.show()

for category in ["batch", "day"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.despine()
    sns.countplot(
        x=category,
        hue=col_cluster,
        data=df_bursts,
        ax=ax,
        native_scale=True,
        palette="Set1",
    )
    fig.show()
# %%
facet_grid = sns.catplot(
    kind="count",
    col="batch",
    x="culture",
    hue=col_cluster,
    data=df_bursts,
    palette="Set1",
    height=5,
    aspect=0.4,
    sharey=False,
)
facet_grid.set_titles("batch {col_name}")
facet_grid.fig.show()


# %% plot bursts
n_plot = 100
fig, ax = plt.subplots()
fig.suptitle("Example bursts")
sns.despine()
for i in range(n_clusters):
    df_bursts_i = df_bursts[df_bursts[col_cluster] == i]
    n_bursts_i = df_bursts_i.shape[0]
    idx_random = np.random.randint(
        0,
        n_bursts_i,
        n_plot,
    )
    for j, idx in enumerate(idx_random):
        ax.plot(
            df_bursts_i.iloc[idx]["burst"],
            # color from set1 palette
            color=sns.color_palette("Set1")[i],
            alpha=0.5,
            label=f"Cluster {i}" if j == 0 else None,
        )
ax.legend()
fig.show()

# %% plot average burst per cluster
fig, ax = plt.subplots()
fig.suptitle("Average burst per cluster")
sns.despine()
for i in range(n_clusters):
    df_bursts_i = df_bursts[df_bursts[col_cluster] == i]
    ax.plot(
        df_bursts_i["burst"].mean(),
        # color from set1 palette
        color=sns.color_palette("Set1")[i],
        label=f"Cluster {i}",
        linewidth=2,
    )
ax.legend()
fig.show()
