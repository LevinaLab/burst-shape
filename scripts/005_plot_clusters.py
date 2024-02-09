import pickle
import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src.persistence import get_labels_file, get_burst_folder

# which clustering to plot
n_clusters = 5
col_cluster = f"cluster_{n_clusters}"

burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
clustering_params = "spectral"
labels_params = "004_clustering_labels.pkl"
cv_params = "cv"  # if cv_split is not None, chooses the cross-validation split
cv_split = (
    None  # set to None for plotting the whole clustering, set to int for specific split
)

# load data
with open(
    get_labels_file(
        labels_params,
        clustering_params,
        burst_extraction_params,
        i_split=cv_split,
    ),
    "rb",
) as f:
    clustering = pickle.load(f)
df_bursts = pd.read_pickle(
    os.path.join(
        get_burst_folder(burst_extraction_params),
        f"002_wagenaar_bursts_df{'_' + cv_params if cv_params is not None else ''}.pkl",
    )
)
if cv_split is not None:
    df_bursts = df_bursts[df_bursts[f"cv_{cv_split}_train"]]
for n_clusters_ in clustering.n_clusters:
    df_bursts[f"cluster_{n_clusters_}"] = clustering.labels_[n_clusters_]

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
            color=sns.color_palette("Set1")[i],
            alpha=0.5,
            label=f"Cluster {i}" if j == 0 else None,
        )
ax.legend()
fig.show()

# %% plot burst with real time
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
        bins = np.linspace(
            -50, df_bursts_i.iloc[idx]["time_extend"] - 50, 51, endpoint=True
        )
        bins_mid = (bins[1:] + bins[:-1]) / 2
        ax.plot(
            bins_mid,
            df_bursts_i.iloc[idx]["burst"],
            color=sns.color_palette("Set1")[i],
            alpha=0.5,
            label=f"Cluster {i}" if j == 0 else None,
        )
ax.legend(frameon=False)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Rate [Hz]")
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
